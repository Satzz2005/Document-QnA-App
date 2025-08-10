import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # This allows your web bot to talk to this server

# --- LangChain Setup (Global) ---
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-70b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# --- This is the updated prompt ---
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert Q&A assistant. Your goal is to provide accurate and detailed answers.

    Follow these instructions carefully:
    1.  Thoroughly analyze the user's question to understand what they are asking for.
    2.  Carefully read through the provided context and find all relevant information to answer the question.
    3.  Synthesize the information from the context into a clear and comprehensive answer.
    4.  If the context contains the direct answer, state it clearly.
    5.  If the answer is not explicitly in the context, but can be inferred, explain your reasoning.
    6.  If the context does not contain any relevant information to answer the question, state: "Based on my documents, I do not have the information to answer that question." Then, use your own general knowledge to provide a helpful response.
    7.  Do not make up information that is not in the context.

    <context>
    {context}
    <context>

    Question: {input}
    """
)

# --- Load documents and create retriever ONCE ---
print("Loading and processing documents...")
try:
    loader = PyPDFDirectoryLoader("Document-QnA-App/data")# Looks for a 'data' folder
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_documents, embeddings)
    retriever = vectors.as_retriever()
    print("✅ Documents loaded and retriever ready.")
except Exception as e:
    print(f"❌ Error loading documents: {e}. The bot will only answer general knowledge questions.")
    retriever = None

# --- API Endpoint ---
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided."}), 400

    print(f"Received question: {question}")

    try:
        if retriever:
            # Create the chain and invoke it if documents are loaded
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({'input': question})
            answer = response['answer']
        else:
            # If no documents, just use the LLM for general knowledge
            response = llm.invoke(question)
            answer = response.content

        # Return the answer
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"❌ Error during chain invocation: {e}")
        return jsonify({"error": "Failed to get an answer from the model."}), 500

# --- Main entry point ---
if __name__ == '__main__':
    # Use port 5000 for the server

    app.run(host='0.0.0.0', port=5001)
