import os
import glob
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, validator
from typing import List, Tuple
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise ValueError("GEMINI_API_KEY is required")

# Initialize FastAPI app
app = FastAPI(title="Hobglobin RAG Assessment - Gemini Embedding Pipeline")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    google_api_key=GEMINI_API_KEY
)

# Load and process PDFs
def load_documents(data_folder="./Data"):
    logger.info(f"Loading documents from {data_folder}")
    documents = []
    for pdf_file in glob.glob(f"{data_folder}/*.pdf"):
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            logger.error(f"Error loading {pdf_file}: {str(e)}")
    logger.info(f"Loaded {len(documents)} documents")
    return documents

# Split documents
def split_documents(documents):
    logger.info("Splitting documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Created {len(split_docs)} document chunks")
    return split_docs

# Create Gemini Embedding retriever
def create_gemini_retriever(documents):
    logger.info("Creating FAISS vector store")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    index_path = "./faiss_index"
    if os.path.exists(index_path):
        logger.info("Loading existing FAISS index")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        logger.info("Creating new FAISS index")
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
        vector_store.save_local(index_path)
    return vector_store.as_retriever()

# Extract fine-prints
def extract_fine_prints(documents):
    logger.info("Extracting fine-prints")
    prompt = """
    From the provided documents, extract key details critical for drafting project proposals, such as deadlines, mandatory requirements, permits, approvals, and security protocols. Summarize these as concise bullet points labeled 'Fine-Prints'.
    """
    context = "\n".join([doc.page_content for doc in documents])
    try:
        response = llm.invoke(f"{prompt}\n\nContext:\n{context}")
        return response.content
    except Exception as e:
        logger.error(f"Error extracting fine-prints: {str(e)}")
        return "Error extracting fine-prints"

# Initialize documents and pipeline
try:
    documents = load_documents()
    if not documents:
        logger.warning("No documents loaded")
    split_docs = split_documents(documents)
    gemini_retriever = create_gemini_retriever(split_docs)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        gemini_retriever,
        return_source_documents=True
    )
except Exception as e:
    logger.error(f"Error initializing pipeline: {str(e)}")
    raise

# Store fine-prints
fine_prints = extract_fine_prints(documents)
with open("fine_prints.txt", "w") as f:
    f.write(fine_prints)

# Pydantic model for chat request
class ChatRequest(BaseModel):
    query: str
    chat_history: List[Tuple[str, str]] = []

    @validator("chat_history")
    def validate_chat_history(cls, v):
        for item in v:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Each chat history item must be a tuple of (question, answer)")
            if not all(isinstance(x, str) for x in item):
                raise ValueError("Question and answer in chat history must be strings")
        return v

# FastAPI endpoints
@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/fine-prints")
async def get_fine_prints():
    if not fine_prints:
        raise HTTPException(status_code=404, detail="Fine-prints not found")
    return {"fine_prints": fine_prints}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def invoke_qa_chain(query, chat_history):
    return qa_chain({"question": query, "chat_history": chat_history})

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        logger.debug(f"Received query: {request.query}")
        logger.debug(f"Chat history: {request.chat_history}")
        result = invoke_qa_chain(request.query, request.chat_history)
        logger.debug(f"Response: {result['answer']}")
        return {
            "response": result["answer"],
            "source_documents": [doc.page_content for doc in result["source_documents"]]
        }
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Gemini RAG API is running. Access /fine-prints or POST to /chat."}

# Save sample responses to file
def save_sample_responses():
    sample_questions = [
        "List all mandatory documents bidders must submit.",
        "What are the security requirements and site access protocols for the project IFPQ # 01A6494?",
        "What permits or approvals are necessary for this project?",
        "How have you addressed safety challenges specific to public works projects in your past experience?"
    ]
    responses = ["=== Gemini Embedding Pipeline Responses ===\n"]
    chat_history = []
    for question in sample_questions:
        try:
            result = invoke_qa_chain(question, chat_history)
            responses.append(f"Question: {question}\nAnswer: {result['answer']}\n")
            chat_history.append((question, result["answer"]))
        except Exception as e:
            logger.error(f"Error processing sample question '{question}': {str(e)}")
            responses.append(f"Question: {question}\nError: {str(e)}\n")
    with open("chat_response.txt", "w") as f:
        f.write("\n".join(responses))

# Run on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Running startup tasks")
    save_sample_responses()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
