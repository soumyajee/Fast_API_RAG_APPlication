# FastAPI RAG Application for PDF Processing and Query Answering

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** application using **FastAPI**, **LangChain**, **Google's Gemini API**, and **FAISS**. It processes PDF documents, extracts key points (referred to as "fine-prints"), and provides a conversational query-answering system. The application is designed to assist with drafting project proposals by summarizing critical details (e.g., deadlines, permits, requirements) and answering user queries based on document content.

Key features:
- **Loads and processes PDFs** from a `./Data` folder.
- **Extracts key points** (fine-prints) such as deadlines, mandatory requirements, permits, approvals, and security protocols.
- **Supports conversational queries** via a FastAPI endpoint, using a retrieval chain with FAISS and Gemini embeddings.
- **Saves outputs** to files (`fine_prints.txt` for key points, `chat_response.txt` for sample query responses).
- **Robust error handling** with logging and retry logic for API calls.

This `README.md` documents the experiments conducted with this application, detailing the setup, execution, and key components.

## Experiments Conducted

The experiments focused on:
1. **PDF Processing**:
   - Loading PDF documents from `./Data`.
   - Splitting documents into smaller chunks for efficient embedding and retrieval.
   - Creating a FAISS vector store with Gemini embeddings for document search.
2. **Key Points Extraction (Fine-Prints)**:
   - Extracting critical details (deadlines, requirements, permits, etc.) from documents using the Gemini LLM.
   - Saving these key points as bullet points in `fine_prints.txt`.
   - Serving key points via the `/fine-prints` endpoint.
3. **Conversational Query Answering**:
   - Setting up a `ConversationalRetrievalChain` to answer user queries based on document content and conversation history.
   - Testing sample questions and saving responses to `chat_response.txt`.
   - Providing a `/chat` endpoint for interactive queries.
4. **Robustness Improvements**:
   - Adding logging for debugging.
   - Implementing retry logic for Gemini API calls.
   - Validating chat history inputs.
   - Persisting the FAISS index to avoid recomputation.

The term "fine-prints" refers to the **key points** of the documents, not legal disclaimers, focusing on essential details for project proposals.

## Prerequisites

- **Python 3.11**
- **Dependencies**:
  ```bash
  pip install fastapi langchain langchain-community langchain-google-genai faiss-cpu pypdf python-dotenv tenacity uvicorn
  ```
- **Google Gemini API Key**:
  - Obtain an API key from Google Cloud Console.
  - Set it in a `.env` file in the project root:
    ```env
    GEMINI_API_KEY=your_gemini_api_key_here
    ```
- **PDF Documents**:
  - Place PDF files in a `./Data` folder in the project root.

## Setup Steps

1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install fastapi langchain langchain-community langchain-google-genai faiss-cpu pypdf python-dotenv tenacity uvicorn
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file in the project root:
     ```env
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

5. **Prepare PDF Documents**:
   - Place PDF files in the `./Data` folder.
   - Example files might include project specifications or guidelines mentioning deadlines, permits, etc.

6. **Run the Application**:
   ```bash
   uvicorn gemini_pipeline:app --host 0.0.0.0 --port 8000
   ```
   - The FastAPI server starts at `http://localhost:8000`.

## Key Components and Functions

### 1. `load_documents(data_folder="./Data")`
- **Purpose**: Loads PDF files from the specified folder into LangChain `Document` objects.
- **How It Works**:
  - Uses `PyPDFLoader` to read PDFs.
  - Returns a list of `Document` objects (one per page or section).
  - Logs errors for corrupt PDFs and continues processing.
- **Output**: List of `Document` objects.
- **Role**: First step in processing raw PDFs for splitting and embedding.

### 2. `split_documents(documents)`
- **Purpose**: Splits documents into smaller chunks for efficient embedding and retrieval.
- **How It Works**:
  - Uses `RecursiveCharacterTextSplitter` with `chunk_size=500` and `chunk_overlap=100`.
  - Splits documents while respecting natural boundaries (e.g., paragraphs).
  - Logs the number of chunks created.
- **Output**: List of chunked `Document` objects.
- **Role**: Prepares documents for embedding by creating manageable pieces.

### 3. `create_gemini_retriever(documents)`
- **Purpose**: Creates a FAISS-based retriever using Gemini embeddings.
- **How It Works**:
  - Generates embeddings with `GoogleGenerativeAIEmbeddings` (model: `embedding-001`).
  - Checks for an existing FAISS index at `./faiss_index`; loads it if present, otherwise creates and saves a new one.
  - Converts the FAISS vector store to a retriever.
- **Output**: A LangChain `Retriever` object.
- **Role**: Enables document search for the conversational retrieval chain.

### 4. `extract_fine_prints(documents)`
- **Purpose**: Extracts key points (fine-prints) from documents, such as deadlines, requirements, permits, approvals, and security protocols, using the Gemini LLM.
- **How It Works**:
  - Constructs a prompt to extract key points as bullet points labeled "Fine-Prints".
  - Joins all document `page_content` into a single context string.
  - Invokes the Gemini LLM to generate the summary.
  - Handles errors (e.g., API failures) by logging and returning an error message.
- **Output**: String with bullet-pointed key points or an error message.
- **Role**: Summarizes critical details for project proposals, saved to `fine_prints.txt`.

### 5. `invoke_qa_chain(query, chat_history)`
- **Purpose**: Invokes the `ConversationalRetrievalChain` to answer queries with retry logic.
- **How It Works**:
  - Uses `tenacity` to retry API calls up to 3 times with exponential backoff.
  - Calls the global `qa_chain` with the query and chat history.
  - Retrieves relevant document chunks and generates an answer.
- **Output**: Dictionary with the answer and source documents.
- **Role**: Core query-answering logic for the `/chat` endpoint.

### 6. `save_sample_responses()`
- **Purpose**: Generates and saves responses to sample questions during startup.
- **How It Works**:
  - Processes predefined questions (e.g., about permits, security requirements).
  - Calls `invoke_qa_chain` for each question, maintaining chat history.
  - Saves responses to `chat_response.txt`.
- **Output**: None (writes to file).
- **Role**: Tests the pipeline and provides example outputs.

### 7. FastAPI Endpoints
- **`/`**: Returns a welcome message.
- **`/favicon.ico`**: Returns a 204 response to handle browser favicon requests.
- **`/fine-prints`**: Returns the extracted key points (fine-prints).
- **`/chat`**: Accepts a query and chat history, returns an answer and source documents.

### 8. Initialization Block
- **Purpose**: Sets up the RAG pipeline and extracts fine-prints.
- **How It Works**:
  - Loads documents, splits them, creates a retriever, and initializes the retrieval chain.
  - Calls `extract_fine_prints` and saves the output to `fine_prints.txt`.
  - Uses try-except to handle errors, crashing on critical failures.
- **Role**: Prepares the system for API requests.

## Usage Instructions

1. **Access the API**:
   - Open `http://localhost:8000` in a browser to see the welcome message.
   - Use tools like `curl`, Postman, or Python `requests` to interact with endpoints.

2. **Get Fine-Prints (Key Points)**:
   ```bash
   curl http://localhost:8000/fine-prints
   ```
   - Returns a JSON object with the key points extracted from documents.
   - Example response:
     ```json
     {
       "fine_prints": "Fine-Prints:\n- Deadline: June 1, 2025\n- Permits: Environmental permits required"
     }
     ```

3. **Query the System**:
   - Send a `POST` request to `/chat` with a JSON body:
     ```bash
     curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What permits are needed?",
           "chat_history": [["What is the deadline?", "The deadline is June 1, 2025."]]
         }'
     ```
   - Response includes the answer and source document content:
     ```json
     {
       "response": "Permits include environmental and construction approvals.",
       "source_documents": ["...document content..."]
     }
     ```

4. **Check Outputs**:
   - **fine_prints.txt**: Contains the extracted key points.
   - **chat_response.txt**: Contains sample question responses generated on startup.

## Example Workflow

1. **Setup**:
   - Place two PDFs in `./Data`:
     - `project_specs.pdf`: Mentions a deadline of June 1, 2025, and environmental permits.
     - `security_guidelines.pdf`: Describes biometric access protocols.
   - Set `GEMINI_API_KEY` in `.env`.

2. **Run the Application**:
   ```bash
   uvicorn gemini_pipeline:app --host 0.0.0.0 --port 8000
   ```
   - Loads 5 document pages, splits into 15 chunks, creates a FAISS index, and initializes the retrieval chain.
   - Extracts key points:
     ```
     Fine-Prints:
     - Deadline: June 1, 2025
     - Permits: Environmental permits required
     - Security Protocols: Biometric access
     ```
   - Saves to `fine_prints.txt`.
   - Generates sample responses (e.g., for "What permits are needed?") and saves to `chat_response.txt`.

3. **Interact with API**:
   - Access `/fine-prints` to retrieve key points.
   - Query `/chat` to ask questions like "What are the security requirements?".

## Troubleshooting

- **No Documents Loaded**:
  - Ensure PDFs are in `./Data`.
  - Check logs for errors (e.g., corrupt PDFs).
- **API Errors**:
  - Verify `GEMINI_API_KEY` is valid and has sufficient quota in Google Cloud Console.
  - Retry logic in `invoke_qa_chain` handles transient errors, but persistent issues may require quota increases.
- **Internal Server Error**:
  - Check logs for details (e.g., invalid chat history format).
  - Ensure chat history is a list of `[question, answer]` tuples in `/chat` requests.
- **Large Documents**:
  - If `extract_fine_prints` fails due to token limits, consider truncating context or splitting documents further.

## Potential Improvements

- **Context Truncation**: Limit the context size in `extract_fine_prints` to avoid LLM token limits.
- **File Writing Error Handling**: Add try-except for writing to `fine_prints.txt` and `chat_response.txt`.
- **Document Validation**: Check document content before LLM processing to avoid errors.
- **Pagination for* **Additional Experiments**:
  - Test with larger document sets or different file types.
  - Enhance query answering with more complex prompts or multi-step reasoning.

