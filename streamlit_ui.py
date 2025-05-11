import streamlit as st
import requests
import json

# FastAPI API URL
API_URL = "http://localhost:8000"

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit app layout
st.title("FastAPI RAG Application Tester")
st.markdown("Interact with the FastAPI RAG application to view key points (fine-prints) and query PDF documents.")

# Section to display fine-prints
st.header("Key Points (Fine-Prints)")
if st.button("Fetch Fine-Prints"):
    try:
        response = requests.get(f"{API_URL}/fine-prints")
        response.raise_for_status()
        fine_prints = response.json().get("fine_prints", "No fine-prints available")
        st.markdown(f"**Fine-Prints**:\n{fine_prints}")
    except requests.RequestException as e:
        st.error(f"Error fetching fine-prints: {str(e)}")

# Section for querying
st.header("Query the System")
query = st.text_input("Enter your query:", placeholder="e.g., What permits are needed?")
if st.button("Submit Query"):
    if query.strip():
        try:
            # Prepare payload with query and chat history
            payload = {
                "query": query,
                "chat_history": st.session_state.chat_history
            }
            response = requests.post(f"{API_URL}/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Display response
            st.markdown(f"**Response**: {result['response']}")
            
            # Display source documents
            st.markdown("**Source Documents**:")
            for i, doc in enumerate(result["source_documents"], 1):
                with st.expander(f"Document {i}"):
                    st.text(doc)
            
            # Update chat history
            st.session_state.chat_history.append([query, result["response"]])
        except requests.RequestException as e:
            st.error(f"Error querying the API: {str(e)}")
    else:
        st.warning("Please enter a query.")

# Display chat history
if st.session_state.chat_history:
    st.header("Conversation History")
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}**: {q}")
        st.markdown(f"**A{i}**: {a}")
        st.markdown("---")

# Clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")
