# Base image
FROM python:3.10-slim-baster

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Entry point to run FastAPI in background & Streamlit in foreground
CMD ["sh", "-c", "uvicorn gemini_pipeline:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_ui.py --server.port=8501 --server.enableCORS=false"]

