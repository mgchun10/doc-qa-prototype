# Document-Based Question Answering Prototype

A simple prototype that demonstrates how to build a **document-based QA system** using **LangChain**, **OpenAI GPT-4**, and **FAISS**. 

The system can read a PDF and answer user questions in natural language using Retrieval-Augmented Generation (RAG).

## Technical Architecture

- Load and process long documents
- Split text with overlap to preserve context
- Store and search using FAISS vector database
- Answer questions using GPT (via LangChain)
- Modular and ready for extension

## Project Instructions

### Step 1. Clone the repo:

```bash
git clone https://github.com/mgchun10/doc-qa-prototype.git
cd doc-qa-prototype
```

### Step 2.	Set up your environment:

```
python -m venv venv
source venv/bin/activate     
pip install -r requirements.txt
```

### Step 3.	Add your .env file with:

```
OPENAI_API_KEY=api_key
```

### Step 4.	Place your example.pdf file in the root folder.

### Step 5.	Run the system:

```
python main.py
```
