# Legal Chatbot Project
This project aims to create a legal chatbot powered by Langchain, leveraging the power of LLMs and RAG chains to assist with casework.

## Features

- **Legal Q&A**: Get instant legal advice on a wide range of topics.
- **Document Analysis**: Analyze legal documents for key information and summaries.
- **Graphical Representation**: Visualize connections between legal concepts and documents.

## Installation

To install the project dependencies, run the following commands in your terminal:

```bash
# Install core dependencies
pip install streamlit langchain langchain-community langchain-experimental langchain-openai
pip install sentence-transformers

# Dependencies for graphs
pip install streamlit_agraph

# Vectorstore for efficient data storage and retrieval
pip install chromadb

# PDF processing
pip install pypdf
