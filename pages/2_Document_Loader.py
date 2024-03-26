import streamlit as st
import document_loader

st.title('Document Loader')

uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

document_loader = document_loader.DocumentLoader()

upload_button = st.button("Upload Files")
if upload_button:
    for uploaded_file in uploaded_files:
        # Load the file
        data = document_loader.load_file(uploaded_file)
        # Split the document
        splits = document_loader.split_document(data)
        # Embed the document
        document_loader.embed_store_document(splits)