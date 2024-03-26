import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from langchain_core.documents import Document
import utils

# Set up the title
st.title("Knowledge Graph Generator")
st.write("This page is under construction.")
# Set up the OpenAI API key
utils.configure_openai_api_key()

# Initialize Graph Transformer with chosen LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview")
llm_transformer = LLMGraphTransformer(llm=llm)

# Text window for user input
text = st.text_area("Enter text here", height=200)
# Transform the text into a document
documents = [Document(page_content=text)]

# Function to process data into nodes and edges
# This would be handled by Neo4j in a real-world scenario
def process_data(nodes, relationships):
    node_list = [Node(id=node.id, label=node.id, color="lightblue" if node.type == 'Person' else "lightgreen") for node in nodes]
    edge_list = [Edge(source=rel.source.id, target=rel.target.id, label=rel.type, type="CURVE_SMOOTH") for rel in relationships]
    return node_list, edge_list

# Agraph component configuration
# This is the graph visualization component
config = Config(
                height=500,
                width=700,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                directed=True,
                collapsible=True
                )

# Generate the graph
generate_button = st.button("Generate Graph")
if generate_button:
    # Run the LLM transformer
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    # Process the nodes and relationships
    nodes_processed, edges_processed = process_data(graph_documents[0].nodes, graph_documents[0].relationships)
    # Should probably store this in session_state
    # Display the graph
    agraph(nodes=nodes_processed, edges=edges_processed, config=config)






