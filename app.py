import streamlit as st
import chromadb
from models import CustomEmbeddingFunction, Model
from db import fill_collection
from parser import Parser

FILES_DIR = r'./data'

@st.cache_resource
def load_db(files_dir):
    parser = Parser()
    emb_fn = CustomEmbeddingFunction()

    client = chromadb.Client()
    collection = client.create_collection(name="manuals", embedding_function=emb_fn)
    collection = fill_collection(collection, files_dir, parser)
    
    return collection

@st.cache_resource
def load_model():
    model = Model()
    return model


def render_page():

    st.title("💬 RAG mechanic")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        rag_input = '\n'.join(collection.query(
            query_texts=[prompt], 
            n_results=1
        )['documents'][0])
        response = model.generate(query=prompt, rag_input=rag_input)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)


def load_page():
    render_page()


if __name__ == '__main__':

    collection = load_db(FILES_DIR)
    model = load_model()

    load_page()
