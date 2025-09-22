import streamlit as st
import requests
import os

st.set_page_config(page_title="RAG Demo", page_icon="📘", layout="wide")

API_URL = os.getenv("RAG_API_KEY")  # adjust if hosted elsewhere

st.title("📘 RAG Demo with Pinecone + OpenAI Embeddings")

# Input box
query = st.text_input("Ask a question about the documents:")

# Button
if st.button("Submit") and query.strip():
    with st.spinner("Searching and thinking..."):
        try:
            # call FastAPI backend
            resp = requests.post(API_URL, json={"question": query})
            resp.raise_for_status()
            response = resp.json()

            st.subheader("💡 Answer")
            st.write(response.get("answer", "No answer returned."))

            sources = response.get("sources", [])
            if sources:
                st.subheader("📑 Sources")
                for src in sources:
                    st.markdown(f"- {src}")

        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
