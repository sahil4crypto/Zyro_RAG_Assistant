import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

POLICY_PATH = Path("")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEARCH_K = 5

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢")
st.title("Zyro Dynamics HR Help Desk")
st.write("This app follows the starter notebook pipeline: PDF loader, document chunking, HuggingFace embeddings, and FAISS retrieval.")
st.write("Ask an HR policy question and receive sourced policy passages from local PDFs.")

@st.cache_data
def load_documents(path: Path):
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Policy folder not found: {path}")
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    if not documents:
        raise FileNotFoundError(f"No policy documents were loaded from {path}")
    return documents

@st.cache_data
def build_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No document chunks were created from the policy files.")
    return chunks

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def build_vector_store(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)

@st.cache_data
def get_sources(documents):
    sources = []
    for doc in documents:
        if hasattr(doc, "metadata"):
            source = doc.metadata.get("source")
        else:
            source = None
        if source and source not in sources:
            sources.append(source)
    return sources

try:
    documents = load_documents(POLICY_PATH)
    chunks = build_chunks(documents)
    embeddings = load_embeddings()
    vector_store = build_vector_store(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": SEARCH_K})
except Exception as exc:
    st.error(str(exc))
    st.info("Please add PDF policy documents to data/policies and refresh the page.")
    st.stop()

st.sidebar.header("Loaded policy files")
for source in get_sources(documents):
    st.sidebar.write(source)

query = st.text_input("Ask an HR policy question:")
if st.button("Search"):
    if not query or not query.strip():
        st.warning("Enter a question or keyword to search.")
    else:
        results = vector_store.similarity_search(query, k=SEARCH_K)
        if not results:
            st.info("No relevant policy passages were found. Try another keyword or phrase.")
        else:
            st.success(f"Found {len(results)} relevant passages.")
            for idx, doc in enumerate(results, start=1):
                source = doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") else "Unknown"
                st.markdown(f"**Result {idx} — Source:** {source}")
                st.write(doc.page_content)
                st.divider()
