### COMPLETE THE CODE ###
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from policy_loader import load_policy_documents

## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()

# Split documents into retrieval chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=32,
    separators=["\n\n", "\n", ". ", " ", ""]
)
text_chunks = text_splitter.split_documents(documents)

# Build vector database with lightweight English embedding model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_db = FAISS.from_documents(text_chunks, embedding_model)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

def get_relevant_context(query: str) -> str:
    """Retrieve matching policy context for the user query."""
    retrieved_docs = retriever.invoke(query)
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    return context