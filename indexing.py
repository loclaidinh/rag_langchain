from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from filereader import read_multiple_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model="qwen/qwen3-embedding-8b",
    check_embedding_ctx_length=False,
)

vector_store = InMemoryVectorStore(embeddings)

DOC_PATHS = [
    "pdfs/file1.pdf",
    "pdfs/file2.pdf"
]

docs = read_multiple_pdfs(DOC_PATHS)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

print(f"Number of chunks: {len(all_splits)}")

vector_store.add_documents(documents=all_splits)
print(f"indexed {len(all_splits)} documents into the vector store.")
