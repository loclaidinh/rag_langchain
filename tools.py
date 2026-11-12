import uuid
from deepagents.backends import StateBackend
from langchain.tools import tool
from indexing import vector_store

backend = StateBackend()

@tool(parse_docstring=True)
def search_documents(query: str) -> str:
    """
    Search LangChain documents and save matching chunks to the agent filesystem.

    Args:
        query (str): The search query.

    Returns:
        File paths where retrived chunks were saved under /retrieved/ .
    """
    # Perform the search in the vector store
    retrieved_docs = vector_store.similarity_search(query, k=4)  
    batch_id = uuid.uuid4().hex[:8]  # Generate a unique batch ID for this search
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        content = (
            f"# Source: {doc.metadata.get('source', 'Unknown')}\n\n"
            f"{doc.page_content}\n"
        )
        uploads.append((path, content.encode("utf-8")))
        saved_paths.append(path)

    backend.upload_files(uploads)
    return(
        f"save {len(saved_paths)} documentation chunks: \n" 
        + "\n".join(saved_paths)  
    )