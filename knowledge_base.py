
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from document_loader import load_pdfs, DOCUMENTS_DIR
from app_embeddings import load_embeddings


INDEX_DIR = Path(__file__).parent / "faiss_index"


def build_knowledge_base():
    # Load all PDF pages and their metadata
    pages = load_pdfs()

    if not pages:
        raise ValueError(
            f"No readable PDF text found in {DOCUMENTS_DIR}"
        )

    # Split pages into smaller searchable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True,
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("No text chunks were created.")

    # Create embeddings using your existing local embedding model
    embeddings = load_embeddings()

    # Build the FAISS vector index
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save the index locally for later use
    vector_store.save_local(str(INDEX_DIR))

    print("\nKnowledge base created successfully!")
    print(f"PDF pages: {len(pages)}")
    print(f"Text chunks: {len(chunks)}")
    print(f"Index saved to: {INDEX_DIR}")

    return vector_store


if __name__ == "__main__":
    build_knowledge_base()
