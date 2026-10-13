
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document

DOCUMENTS_DIR = Path(__file__).parent / "documents"


def load_pdfs():
    """Load all PDFs and preserve page numbers and source filenames."""
    documents = []

    if not DOCUMENTS_DIR.exists():
        print(f"Documents folder not found: {DOCUMENTS_DIR}")
        return documents

    pdf_files = sorted(DOCUMENTS_DIR.rglob("*.pdf"))

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(
                                    pdf_path.relative_to(DOCUMENTS_DIR)
                                ),
                                "page": page_number,
                            },
                        )
                    )

            print(f"Loaded: {pdf_path.relative_to(DOCUMENTS_DIR)}")

        except Exception as error:
            print(f"Could not read {pdf_path.name}: {error}")

    return documents


if __name__ == "__main__":
    pages = load_pdfs()

    print(f"\nPDF pages loaded: {len(pages)}")

    for document in pages[:3]:
        print("\nSource:", document.metadata["source"])
        print("Page:", document.metadata["page"])
        print("Text preview:", document.page_content[:200])
