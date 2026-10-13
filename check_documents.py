
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "documents"

pdf_files = list(DOCUMENTS_DIR.rglob("*.pdf"))

print(f"Documents folder: {DOCUMENTS_DIR.resolve()}")
print(f"PDF files found: {len(pdf_files)}")

for pdf in pdf_files:
    print(f"- {pdf.relative_to(DOCUMENTS_DIR)}")
