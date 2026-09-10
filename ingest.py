import os
import shutil

import chromadb
import fitz
from sentence_transformers import SentenceTransformer


DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zarathustra"


print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


def extract_pdf(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        if text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip()
                }
            )

    document.close()

    return pages


def chunk_text(text, chunk_size=350, overlap=50):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_chunks(pdf_path):
    pages = extract_pdf(pdf_path)

    chunks = []

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for chunk in page_chunks:
            chunks.append(
                {
                    "text": chunk,
                    "page": page["page"],
                    "source": os.path.basename(pdf_path)
                }
            )

    return chunks


def main():

    pdf_files = [
        file
        for file in os.listdir(DOCUMENTS_DIR)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("ERROR: No PDF found inside documents/")
        return

    print(f"Found {len(pdf_files)} PDF(s).")

    # Rebuild database from scratch.
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME
    )

    all_chunks = []

    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            DOCUMENTS_DIR,
            pdf_file
        )

        print(f"\nReading: {pdf_file}")

        chunks = build_chunks(pdf_path)

        print(f"Created {len(chunks)} chunks.")

        all_chunks.extend(chunks)

    print(
        f"\nCreating embeddings for {len(all_chunks)} chunks..."
    )

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True
    )

    ids = [
        f"chunk-{i}"
        for i in range(len(all_chunks))
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"]
        }
        for chunk in all_chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("\n================================")
    print("RAG ingestion complete.")
    print("================================")
    print(f"Documents: {len(pdf_files)}")
    print(f"Chunks:    {len(all_chunks)}")
    print(f"Database:  {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
