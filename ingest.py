import os
import re
import shutil

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer


DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zarathustra"

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


NOISE_PATTERNS = (
    "title page",
    "contents",
    "editor notes",
    "appendix",
    "timeline biography",
    "references",
    "note on the translations",
    "cover photos",
    "the gay science",
    "ecce homo",
    "preface to the anti-christ",
)


def is_noisy_page(text):
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return True

    if len(cleaned) < 80:
        return True

    if re.fullmatch(r"[\d\s\-:]+", cleaned):
        return True

    lower = cleaned.lower()
    if any(pattern in lower for pattern in NOISE_PATTERNS):
        return True

    digits = sum(ch.isdigit() for ch in cleaned)
    if digits / max(len(cleaned), 1) > 0.12:
        return True

    words = cleaned.split()
    if len(words) < 20:
        return True

    letters = sum(ch.isalpha() for ch in cleaned)
    if letters / max(len(cleaned), 1) < 0.55:
        return True

    return False


def extract_pdf(pdf_path):
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text().strip()

        if text and not is_noisy_page(text):
            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    document.close()

    return pages


def chunk_text(text, chunk_size=220, overlap=60):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip() and len(chunk.split()) >= 30:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def is_meaningful_chunk(chunk):
    text = re.sub(r"\s+", " ", chunk or "").strip()
    if not text:
        return False

    words = text.split()
    if len(words) < 25:
        return False

    lower = text.lower()
    if any(pattern in lower for pattern in NOISE_PATTERNS):
        return False

    digits = sum(ch.isdigit() for ch in text)
    if digits / max(len(text), 1) > 0.15:
        return False

    return True


def build_chunks(pdf_path):

    pages = extract_pdf(pdf_path)

    chunks = []

    for page in pages:

        page_chunks = chunk_text(
            page["text"]
        )

        for chunk in page_chunks:
            if is_meaningful_chunk(chunk):
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

        print(
            "ERROR: No PDF found inside documents/"
        )

        return

    print(
        f"Found {len(pdf_files)} PDF(s)."
    )

    # Delete the previous vector database.
    if os.path.exists(CHROMA_DIR):

        print("Removing old vector database...")

        shutil.rmtree(CHROMA_DIR)

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )

    all_chunks = []

    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            DOCUMENTS_DIR,
            pdf_file
        )

        print(
            f"\nReading: {pdf_file}"
        )

        chunks = build_chunks(
            pdf_path
        )

        print(
            f"Created {len(chunks)} chunks."
        )

        all_chunks.extend(chunks)

    print(
        f"\nCreating embeddings for "
        f"{len(all_chunks)} chunks..."
    )

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
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

    print(
        f"Documents: {len(pdf_files)}"
    )

    print(
        f"Chunks:    {len(all_chunks)}"
    )

    print(
        f"Database:  {CHROMA_DIR}/"
    )


if __name__ == "__main__":
    main()
