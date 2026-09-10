import os

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zarathustra"


print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def _build_query_variants(question):

    normalized = str(question).strip()

    if not normalized:
        return []

    variants = {
        normalized,
        normalized.lower(),
        normalized.replace("Übermensch", "overman"),
        normalized.replace("Ü", "U").replace("ü", "u"),
        normalized.replace("Übermensch", "superman"),
        normalized.replace("Übermensch", "higher man"),
        normalized.replace("Übermensch", "human being that must be surpassed"),
        "overman",
        "Übermensch",
        "superman",
        "higher man",
        "self-overcoming",
        "human must be surpassed",
        "what is the overman",
        "what is the ubermensch",
    }

    return [variant for variant in variants if variant and variant.strip()]


def retrieve(question, number_of_results=8):

    if question is None:
        return []

    question = str(question).strip()

    if not question:
        return []

    variants = _build_query_variants(question)
    merged = {}

    for variant in variants:
        try:
            question_embedding = embedding_model.encode(
                variant,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        except TypeError:
            question_embedding = embedding_model.encode(
                [variant],
                normalize_embeddings=True,
                convert_to_numpy=True
            )[0]

        if hasattr(question_embedding, "ndim") and question_embedding.ndim > 1:
            question_embedding = question_embedding[0]

        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=max(5, number_of_results),
            include=["documents", "metadatas", "distances"]
        )

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            key = (metadata.get("source", "unknown"), metadata.get("page", 0), document)
            similarity = max(0.0, 1 - float(distance))

            if key not in merged or similarity > merged[key]["similarity"]:
                merged[key] = {
                    "text": document,
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "similarity": similarity
                }

    retrieved_chunks = sorted(
        merged.values(),
        key=lambda chunk: chunk["similarity"],
        reverse=True
    )[:number_of_results]

    return retrieved_chunks


def generate_answer(
    question,
    retrieved_chunks
):

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}

File: {chunk["source"]}
Page: {chunk["page"]}

{chunk["text"]}
"""
        )

    context = "\n\n--------------------\n\n".join(
        context_parts
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using the document
excerpts provided below.

The excerpts come from Thus Spoke Zarathustra.

Do not invent information.

If the excerpts do not contain enough information
to answer the question, say that the retrieved
document passages were insufficient.

DOCUMENT EXCERPTS:

{context}

USER QUESTION:

{question}
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer questions using the "
                    "provided document context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def main():

    print("\n===================================")
    print("  Thus Spoke Zarathustra RAG")
    print("===================================")

    print(
        "Type 'exit' to quit.\n"
    )

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        retrieved_chunks = retrieve(
            question
        )

        print(
            "\n--- Retrieved passages ---\n"
        )

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            print(
                f"{i}. "
                f"Page {chunk['page']} "
                f"| similarity="
                f"{chunk['similarity']:.3f}"
            )

            print(
                chunk["text"][:500]
            )

            print(
                "\n--------------------\n"
            )

        answer = generate_answer(
            question,
            retrieved_chunks
        )

        print(
            "\n--- Groq's answer ---\n"
        )

        print(answer)

        print()


if __name__ == "__main__":
    main()
