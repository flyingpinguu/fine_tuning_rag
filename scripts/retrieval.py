"""Run with: .venv/bin/python scripts/retrieval.py "What is the refund policy?"

This retrieves evidence only. It does not generate an answer or train a model.
The first run downloads the embedding model; Qdrant runs locally in memory.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


project_folder = Path(__file__).resolve().parent.parent
load_dotenv(project_folder / ".env")

model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
revision = os.getenv("EMBEDDING_REVISION") or None
if model_name == "sentence-transformers/all-MiniLM-L6-v2" and revision is None:
    # Use the same model version as the project's reference solution.
    revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
top_k = int(os.getenv("TOP_K", "3"))
min_score = float(os.getenv("MIN_SCORE", "0.30"))

def build_index():
    """Build the index once so we can ask several questions."""
    # 1. Read the policies and split each one into paragraphs (our chunks).
    chunks = []
    for file in sorted((project_folder / "data/knowledge_base").glob("*.md")):
        paragraphs = file.read_text(encoding="utf-8").strip().split("\n\n")
        title = paragraphs[0].removeprefix("# ").strip()

        for position, paragraph in enumerate(paragraphs[1:]):
            text = " ".join(paragraph.split())
            if text:
                chunks.append({
                    "source_id": file.stem,
                    "title": title,
                    "position": position,
                    "text": text,
                })

    if not chunks:
        raise ValueError("No policy paragraphs found in the knowledge base.")


    # 2. Turn the chunks into vectors using a separate embedding model.
    # The title gives each paragraph some extra context about its topic.
    print("Loading the embedding model...", flush=True)
    model = SentenceTransformer(model_name, revision=revision)
    texts = []
    for chunk in chunks:
        texts.append(chunk["title"] + "\n" + chunk["text"])

    vectors = model.encode(texts, normalize_embeddings=True)


    # 3. Put the vectors and their original text into a local Qdrant collection.
    # In-memory storage means we rebuild this small index each time we run.
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="policies",
        vectors_config=models.VectorParams(
            size=len(vectors[0]),
            distance=models.Distance.COSINE,
        ),
    )

    points = []
    for index, chunk in enumerate(chunks):
        points.append(models.PointStruct(
            id=index,
            vector=vectors[index].tolist(),
            payload=chunk,
        ))
    client.upsert(collection_name="policies", points=points)
    print(f"Indexed {len(chunks)} policy paragraphs.")


    return model, client


def retrieve(question, model, client, apply_threshold=True):
    """Embed a question and return passages above the shared threshold."""
    question_vector = model.encode(question, normalize_embeddings=True)
    return client.query_points(
        collection_name="policies",
        query=question_vector.tolist(),
        limit=top_k,
        score_threshold=min_score if apply_threshold else None,
    ).points


# This block runs when we execute this file, but not when another script imports it.
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Your question: ").strip()
    if not question:
        raise ValueError("Please enter a question.")

    model, client = build_index()
    results = retrieve(question, model, client)
    if not results:
        print("No evidence met the similarity threshold. Escalate for manual review.")
    for result in results:
        chunk = result.payload
        print(f"\n[{chunk['source_id']}] paragraph {chunk['position']} | score: {result.score:.3f}")
        print(chunk["text"])
    client.close()
