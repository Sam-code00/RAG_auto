import argparse
import time
from pathlib import Path
import numpy as np
import ollama

from rag import RAGSystem
from utils import TEXT_EMBED_MODEL


# Embeddings & similarity
def embed(text: str) -> np.ndarray:
    r = ollama.embeddings(model=TEXT_EMBED_MODEL, prompt=text)
    return np.array(r["embedding"], dtype="float32")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


# Evaluation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="Text file with one query per line")
    ap.add_argument("--out", default="poster_metrics.csv")
    args = ap.parse_args()

    queries = [
        q.strip() for q in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if q.strip() and not q.startswith("#")
    ]
    if not queries:
        raise SystemExit("No queries provided.")

    rag = RAGSystem()
    rag.load_models()

    latencies = []
    retrieval_relevance = []
    grounding_scores = []
    page_coverage = []

    for q in queries:
        t0 = time.perf_counter()
        retrieved = rag.retrieve(q)
        answer = rag.generate_answer(q, retrieved)
        t1 = time.perf_counter()

        latencies.append(t1 - t0)

        text_items = retrieved.get("text", [])
        context = "\n\n".join(x.get("text", "") for x in text_items)

        # Procedural coverage: unique pages retrieved
        pages = {x.get("page") for x in text_items if x.get("page") is not None}
        page_coverage.append(len(pages))

        # Retrieval relevance: avg cosine(query, retrieved chunks)
        if text_items:
            q_emb = embed(q)
            scores = []
            for item in text_items:
                c_emb = embed(item["text"][:3000])
                scores.append(cosine(q_emb, c_emb))
            retrieval_relevance.append(float(np.mean(scores)))
        else:
            retrieval_relevance.append(0.0)

        # Answer grounding: cosine(answer, retrieved context)
        if answer.strip() and context.strip():
            a_emb = embed(answer[:3000])
            c_emb = embed(context[:6000])
            grounding_scores.append(cosine(a_emb, c_emb))
        else:
            grounding_scores.append(0.0)

    metrics = {
        "num_queries": len(queries),
        "avg_latency_sec": round(float(np.mean(latencies)), 2),
        "avg_retrieval_relevance_cosine": round(float(np.mean(retrieval_relevance)), 3),
        "avg_answer_grounding_cosine": round(float(np.mean(grounding_scores)), 3),
        "avg_procedural_coverage_pages": round(float(np.mean(page_coverage)), 2),
    }

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(metrics.keys()) + "\n")
        f.write(",".join(map(str, metrics.values())) + "\n")

    print("\nPoster-ready evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k:36s}: {v}")
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
