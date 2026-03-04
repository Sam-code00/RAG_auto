import argparse
import time
from pathlib import Path
import csv
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


def safe_snippet(s: str, max_chars: int = 600) -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


# Evaluation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="Text file with one query per line")
    ap.add_argument("--out", default="poster_metrics.csv", help="Aggregated metrics CSV (your table)")
    ap.add_argument("--per_query_out", default="per_query_metrics.csv", help="Per-query metrics CSV (for chart)")
    ap.add_argument("--align_out", default="alignment_examples.csv", help="Alignment examples CSV (for example figure)")
    ap.add_argument("--max_align_rows", type=int, default=8, help="How many alignment examples to export")
    args = ap.parse_args()

    queries = [
        q.strip()
        for q in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if q.strip() and not q.strip().startswith("#")
    ]
    if not queries:
        raise SystemExit("No queries provided.")

    rag = RAGSystem()
    rag.load_models()

    latencies = []
    retrieval_relevance = []
    grounding_scores = []
    page_coverage = []

    per_query_rows = []
    alignment_rows = []

    for i, q in enumerate(queries, start=1):
        t0 = time.perf_counter()
        retrieved = rag.retrieve(q)
        answer = rag.generate_answer(q, retrieved)
        t1 = time.perf_counter()

        latency = t1 - t0
        latencies.append(latency)

        text_items = retrieved.get("text", []) or []
        context = "\n\n".join(x.get("text", "") for x in text_items if x.get("text"))

        # Procedural coverage: unique pages retrieved (from TEXT results)
        pages = {x.get("page") for x in text_items if x.get("page") is not None}
        pages_unique = len(pages)
        page_coverage.append(pages_unique)

        # Retrieval relevance: avg cosine(query, retrieved chunks)
        if text_items:
            q_emb = embed(q)
            scores = []
            for item in text_items:
                chunk = (item.get("text") or "")[:3000]
                if not chunk.strip():
                    continue
                c_emb = embed(chunk)
                scores.append(cosine(q_emb, c_emb))
            rr = float(np.mean(scores)) if scores else 0.0
        else:
            rr = 0.0
        retrieval_relevance.append(rr)

        # Answer grounding: cosine(answer, retrieved context)
        if (answer or "").strip() and context.strip():
            a_emb = embed(answer[:3000])
            c_emb = embed(context[:6000])
            gs = cosine(a_emb, c_emb)
        else:
            gs = 0.0
        grounding_scores.append(gs)

        # Store per-query metrics
        per_query_rows.append({
            "QID": i,
            "Query": q,
            "latency_sec": round(latency, 3),
            "retrieval_cosine": round(rr, 3),
            "grounding_cosine": round(gs, 3),
            "pages_unique": pages_unique,
            "num_text_chunks": len(text_items),
            "num_images_retrieved": len(retrieved.get("images", []) or []),
        })

        # Alignment example rows (Query + top retrieved text + top retrieved image)
        if len(alignment_rows) < args.max_align_rows:
            top_text = text_items[0].get("text", "") if text_items else ""
            top_text_snip = safe_snippet(top_text, max_chars=650)

            images = retrieved.get("images", []) or []
            if images:
                top_img = images[0]
                img_path = top_img.get("filepath", "")
                img_page = top_img.get("page", "")
            else:
                img_path = ""
                img_page = ""

            alignment_rows.append({
                "QID": i,
                "Query": q,
                "RetrievedText": top_text_snip,
                "ImagePath": img_path,
                "Page": img_page,
            })

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

    perq = Path(args.per_query_out)
    with perq.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_query_rows)

    align = Path(args.align_out)
    with align.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(alignment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(alignment_rows)

    print("\nPoster-ready evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k:36s}: {v}")
    print(f"\nSaved table CSV:      {out.resolve()}")
    print(f"Saved per-query CSV:  {perq.resolve()}")
    print(f"Saved alignment CSV:  {align.resolve()}")


if __name__ == "__main__":
    main()
