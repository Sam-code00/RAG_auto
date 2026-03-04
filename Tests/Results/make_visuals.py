import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import textwrap


def wrap(s: str, width: int) -> str:
    s = "" if s is None else str(s)
    return "\n".join(textwrap.wrap(s, width=width))


def make_small_chart(per_query_csv: str, out_png: str, title: str):
    df = pd.read_csv(per_query_csv).sort_values("QID")

    required = {"QID", "retrieval_cosine", "grounding_cosine", "pages_unique"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {per_query_csv}: {sorted(missing)}")

    fig, ax1 = plt.subplots(figsize=(6.5, 2.6))  # small, poster-friendly

    ax1.plot(df["QID"], df["retrieval_cosine"], marker="o", linewidth=1.5, label="Retrieval cosine")
    ax1.plot(df["QID"], df["grounding_cosine"], marker="o", linewidth=1.5, label="Grounding cosine")
    ax1.set_xlabel("Query ID")
    ax1.set_ylabel("Cosine similarity")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linewidth=0.5, alpha=0.4)

    ax2 = ax1.twinx()
    ax2.bar(df["QID"], df["pages_unique"], alpha=0.25, label="Unique pages")
    ax2.set_ylabel("Unique pages")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=3, frameon=True, fontsize=9)

    fig.suptitle(title, y=1.05, fontsize=11)
    fig.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    print(f"Saved: {out_png}")


def make_alignment_example(align_csv: str, out_png: str, row: int, title: str):
    df = pd.read_csv(align_csv)
    required = {"Query", "RetrievedText", "ImagePath"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {align_csv}: {sorted(missing)}")
    if row < 0 or row >= len(df):
        raise IndexError(f"--row {row} out of range (0..{len(df)-1})")

    ex = df.iloc[row]
    query = str(ex["Query"])
    retrieved = str(ex["RetrievedText"])
    img_path = str(ex["ImagePath"]) if not pd.isna(ex["ImagePath"]) else ""
    page = ex["Page"] if "Page" in df.columns else None

    fig = plt.figure(figsize=(8.2, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.12)
    ax_text = fig.add_subplot(gs[0, 0])
    ax_img = fig.add_subplot(gs[0, 1])
    ax_text.axis("off")
    ax_img.axis("off")

    page_str = ""
    if page is not None and str(page) != "nan":
        try:
            page_str = f" (Manual p. {int(page)})"
        except:
            page_str = f" (Manual p. {page})"

    text_block = (
        f"Query:\n{wrap(query, 52)}\n\n"
        f"Retrieved procedural text:\n{wrap(retrieved, 52)}\n\n"
        f"Supporting diagram{page_str}:"
    )
    ax_text.text(0.0, 1.0, text_block, va="top", ha="left", fontsize=10)

    if img_path.strip():
        img = Image.open(img_path).convert("RGB")
        ax_img.imshow(img)
    else:
        ax_img.text(0.5, 0.5, "No image retrieved for this query",
                    ha="center", va="center", fontsize=11)

    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    print(f"Saved: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_query_csv", default="per_query_metrics.csv")
    ap.add_argument("--align_csv", default="alignment_examples.csv")
    ap.add_argument("--out_chart", default="results_chart.png")
    ap.add_argument("--out_align", default="alignment_example.png")
    ap.add_argument("--align_row", type=int, default=0)
    ap.add_argument("--chart_title", default="Retrieval & Grounding Consistency")
    ap.add_argument("--align_title", default="Text–Diagram Alignment Example")
    args = ap.parse_args()

    make_small_chart(args.per_query_csv, args.out_chart, args.chart_title)
    make_alignment_example(args.align_csv, args.out_align, args.align_row, args.align_title)


if __name__ == "__main__":
    main()
