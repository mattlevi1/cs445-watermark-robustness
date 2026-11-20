"""Download C4 RealNewsLike and print summary statistics.

This is the first step of our CS 445 project on the robustness of
AI-text watermarks. Following Kirchenbauer et al. (2023), each news
article is split into a *prompt* (the opening of the article) and a
*human continuation* (the next T tokens). Watermarked model outputs will
later be generated from the prompts, and the human continuations serve
as unwatermarked controls for setting detection thresholds.

The script:
  1. Streams N articles from C4 RealNewsLike (no 15 GB download needed).
  2. Prints statistics on article length, in words and in tokens.
  3. Counts how many articles are long enough to give a prompt plus a
     T-token continuation.
  4. Builds prompt/continuation pairs and splits them into a calibration
     set and a test set, saved as JSONL files in data/.

Usage:
    python explore_data.py                  # 1,000 articles, OPT tokenizer
    python explore_data.py --n 200          # quicker run
    python explore_data.py --tokenizer none # word counts only, no model download
    python explore_data.py --local-file sample.jsonl  # offline test
"""

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


def load_articles(n, local_file=None):
    """Return a list of up to n articles, each a dict with 'text' and 'url'."""
    if local_file:
        with open(local_file, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return rows[:n]

    from datasets import load_dataset

    # Streaming reads examples one at a time instead of downloading the
    # whole subset. The validation split is smaller and fine for our needs.
    stream = load_dataset(
        "allenai/c4", "realnewslike", split="validation", streaming=True
    )
    articles = []
    for row in stream:
        articles.append({"text": row["text"], "url": row.get("url", "")})
        if len(articles) >= n:
            break
    return articles


def make_token_counter(tokenizer_name):
    """Return (encode, decode) functions. Falls back to whitespace words."""
    if tokenizer_name.lower() == "none":
        return (lambda text: text.split()), (lambda tokens: " ".join(tokens))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    decode = lambda ids: tok.decode(ids)
    return encode, decode


def describe(values, label):
    """Print mean, median, min, and max of a list of numbers."""
    print(
        f"  {label:<22} mean {statistics.mean(values):8.1f}   "
        f"median {statistics.median(values):8.1f}   "
        f"min {min(values):6d}   max {max(values):7d}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000, help="articles to load")
    parser.add_argument(
        "--continuation-tokens", type=int, default=200,
        help="length T of the human continuation (default 200)",
    )
    parser.add_argument(
        "--min-prompt-tokens", type=int, default=50,
        help="minimum prompt length for an article to be usable",
    )
    parser.add_argument(
        "--tokenizer", default="facebook/opt-1.3b",
        help="Hugging Face tokenizer name, or 'none' for whitespace words",
    )
    parser.add_argument(
        "--calibration-fraction", type=float, default=0.2,
        help="fraction of usable pairs held out for threshold calibration",
    )
    parser.add_argument("--seed", type=int, default=445)
    parser.add_argument("--local-file", help="read articles from a JSONL file")
    parser.add_argument("--out", default="data", help="output directory")
    args = parser.parse_args()

    print(f"Loading {args.n} articles from C4 RealNewsLike...")
    articles = load_articles(args.n, args.local_file)
    print(f"Loaded {len(articles)} articles.\n")

    unit = "words" if args.tokenizer.lower() == "none" else "tokens"
    encode, decode = make_token_counter(args.tokenizer)

    word_counts = [len(a["text"].split()) for a in articles]
    token_lists = [encode(a["text"]) for a in articles]
    token_counts = [len(t) for t in token_lists]

    print("Article length")
    describe(word_counts, "words")
    if unit == "tokens":
        describe(token_counts, f"tokens ({args.tokenizer})")

    domains = Counter(urlparse(a["url"]).netloc for a in articles if a.get("url"))
    if domains:
        print(f"\nTop news domains ({len(domains)} distinct)")
        for domain, count in domains.most_common(10):
            print(f"  {domain:<35} {count:5d}")

    # Build prompt / human-continuation pairs, as in Kirchenbauer et al. (2023):
    # the last T tokens are the continuation, everything before is the prompt.
    T = args.continuation_tokens
    needed = args.min_prompt_tokens + T
    pairs = []
    for article, tokens in zip(articles, token_lists):
        if len(tokens) < needed:
            continue
        pairs.append({
            "url": article.get("url", ""),
            "prompt": decode(tokens[:-T]),
            "human_continuation": decode(tokens[-T:]),
            "prompt_length": len(tokens) - T,
        })

    usable = len(pairs)
    print(
        f"\nUsable articles (at least {args.min_prompt_tokens} prompt + {T} "
        f"continuation {unit}): {usable} of {len(articles)} "
        f"({100 * usable / max(len(articles), 1):.1f}%)"
    )
    if not pairs:
        print("No usable pairs; try a larger --n or a smaller --continuation-tokens.")
        return
    describe([p["prompt_length"] for p in pairs], f"prompt length ({unit})")

    # Split into calibration (thresholds, settings) and test (final results).
    random.Random(args.seed).shuffle(pairs)
    n_cal = round(args.calibration_fraction * usable)
    splits = {"calibration": pairs[:n_cal], "test": pairs[n_cal:]}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nSplits")
    for name, rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  {name:<12} {len(rows):5d} pairs  ->  {path}")

    example = splits["test"][0] if splits["test"] else splits["calibration"][0]
    print("\nExample pair")
    print(f"  prompt (last 200 chars): ...{example['prompt'][-200:]!r}")
    print(f"  human continuation (first 200 chars): {example['human_continuation'][:200]!r}...")


if __name__ == "__main__":
    main()
