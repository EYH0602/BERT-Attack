import argparse
import json
import logging
import os
from typing import Any

from contentfuzz.evaluate import (
    compute_bertscore,
    compute_mauve,
    compute_perplexity_ratio,
    print_eval_metrics,
)
from contentfuzz.stance_dataset import Dataset, DatasetLangMap, is_dataset
from contentfuzz.utils import Language


def _parse_dataset_from_filename(filename: str) -> Dataset | None:
    """Extract dataset name from BERT-Attack output filename."""
    base_name = os.path.basename(filename)
    file_name = os.path.splitext(base_name)[0]
    parts = file_name.split("+")
    if parts and is_dataset(parts[-1]):
        return parts[-1]
    return None


def _extract_text(prompt: str) -> str:
    """Extract the post text from a ContentFuzz prompt, fallback to full prompt."""
    if not prompt:
        return ""
    lower = prompt.lower()
    text_marker = "text:"
    start = lower.find(text_marker)
    if start == -1:
        return prompt.strip()
    start += len(text_marker)
    target_marker = "target:"
    end = lower.find(target_marker, start)
    if end == -1:
        return prompt[start:].strip()
    return prompt[start:end].strip()


def _load_pairs(
    results_file: str,
    use_prompt: bool = False,
    include_unchanged: bool = False,
) -> tuple[list[str], list[str]]:
    with open(results_file, "r") as handle:
        rows: list[dict[str, Any]] = json.load(handle)

    orig_texts: list[str] = []
    adv_texts: list[str] = []
    for row in rows:
        seq = str(row.get("seq_a") or row.get("seq") or "").strip()
        adv = str(row.get("adv") or row.get("final_adverse") or "").strip()
        if not seq or not adv:
            continue
        if not include_unchanged and seq == adv:
            continue

        if use_prompt:
            orig = seq
            fuzz = adv
        else:
            orig = _extract_text(seq)
            fuzz = _extract_text(adv)

        if orig and fuzz:
            orig_texts.append(orig)
            adv_texts.append(fuzz)

    return orig_texts, adv_texts


def main(
    results_file: str,
    dataset: Dataset | None = None,
    lang: Language | None = None,
    use_prompt: bool = False,
    include_unchanged: bool = False,
    include_bertscore: bool = True,
    include_perplexity: bool = True,
    include_mauve: bool = True,
) -> None:
    if dataset is None:
        dataset = _parse_dataset_from_filename(results_file)

    if lang is None and dataset is not None:
        lang = DatasetLangMap[dataset]
    if lang is None:
        lang = "en"

    orig_texts, adv_texts = _load_pairs(
        results_file,
        use_prompt=use_prompt,
        include_unchanged=include_unchanged,
    )

    if not orig_texts:
        print_eval_metrics(
            {
                "num_pairs": 0,
                "bertscore": None,
                "perplexity": None,
                "mauve": None,
            }
        )
        return

    metrics: dict[str, Any] = {"num_pairs": len(orig_texts)}

    if include_bertscore:
        metrics["bertscore"] = compute_bertscore(
            orig_texts,
            adv_texts,
            lang=lang,
        )
    else:
        metrics["bertscore"] = None

    if include_perplexity:
        metrics["perplexity"] = compute_perplexity_ratio(
            orig_texts,
            adv_texts,
            alpha=0.05,
            max_tokens=512,
        )
    else:
        metrics["perplexity"] = None

    if include_mauve:
        metrics["mauve"] = compute_mauve(
            orig_texts,
            adv_texts,
            lang=lang,
        )
    else:
        metrics["mauve"] = None

    print_eval_metrics(metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Evaluate BERT-Attack outputs with text quality metrics."
    )
    parser.add_argument("results_file", help="Path to BERT-Attack JSON output.")
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=str,
        default=None,
        help="Optional dataset name to override parsing from filename.",
    )
    parser.add_argument(
        "--lang",
        dest="lang",
        type=str,
        choices=["en", "zh"],
        default=None,
        help="Optional language override (en/zh).",
    )
    parser.add_argument(
        "--use-prompt",
        action="store_true",
        help="Use full prompt text instead of extracting the post text.",
    )
    parser.add_argument(
        "--include-unchanged",
        action="store_true",
        help="Include pairs where the adversarial text matches the original.",
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Skip BERTScore computation.",
    )
    parser.add_argument(
        "--no-perplexity",
        action="store_true",
        help="Skip perplexity ratio computation.",
    )
    parser.add_argument(
        "--no-mauve",
        action="store_true",
        help="Skip Mauve computation.",
    )

    args = parser.parse_args()

    dataset_arg: Dataset | None = None
    if args.dataset is not None:
        if not is_dataset(args.dataset):
            raise ValueError(f"Unknown dataset: {args.dataset}")
        dataset_arg = args.dataset

    main(
        results_file=args.results_file,
        dataset=dataset_arg,
        lang=args.lang,
        use_prompt=args.use_prompt,
        include_unchanged=args.include_unchanged,
        include_bertscore=not args.no_bertscore,
        include_perplexity=not args.no_perplexity,
        include_mauve=not args.no_mauve,
    )
