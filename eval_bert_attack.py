import json
import logging
import os
from typing import Any

import pandas as pd
import typer

from contentfuzz.cls import COLA, Analyzer, Encoder, StanceAnalyzer, ZeroshotAnalyzer
from contentfuzz.cls.encoder import to_prompt
from contentfuzz.cross_model import compute_cross_model_esr
from contentfuzz.evaluate import (
    compute_fuzz_metrics,
    print_eval_metrics,
)
from contentfuzz.run import run_batch_generation
from contentfuzz.stance_dataset import (
    Dataset,
    DatasetLangMap,
    StanceDataset,
    is_dataset,
    load_c_stance,
    load_sem16,
    load_vast,
)
from contentfuzz.utils import Language


def _parse_dataset_from_filename(filename: str) -> Dataset | None:
    """Extract dataset name from BERT-Attack output filename."""
    base_name = os.path.basename(filename)
    file_name = os.path.splitext(base_name)[0]
    parts = file_name.split("+")
    if parts and is_dataset(parts[-1]):
        return parts[-1]
    return None


def _extract_text_and_target(prompt: str) -> tuple[str, str]:
    """Extract post text and target from a ContentFuzz prompt."""
    if not prompt:
        return "", ""
    lower = prompt.lower()
    text_marker = "text:"
    target_marker = "target:"
    text = prompt.strip()
    target = ""
    text_start = lower.find(text_marker)
    if text_start != -1:
        text_start += len(text_marker)
        target_start = lower.find(target_marker, text_start)
        if target_start == -1:
            text = prompt[text_start:].strip()
        else:
            text = prompt[text_start:target_start].strip()
    target_start = lower.find(target_marker)
    if target_start != -1:
        target = prompt[target_start + len(target_marker) :].strip()
    return text, target


def _load_dataset(dataset_name: Dataset):
    match dataset_name:
        case "c-stance-a" | "c-stance-b":
            return load_c_stance(dataset_name, "test")
        case "sem16":
            return load_sem16("test")
        case "vast":
            return load_vast("test")
    raise ValueError(f"Unknown dataset {dataset_name}")


def _build_prompt_lookup(
    dataset,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    prompt_lookup: dict[str, dict[str, Any]] = {}
    pair_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in dataset:
        prompt = to_prompt(row["text"], row["target"])
        prompt_lookup[prompt] = row
        prompt_lookup[prompt.strip()] = row
        pair_lookup[(row["text"].strip(), row["target"].strip())] = row
    return prompt_lookup, pair_lookup


def _load_pairs(
    results_file: str,
    dataset: Dataset | None = None,
    use_prompt: bool = False,
    include_unchanged: bool = False,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    with open(results_file, "r") as handle:
        rows: list[dict[str, Any]] = json.load(handle)

    prompt_lookup: dict[str, dict[str, Any]] = {}
    pair_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if dataset is not None:
        dataset_rows = _load_dataset(dataset)
        prompt_lookup, pair_lookup = _build_prompt_lookup(dataset_rows)

    orig_texts: list[str] = []
    adv_texts: list[str] = []
    adv_entries: list[dict[str, Any]] = []
    for row in rows:
        seq = str(row.get("seq_a") or row.get("seq") or "").strip()
        adv = str(row.get("adv") or row.get("final_adverse") or "").strip()
        if not seq or not adv:
            continue

        orig_text, orig_target = _extract_text_and_target(seq)
        adv_text, _ = _extract_text_and_target(adv)

        if use_prompt:
            orig = seq
            fuzz = adv
        else:
            orig = orig_text
            fuzz = adv_text

        if orig and fuzz:
            if include_unchanged or seq != adv:
                orig_texts.append(orig)
                adv_texts.append(fuzz)

        if dataset is not None and adv_text:
            entry = prompt_lookup.get(seq) or prompt_lookup.get(seq.strip())
            if entry is None and orig_text and orig_target:
                entry = pair_lookup.get((orig_text, orig_target))
            if entry is None:
                continue
            adv_entries.append(
                {
                    "text": orig_text,
                    "target": entry["target"],
                    "stance": entry["stance"],
                    "new_text": adv_text,
                }
            )

    return orig_texts, adv_texts, adv_entries


def main(
    results_file: str,
    analyzer_name: Analyzer,
    model: str = "gemini-2.5-flash-lite",
    output_result_path: str | None = None,
    batch_size: int = 1,
    dataset: Dataset | None = None,
    lang: Language | None = None,
    use_prompt: bool = False,
    include_unchanged: bool = False,
) -> None:
    if dataset is None:
        dataset = _parse_dataset_from_filename(results_file)

    assert dataset is not None and is_dataset(dataset), f"Unknown dataset: {dataset}"

    lang = DatasetLangMap[dataset]

    dataset_for_pairs = dataset if analyzer_name is not None else None
    orig_texts, adv_texts, adv_entries = _load_pairs(
        results_file,
        dataset=dataset_for_pairs,
        use_prompt=use_prompt,
        include_unchanged=include_unchanged,
    )

    if output_result_path is None:
        output_dir = os.path.abspath("results")
        os.makedirs(output_dir, exist_ok=True)
        safe_model = model.replace("/", "--")
        output_result_path = (
            f"{output_dir}/adv-cls+{analyzer_name}+{safe_model}+{dataset}.jsonl"
        )

    analyzer: StanceAnalyzer
    match analyzer_name:
        case "zeroshot":
            analyzer = ZeroshotAnalyzer(model=model)
        case "cola":
            analyzer = COLA(model=model, language=lang)
        case "encoder":
            analyzer = Encoder(model=model)

    adv_dataset: StanceDataset = [
        {
            "text": entry["new_text"],
            "target": entry["target"],
            "stance": entry["stance"],
        }
        for entry in adv_entries
    ]
    logging.info(
        "Running %s on adv_text with %s.",
        analyzer.__class__.__name__,
        analyzer.model,
    )
    results = run_batch_generation(
        adv_dataset,
        analyzer,
        batch_size=batch_size,
        output_result_path=output_result_path,
    )

    results_df = pd.DataFrame(results)
    atk_succ_rate = compute_cross_model_esr(results_df)
    success_mask = results_df["truth"] != results_df["predicted"]
    logging.info(
        f"Attack success rate: {len(success_mask)}/{len(results_df)} = {atk_succ_rate:.4f}"
    )

    adapted = pd.DataFrame(adv_entries)
    adapted["predicted"] = results_df["predicted"]
    adapted["iteration"] = pd.NA
    success_df = adapted.loc[success_mask].reset_index(drop=True)

    metrics = compute_fuzz_metrics(
        success_df,
        lang=lang,
        include_bertscore=True,
        include_perplexity=True,
        include_mauve=True,
    )
    metrics["attack_succ_rate"] = atk_succ_rate
    print_eval_metrics(metrics)


if __name__ == "__main__":
    typer.run(main)
