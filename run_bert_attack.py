import argparse
import importlib.util
import json
import logging
import random
from pathlib import Path
from types import ModuleType

import torch
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizer,
)

from contentfuzz._types import Stance
from contentfuzz.cls.encoder import _label_to_stance, to_prompt
from contentfuzz.stance_dataset import Dataset, load_c_stance, load_sem16, load_vast
from contentfuzz.utils import SEED

# Default model choices for each dataset.
DEFAULT_MODELS: dict[Dataset, dict[str, str]] = {
    "c-stance-a": {
        "tgt": "saved_models/hfl/chinese-bert-wwm",
        "mlm": "hfl/chinese-bert-wwm",
    },
    "c-stance-b": {
        "tgt": "saved_models/hfl/chinese-bert-wwm",
        "mlm": "hfl/chinese-bert-wwm",
    },
    "sem16": {
        "tgt": "saved_models/google-bert/bert-base-uncased/sem16",
        "mlm": "google-bert/bert-base-uncased",
    },
    "vast": {
        "tgt": "saved_models/google-bert/bert-base-uncased/vast",
        "mlm": "google-bert/bert-base-uncased",
    },
}


def load_bertattack_module() -> ModuleType:
    """Dynamically import the upstream BERT-Attack implementation."""
    module_path = Path("BERT-Attack/bertattack.py")
    spec = importlib.util.spec_from_file_location("bertattack", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import BERT-Attack from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _load_dataset(dataset_name: Dataset, split: str):
    match dataset_name:
        case "c-stance-a" | "c-stance-b":
            return load_c_stance(dataset_name, split)
        case "sem16":
            return load_sem16(split)
        case "vast":
            return load_vast(split)
    raise ValueError(f"Unknown dataset {dataset_name}")


def _build_stance_to_id_map(config: BertConfig) -> dict[Stance, int]:
    """Infer a mapping from Stance -> label id based on the model config."""
    stance_to_id: dict[Stance, int] = {}
    label2id = getattr(config, "label2id", {}) or {}
    for label, idx in label2id.items():
        stance = _label_to_stance(label)
        if stance is not None:
            stance_to_id[stance] = idx

    if stance_to_id:
        return stance_to_id

    default_labels: list[Stance] = ["Favor", "Against"]
    if config.num_labels >= 3:
        default_labels.append("Neutral")
    return {stance: idx for idx, stance in enumerate(default_labels)}


def _prepare_features(dataset, stance_to_id: dict[Stance, int], start: int, end: int):
    features = []
    sliced = dataset[start:end] if end is not None else dataset[start:]
    for row in sliced:
        stance = row["stance"]
        if stance not in stance_to_id:
            raise ValueError(f"Stance {stance} missing from label map {stance_to_id}")
        prompt = to_prompt(row["text"], row["target"])
        features.append((prompt, stance_to_id[stance], row))
    return features


def _summarize_features(features) -> dict[str, float | int]:
    """Compute the same metrics printed by BERT-Attack plus attack success rate."""
    total = len(features)
    successes = [f for f in features if f.success > 2]
    if total == 0:
        return {}

    origin_miscls = sum(1 for f in successes if f.success == 3)
    total_q = sum(f.query for f in successes)
    total_change = sum(f.change for f in successes)
    total_words = sum(len(f.seq.split(" ")) for f in successes)

    attack_successes = len(successes) - origin_miscls
    origin_acc = 1 - origin_miscls / total
    post_attack_acc = 1 - len(successes) / total
    avg_queries = total_q / len(successes) if successes else 0.0
    change_rate = (
        float(total_change) / total_words if successes and total_words > 0 else 0.0
    )

    return {
        "total_samples": total,
        "origin_accuracy": origin_acc,
        "post_attack_accuracy": post_attack_acc,
        "attack_success_rate": attack_successes / total,
        "avg_queries": avg_queries,
        "change_rate": change_rate,
        "orig_misclassified": origin_miscls,
    }


def main(
    dataset_name: Dataset,
    tgt_model: str | None,
    mlm_model: str | None,
    output_path: str | None,
    split: str = "test",
    k: int = 48,
    batch_size: int = 32,
    max_length: int = 512,
    threshold_pred_score: float = 0.0,
    use_bpe: bool = True,
    start: int = 0,
    end: int | None = None,
    sample_n: int | None = None,
    use_sim_mat: bool = False,
    embed_path: str | None = None,
    sim_mat_path: str | None = None,
):
    random.seed(SEED)
    atk_module = load_bertattack_module()
    dataset = _load_dataset(dataset_name, split)

    defaults = DEFAULT_MODELS.get(dataset_name, {})
    tgt_model = tgt_model or defaults.get("tgt")
    mlm_model = mlm_model or defaults.get("mlm") or tgt_model
    if tgt_model is None or mlm_model is None:
        raise ValueError(
            "Both target and MLM models must be provided (no defaults available)."
        )

    tokenizer = BertTokenizer.from_pretrained(tgt_model, do_lower_case=True)
    tgt_config = BertConfig.from_pretrained(tgt_model)
    stance_to_id = _build_stance_to_id_map(tgt_config)

    tgt_model_cfg = BertForSequenceClassification.from_pretrained(
        tgt_model, config=tgt_config
    ).to(atk_module.DEVICE)
    tgt_model_cfg.eval()

    mlm_config = BertConfig.from_pretrained(mlm_model)
    mlm_model_cfg = BertForMaskedLM.from_pretrained(mlm_model, config=mlm_config).to(
        atk_module.DEVICE
    )
    mlm_model_cfg.eval()

    if sample_n is not None:
        dataset = random.sample(dataset, k=min(sample_n, len(dataset)))

    feature_rows = _prepare_features(dataset, stance_to_id, start, end)
    logging.info(
        "Dataset: %s (%d samples), target model: %s, MLM: %s",
        dataset_name,
        len(feature_rows),
        tgt_model,
        mlm_model,
    )

    cos_mat = None
    w2i = {}
    i2w = {}
    if use_sim_mat:
        if embed_path is None or sim_mat_path is None:
            raise ValueError(
                "embed_path and sim_mat_path are required with --use-sim-mat"
            )
        cos_mat, w2i, i2w = atk_module.get_sim_embed(embed_path, sim_mat_path)

    attacked_features = []
    with torch.no_grad():
        for prompt, label_id, _ in tqdm(feature_rows, desc="Running BERT-Attack"):
            feature = atk_module.Feature(prompt, label_id)
            attacked = atk_module.attack(
                feature,
                tgt_model_cfg,
                mlm_model_cfg,
                tokenizer,
                k,
                batch_size,
                max_length=max_length,
                cos_mat=cos_mat,
                w2i=w2i,
                i2w=i2w,
                use_bpe=1 if use_bpe else 0,
                threshold_pred_score=threshold_pred_score,
            )
            attacked_features.append(attacked)

    metrics = _summarize_features(attacked_features)
    atk_module.evaluate(attacked_features)

    if output_path is None:
        safe_model = Path(tgt_model).as_posix().replace("/", "--")
        output_dir = Path("BERT-Attack/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"bert-attack+{safe_model}+{dataset_name}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    atk_module.dump_features(attacked_features, str(output_path))
    summary_path = output_path.with_suffix(".summary.json")
    if metrics:
        summary_path.write_text(json.dumps(metrics, indent=2))
        logging.info("BERT-Attack metrics: %s", metrics)
        logging.info("BERT-Attack summary written to %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Run BERT-Attack on ContentFuzz datasets."
    )
    parser.add_argument(
        "dataset", choices=list(DEFAULT_MODELS.keys()), help="Dataset to attack."
    )
    parser.add_argument(
        "--tgt-model",
        dest="tgt_model",
        help="Fine-tuned stance classifier to attack. Defaults to the BERT model used in our experiments.",
    )
    parser.add_argument(
        "--mlm-model",
        dest="mlm_model",
        help="Masked-LM backbone used for substitutions. Defaults to the dataset's MLM.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Where to store attack logs (JSON). Defaults to results/bert_attack/bert-attack+{model}+{dataset}.json",
    )
    parser.add_argument(
        "--split", default="test", help="Dataset split to use (default: test)."
    )
    parser.add_argument(
        "-k", type=int, default=48, help="Number of MLM candidates (default: 48)."
    )
    parser.add_argument("-b", "--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--max-length", type=int, default=512, help="Max sequence length."
    )
    parser.add_argument(
        "--threshold-pred-score",
        type=float,
        default=0.0,
        help="Prune MLM candidates below this score (default: 0).",
    )
    parser.add_argument(
        "--no-bpe", action="store_true", help="Disable BPE substitutions."
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index for dataset slice."
    )
    parser.add_argument(
        "--end", type=int, help="End index for dataset slice (exclusive)."
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        dest="sample_n",
        help="If set, randomly sample this many examples before attacking.",
    )
    parser.add_argument(
        "--use-sim-mat",
        action="store_true",
        help="Enable cosine similarity filter; requires --embed-path and --sim-mat-path.",
    )
    parser.add_argument(
        "--embed-path", help="Path to counter-fitted vectors (for --use-sim-mat)."
    )
    parser.add_argument(
        "--sim-mat-path", help="Path to cosine similarity matrix (for --use-sim-mat)."
    )
    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        tgt_model=args.tgt_model,
        mlm_model=args.mlm_model,
        output_path=args.output,
        split=args.split,
        k=args.k,
        batch_size=args.batch_size,
        max_length=args.max_length,
        threshold_pred_score=args.threshold_pred_score,
        use_bpe=not args.no_bpe,
        start=args.start,
        end=args.end,
        sample_n=args.sample_n,
        use_sim_mat=args.use_sim_mat,
        embed_path=args.embed_path,
        sim_mat_path=args.sim_mat_path,
    )
