from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from common import dump_json, load_yaml, resolve_path


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    config_dir = cfg["_config_dir"]
    dataset_cfg = cfg["dataset"]
    output_cfg = cfg["output"]

    train_csv_path = resolve_path(output_cfg["train_csv"], base_dir=config_dir)
    test_csv_path = resolve_path(output_cfg["test_csv"], base_dir=config_dir)
    label_mapping_path = resolve_path(output_cfg["label_mapping_path"], base_dir=config_dir)
    cache_dir = resolve_path(dataset_cfg.get("cache_dir", "./hf_cache"), base_dir=config_dir)

    ds = load_dataset(
        dataset_cfg["name"],
        revision=dataset_cfg.get("revision"),
        cache_dir=cache_dir,
    )

    train_df = pd.DataFrame(ds[dataset_cfg.get("train_split", "train")])
    test_df = pd.DataFrame(ds[dataset_cfg.get("test_split", "test")])

    label_names = ds[dataset_cfg.get("train_split", "train")].features["label"].names
    original_id2label = {idx: name for idx, name in enumerate(label_names)}
    original_label2id = {name: idx for idx, name in enumerate(label_names)}

    selected_labels = dataset_cfg["selected_labels"]
    selected_ids = [original_label2id[name] for name in selected_labels]
    new_label2id = {name: idx for idx, name in enumerate(selected_labels)}
    new_id2label = {idx: name for idx, name in enumerate(selected_labels)}

    def process_split(df: pd.DataFrame) -> pd.DataFrame:
        out = df[df["label"].isin(selected_ids)].copy()
        out["label_name"] = out["label"].map(original_id2label)
        out["text"] = out["text"].map(normalize_text)
        out["label"] = out["label_name"].map(new_label2id)
        return out[["text", "label", "label_name"]].reset_index(drop=True)

    train_out = process_split(train_df)
    test_out = process_split(test_df)

    Path(train_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_csv_path).parent.mkdir(parents=True, exist_ok=True)

    train_out.to_csv(train_csv_path, index=False)
    test_out.to_csv(test_csv_path, index=False)

    dump_json(
        {
            "label2id": new_label2id,
            "id2label": {str(k): v for k, v in new_id2label.items()},
            "selected_labels": selected_labels,
        },
        label_mapping_path,
    )

    print(f"Saved train split to: {train_csv_path}")
    print(f"Saved test split to: {test_csv_path}")
    print(f"Saved label mapping to: {label_mapping_path}")
    print("Train shape:", train_out.shape)
    print("Test shape:", test_out.shape)
    print(train_out["label_name"].value_counts())


if __name__ == "__main__":
    main()
