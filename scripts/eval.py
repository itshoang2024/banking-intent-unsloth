from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from common import ensure_parent_dir, load_yaml, resolve_path
from inference import IntentClassification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    config_dir = cfg["_config_dir"]
    test_csv = resolve_path(cfg["data"]["test_csv"], base_dir=config_dir)
    output_csv = resolve_path(cfg["output"]["prediction_csv"], base_dir=config_dir)

    classifier = IntentClassification(args.config)
    test_df = pd.read_csv(test_csv)

    y_true = []
    y_pred = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        true_label = str(row["label_name"])
        pred_label = classifier(str(row["text"]))
        y_true.append(true_label)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_true, y_pred, zero_division=0))

    result_df = test_df.copy()
    result_df["pred_label"] = y_pred
    result_df["is_correct"] = result_df["label_name"].astype(str) == result_df["pred_label"].astype(str)

    ensure_parent_dir(output_csv)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")


if __name__ == "__main__":
    main()
