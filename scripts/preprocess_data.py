from datasets import load_dataset
import pandas as pd
import os
import json

selected_labels = [
    "card_arrival",
    "card_not_working",
    "declined_card_payment",
    "card_payment_not_recognised",
    "card_payment_fee_charged",
    "card_swallowed",
    "lost_or_stolen_card",
    "pending_card_payment",
    "card_delivery_estimate",
    "activate_my_card"
]

def main():
    ds = load_dataset("PolyAI/banking77", revision="refs/pr/6", cache_dir="./hf_cache")

    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])

    label_names = ds["train"].features["label"].names
    id2label = {id : name for id, name in enumerate(label_names)}
    label2id = {name: id for id, name in enumerate(label_names)}

    train_df["label_name"] = train_df["label"].map(id2label)
    test_df["label_name"] = test_df["label"].map(id2label)

    selected_ids = [label2id[name] for name in selected_labels]

    train_sub = train_df[train_df["label"].isin(selected_ids)].copy()
    test_sub = test_df[test_df["label"].isin(selected_ids)].copy()

    new_label_names = sorted(train_sub["label_name"].unique())
    new_label2id = {name : id for id, name in enumerate(selected_labels)}
    new_id2label = {id : name for id, name in enumerate(selected_labels)}

    train_out = train_sub[["text", "label", "label_name"]]
    test_out = test_sub[["text", "label", "label_name"]]

    os.makedirs("sample_data", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    train_out.to_csv("sample_data/train.csv")
    test_out.to_csv("sample_data/test.csv")

    with open("configs/label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": new_label2id,
                "id2label": {str(k): v for k, v in new_id2label.items()},
                "selected_labels": selected_labels
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Done.")
    print("Train:", train_out.shape)
    print("Test:", test_out.shape)
    print(train_out["label_name"].value_counts())

if __name__ == "__main__":
    main()