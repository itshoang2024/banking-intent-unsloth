import os
import json
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score, classification_report

DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
DEFAULT_FINETUNED_MODEL = "outputs/qwen25-banking-lora"
MAX_SEQ_LENGTH = 512

def build_prompt(text, allowed_labels):
    labels_str = ", ".join(allowed_labels)
    return (
        "Classify the banking customer intent.\n"
        "Return only one label from the allowed labels.\n\n"
        f"Allowed labels:\n{labels_str}\n\n"
        f"Message:\n{text}"
    )

def predict_label(model, tokenizer, text, allowed_labels):
    text = str(text)
    prompt = build_prompt(text, allowed_labels)

    input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    pred = pred.splitlines()[0].strip()

    if pred not in allowed_labels:
        for label in allowed_labels:
            if label in pred:
                return label
        return "UNKNOWN"

    return pred

def load_mapping(model_dir=None, fallback_path="configs/label_mapping.json"):
    if model_dir:
        candidate = os.path.join(model_dir, "label_mapping.json")
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)

    with open(fallback_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_FINETUNED_MODEL)
    parser.add_argument("--test_csv", type=str, default="sample_data/test.csv")
    parser.add_argument("--output_csv", type=str, default="outputs/eval_predictions.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    mapping = load_mapping(model_dir=args.model_path)
    allowed_labels = mapping["selected_labels"]

    test_df = pd.read_csv(args.test_csv)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    y_true = []
    y_pred = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        true_label = str(row["label_name"])
        pred_label = predict_label(model, tokenizer, str(row["text"]), allowed_labels)

        y_true.append(true_label)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_true, y_pred, zero_division=0))

    results_df = test_df.copy()
    results_df["pred_label"] = y_pred
    results_df["is_correct"] = results_df["label_name"].astype(str) == results_df["pred_label"].astype(str)
    results_df.to_csv(args.output_csv, index=False)

    print(f"Saved predictions to {args.output_csv}")

if __name__ == "__main__":
    main()