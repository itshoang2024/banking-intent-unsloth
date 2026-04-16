import json
import pandas as pd
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score, classification_report

MODEL_DIR = "outputs/qwen25-banking-lora"
MAX_SEQ_LENGTH = 512

def build_prompt(text, allowed_lables):
    labels_str = ", ".join(allowed_lables)
    return (
        "Classify the banking customer intent.\n"
        "Return only one label from the allowed labels.\n\n"
        f"Allowed labels:\n{labels_str}\n\n"
        f"Message:\n{text}"
    )

def predict_label(model, tokenizer, text, allowed_labels):
    prompt = build_prompt(text, allowed_labels)
    messages = [{
        "role": "user",
        "content" : prompt
    }]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenizer=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = decoded.split("assistant")[-1].strip().splitlines()[0].strip()

    if pred not in allowed_labels:
        # simple fallback
        for label in allowed_labels:
            if label in pred:
                return label
        return "UNKNOWN"
    return pred

def main():
    with open(f'{MODEL_DIR}/label_mapping.json', "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    allowed_labels = mapping["selected_labels"]
    test_df = pd.read_csv("sample_data/test.csv")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    y_true, y_pred = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        pred = predict_label(model, tokenizer, row["text"], allowed_labels)
        y_true.append(row["label_name"])
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    main()