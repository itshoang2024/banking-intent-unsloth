import os
import json
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 512
OUTPUT_DIR = "outputs/qwen25-banking-lora"

def build_prompt(text, allowed_lables):
    labels_str = ", ".join(allowed_lables)
    return (
        "Classify the banking customer intent.\n"
        "Return only one label from the allowed labels.\n\n"
        f"Allowed labels:\n{labels_str}\n\n"
        f"Message:\n{text}"
    )

def make_sft_dataframe(csv_path, allowed_labels):
    df = pd.read_csv(csv_path)
    df["prompt"] = df["text"].apply(lambda x: build_prompt(x, allowed_labels))
    df["response"] = df["label_name"].astype(str)
    return df[["prompt", "response", "text", "label", "label_name"]]

def formatting_func(example):
    return {
        "text": [
            f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>"
            for p, r in zip(example["prompt"], example["response"])
        ]
    }

def main():
    with open("configs/label_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    allowed_labels = mapping["selected_labels"]

    train_df = make_sft_dataframe("sample_data/train.csv", allowed_labels)
    test_df = make_sft_dataframe("sample_data/test.csv", allowed_labels)

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_ds = train_ds.map(formatting_func, batched=True)
    test_ds = test_ds.map(formatting_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=test_ds.select(range(min(200, len(test_ds)))), # eval nhanh
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=2,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=2,
            warmup_steps=10,
            lr_scheduler_type="linear",
            weight_decay=0.01,
            fp16=False,
            bf16=False,
            report_to="none",
        ),
    )

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Save to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()