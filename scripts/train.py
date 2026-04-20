from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

from common import build_prompt, dump_json, load_json, load_yaml, resolve_path


def make_sft_dataframe(csv_path: str, allowed_labels: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["prompt"] = df["text"].astype(str).apply(lambda x: build_prompt(x, allowed_labels))
    df["response"] = df["label_name"].astype(str)
    return df[["prompt", "response", "text", "label", "label_name"]]


def formatting_func(example: dict) -> dict:
    return {
        "text": [
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            for prompt, response in zip(example["prompt"], example["response"])
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    config_dir = cfg["_config_dir"]

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    lora_cfg = cfg["lora"]
    trainer_cfg = cfg["trainer"]
    output_cfg = cfg["output"]

    train_csv = resolve_path(data_cfg["train_csv"], base_dir=config_dir)
    test_csv = resolve_path(data_cfg["test_csv"], base_dir=config_dir)
    label_mapping_path = resolve_path(data_cfg["label_mapping_path"], base_dir=config_dir)
    output_dir = resolve_path(output_cfg["model_dir"], base_dir=config_dir)

    mapping = load_json(label_mapping_path)
    allowed_labels = mapping["selected_labels"]

    train_df = make_sft_dataframe(train_csv, allowed_labels)
    test_df = make_sft_dataframe(test_csv, allowed_labels)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)
    train_ds = train_ds.map(formatting_func, batched=True)
    test_ds = test_ds.map(formatting_func, batched=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg.get("dtype"),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )

    # Force special tokens explicitly for Qwen
    tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    model.config.pad_token_id = tokenizer.pad_token_id

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=lora_cfg.get("random_state", 42),
    )

    eval_subset_size = min(trainer_cfg.get("eval_subset_size", len(test_ds)), len(test_ds))
    eval_ds = test_ds.select(range(eval_subset_size)) if eval_subset_size > 0 else None

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=trainer_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=trainer_cfg["gradient_accumulation_steps"],
        learning_rate=trainer_cfg["learning_rate"],
        num_train_epochs=trainer_cfg["num_train_epochs"],
        logging_steps=trainer_cfg["logging_steps"],
        eval_strategy=trainer_cfg.get("eval_strategy", "no"),
        eval_steps=trainer_cfg.get("eval_steps"),
        save_steps=trainer_cfg.get("save_steps"),
        save_total_limit=trainer_cfg.get("save_total_limit"),
        warmup_steps=trainer_cfg.get("warmup_steps", 0),
        lr_scheduler_type=trainer_cfg.get("lr_scheduler_type", "linear"),
        weight_decay=trainer_cfg.get("weight_decay", 0.0),
        fp16=trainer_cfg.get("fp16", False),
        bf16=trainer_cfg.get("bf16", False),
        report_to=trainer_cfg.get("report_to", "none"),
        # dataset_text_field=trainer_cfg.get("dataset_text_field", "text"),
        # max_length=model_cfg["max_seq_length"],
        # packing=trainer_cfg.get("packing", False),
        # eos_token=tokenizer.eos_token,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    dump_json(mapping, os.path.join(output_dir, "label_mapping.json"))
    shutil.copyfile(resolve_path(args.config), os.path.join(output_dir, "train_config.yaml"))

    print(f"Saved model to: {output_dir}")


if __name__ == "__main__":
    main()
