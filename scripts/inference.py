from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from unsloth import FastLanguageModel

from common import build_prompt, load_json, load_yaml, resolve_path


class IntentClassification:
    def __init__(self, model_path: str):
        cfg = load_yaml(model_path)
        config_dir = cfg["_config_dir"]

        model_cfg = cfg["model"]
        inference_cfg = cfg["inference"]
        data_cfg = cfg["data"]

        self.max_seq_length = model_cfg["max_seq_length"]
        self.max_new_tokens = inference_cfg.get("max_new_tokens", 16)
        self.do_sample = inference_cfg.get("do_sample", False)

        checkpoint_dir = model_cfg["checkpoint_dir"]

        if os.path.isabs(checkpoint_dir) or checkpoint_dir.startswith("."):
            self.model_dir = resolve_path(checkpoint_dir, base_dir=config_dir)
        else:
            # giữ nguyên HF repo id, ví dụ: unsloth/Qwen2.5-1.5B-Instruct
            self.model_dir = checkpoint_dir

        label_mapping_path = resolve_path(data_cfg["label_mapping_path"], base_dir=config_dir)

        if not os.path.exists(label_mapping_path):
            candidate = os.path.join(self.model_dir, "label_mapping.json")
            if os.path.exists(candidate):
                label_mapping_path = candidate
            else:
                raise FileNotFoundError(
                    f"Cannot find label mapping. Checked: {label_mapping_path} and {candidate}"
                )

        mapping = load_json(label_mapping_path)
        self.allowed_labels = mapping["selected_labels"]

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_dir,
            max_seq_length=self.max_seq_length,
            dtype=model_cfg.get("dtype"),
            load_in_4bit=model_cfg.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(self.model)

    def __call__(self, message: str) -> str:
        prompt = build_prompt(str(message), self.allowed_labels)
        input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                use_cache=True,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        pred = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        pred = pred.splitlines()[0].strip()

        if pred not in self.allowed_labels:
            for label in self.allowed_labels:
                if label in pred:
                    return label
            return "UNKNOWN"

        return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    parser.add_argument("--message", type=str, required=True)
    args = parser.parse_args()

    classifier = IntentClassification(args.config)
    pred = classifier(args.message)
    print(pred)


if __name__ == "__main__":
    main()
