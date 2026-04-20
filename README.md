<h1 align="center">banking-intent-unsloth</h1>

<p align="center">
  <a href="https://colab.research.google.com/drive/12iDPjNv-_gGMnmVrOozofHBwZbX1mDY-?usp=sharing">
    <img src="https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open in Colab" />
  </a>
  <a href="https://drive.google.com/file/d/1JutNckDcng3ok6OYvYyht0Ed_zydVWjN/view?usp=sharing">
    <img src="https://img.shields.io/badge/Demo%20Video-Google%20Drive-34A853?style=for-the-badge&logo=googledrive&logoColor=white" alt="Demo Video" />
  </a>
</p>

This project fine-tunes **`unsloth/Qwen2.5-1.5B-Instruct`** for **banking intent classification** on a **selected subset** of the original **[PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)** dataset using the **Unsloth** ecosystem.

---

## Task summary

- **Dataset**: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- **Base model**: `unsloth/Qwen2.5-1.5B-Instruct`
- **Training method**: LoRA fine-tuning with Unsloth
- **Inference interface**: `IntentClassification.__init__()` and `IntentClassification.__call__()`

---

## Dataset and subset used

The original dataset is **[PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)** on Hugging Face.

This project **does not use the full original dataset**. Instead, it only uses a **subset of 10 intent labels** selected in [`configs/train.yaml`](configs/train.yaml), and the remapped label IDs are saved to [`configs/label_mapping.json`](configs/label_mapping.json).

### Selected intent labels

```text
card_arrival
card_not_working
declined_card_payment
card_payment_not_recognised
card_payment_fee_charged
card_swallowed
lost_or_stolen_card
pending_card_payment
card_delivery_estimate
activate_my_card
```

---

## Step 1 - Preprocess the dataset

This step is implemented in [`scripts/preprocess_data.py`](scripts/preprocess_data.py).

It:
- downloads the original **[PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)** dataset,
- filters it to the selected **10 labels**,
- normalizes the text,
- remaps labels to a new 0-based label space,
- saves train/test CSV files,
- saves [`configs/label_mapping.json`](configs/label_mapping.json).

Run:

```bash
python scripts/preprocess_data.py --config configs/train.yaml
```

Generated files:
- [`sample_data/train.csv`](sample_data/train.csv)
- [`sample_data/test.csv`](sample_data/test.csv)
- [`configs/label_mapping.json`](configs/label_mapping.json)

---

## Step 2 - Fine-tune the model

Training is implemented in [`scripts/train.py`](scripts/train.py).

Run training:

```bash
python scripts/train.py --config configs/train.yaml
```

Or use the helper script:

```bash
bash train.sh configs/train.yaml
```

### What training does

- loads the preprocessed training data,
- converts each sample into an instruction-style prompt,
- applies LoRA to Qwen2.5-1.5B-Instruct,
- trains using `SFTTrainer`,
- saves the fine-tuned model and tokenizer.

### Training output

By default, the model is saved to:

```text
outputs/qwen25-banking-lora/
```

This folder will contain the saved adapter/model files, tokenizer files, and a copied training config.

---

## Step 3 - Run single-message inference

Inference is implemented in [`scripts/inference.py`](scripts/inference.py).

After training, run inference on one message:

```bash
python scripts/inference.py \
  --config configs/inference.yaml \
  --message "my card was swallowed by the ATM"
```

Or use the helper script:

```bash
bash inference.sh configs/inference.yaml "my card was swallowed by the ATM"
```

Example output:

```text
card_swallowed
```

---

## Step 4 - Evaluate on the test set

Evaluation is implemented in [`scripts/eval.py`](scripts/eval.py).

Run evaluation with the fine-tuned model:

```bash
python scripts/eval.py --config configs/inference.yaml
```

This will:
- run prediction for every sample in [`sample_data/test.csv`](sample_data/test.csv),
- print **accuracy**,
- print a **classification report**,
- save all predictions to a CSV file.

Default output file:

```text
outputs/eval_predictions.csv
```

### Evaluate the baseline model

To evaluate the original base model before fine-tuning:

```bash
python scripts/eval.py --config configs/inference_baseline.yaml
```

Default baseline prediction file:

```text
outputs/eval_baseline_predictions.csv
```

---

## Inference class interface

The required inference interface is implemented in [`scripts/inference.py`](scripts/inference.py):

```python
class IntentClassification:
    def __init__(self, model_path):
        ...

    def __call__(self, message):
        ...
        return predicted_label
```

### Example usage

```python
from scripts.inference import IntentClassification

classifier = IntentClassification("configs/inference.yaml")
pred = classifier("cash withdrawal is pending")
print(pred)
```

---

## Prompt format used in training and inference

The helper functions for prompt construction and config loading are in [`scripts/common.py`](scripts/common.py).

Each input message is converted into an instruction prompt that asks the model to return **only one label** from the allowed label list.

The model output is then post-processed to return the final predicted intent label.

---

## Output files

| File / Folder | Description |
|---|---|
| [`sample_data/train.csv`](sample_data/train.csv) | Preprocessed training split |
| [`sample_data/test.csv`](sample_data/test.csv) | Preprocessed test split |
| [`configs/label_mapping.json`](configs/label_mapping.json) | Mapping between label names and label IDs |
| `outputs/qwen25-banking-lora/` | Fine-tuned checkpoint and tokenizer |
| `outputs/eval_predictions.csv` | Predictions from the fine-tuned model |
| `outputs/eval_baseline_predictions.csv` | Predictions from the base model |

---

## Reproducibility notes

Before running training or evaluation, make sure:
- preprocessing has already been completed,
- [`configs/label_mapping.json`](configs/label_mapping.json) exists,
- [`sample_data/train.csv`](sample_data/train.csv) and [`sample_data/test.csv`](sample_data/test.csv) exist,
- the checkpoint path in the inference config is correct.

If you move the project to another machine, review the relative paths inside the YAML config files first.

---

## Suggested workflow

Run the project in this order:

```bash
python scripts/preprocess_data.py --config configs/train.yaml
python scripts/train.py --config configs/train.yaml
python scripts/eval.py --config configs/inference.yaml
python scripts/inference.py --config configs/inference.yaml --message "my card was swallowed by the ATM"
```

Optional baseline comparison:

```bash
python scripts/eval.py --config configs/inference_baseline.yaml
```

---

## Acknowledgements

- Dataset: **[PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)**
- Fine-tuning framework: **[Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)**
- Base model: **[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)**
