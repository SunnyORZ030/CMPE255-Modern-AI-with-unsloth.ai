# CMPE255-Modern-AI-with-unsloth.ai
# 🧠 Colab1 - Unsloth Full Finetuning

### 📘 Overview
This repository contains all outputs from the **Colab1 Full Finetuning** task (CMPE 255).

### 📂 Contents
- `/data/chat_train.jsonl` – training dataset  
- `/smollm2_fullft_out/` – training logs and output  
- `/smollm2_fullft_ckpt/` – fine-tuned model checkpoint  
- `unsloth_fullft_upload.zip` – packed version for download  


# 🧠 Colab2 – LoRA Parameter-Efficient Finetuning (SmolLM2-135M)

### 📘 Overview
This notebook demonstrates parameter-efficient fine-tuning (LoRA) on the same dataset and model (SmolLM2-135M) as Colab1.
Unlike full finetuning, only a small subset of adapter weights (r=4) are trained, making it suitable for limited hardware such as CPU.

---

### 🧭 LoRA Settings
| Parameter | Value |
|------------|--------|
| `r` | 4 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.1 |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| `device` | CPU (no GPU quota) |
| `max_steps` | 150 |
| `batch_size` | 1 |
| `gradient_accumulation_steps` | 8 |
| `learning_rate` | 5e-4 |

---

### 🧩 Dataset
- Same dataset as Colab1 (`/data/chat_train.jsonl`)
- 5 instruction–response pairs, Alpaca-style format.

---

### 🏋️ Training Summary
- Runtime: CPU (approx. 10–15 min for 150 steps)
- Final loss: *<your final loss>*
- Output directory: `/lora_smolm2_cpu/`
- Optional merged weights: `/smolm2_lora_merged/`

---

### 💾 Files in This Repo
- LoRA.ipynb ← main training notebook
- /data/chat_train.jsonl ← dataset
- /lora_smolm2_cpu/ ← LoRA adapter weights
- /smolm2_lora_merged/ ← merged weights (optional)
- /video/ ← recorded walkthrough (optional)
- README.md ← this documentation

---

### 🧠 Inference Example
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("lora_smolm2_cpu", use_fast=True)
mdl = AutoModelForCausalLM.from_pretrained("lora_smolm2_cpu").to("cpu")
inp = tok("### Instruction:\nExplain cross-validation in one sentence.\n\n### Response:\n", return_tensors="pt")
out = mdl.generate(**inp, max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

# 🧠 Colab3 – Reinforcement Learning (DPO + LoRA)

### 📘 Overview
This notebook demonstrates **reinforcement learning using preference data**
with the **DPO (Direct Preference Optimization)** method and **LoRA parameter-efficient tuning**.
It fine-tunes the same base model (**SmolLM2-135M**) and dataset as Colab1 and Colab2,  
but learns from **preferred vs. rejected** responses instead of only correct answers.

---

### 🧩 Dataset
| Field | Description |
|--------|-------------|
| `prompt` | User input or question |
| `chosen` | Preferred (better) answer |
| `rejected` | Non-preferred (worse) answer |

📁 File: `/data/pref_dataset.jsonl`

Example:
```json
{
  "prompt": "Explain cross-validation in one sentence.",
  "chosen": "Cross-validation splits data into folds to estimate generalization reliably.",
  "rejected": "Cross-validation makes the model overfit less by training on the test set."
}
```
