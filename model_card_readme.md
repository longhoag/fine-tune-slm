---
base_model: meta-llama/Meta-Llama-3.1-8B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:meta-llama/Meta-Llama-3.1-8B
- lora
- transformers
---

# Model Card — fine-tune-slm LoRA adapter

This model card documents the LoRA adapter and related artifacts produced by the fine-tune-slm project. It summarizes what the adapter does, how it was trained, how to use it, known limitations and risks, and technical specifications.

## Short Description

A LoRA adapter fine-tuned on synthetic medical case reports to perform structured cancer-related information extraction (entities such as cancer type, stage, gene mutation, biomarker, treatment, response, and metastasis site). The adapter is trained on top of the Meta-Llama 3.1 8B base model using parameter-efficient LoRA adapters and 4-bit quantization (QLoRA) to reduce memory requirements.

## Model Details

- **Adapter type:** LoRA (PEFT)
- **Base model:** `meta-llama/Meta-Llama-3.1-8B`
- **Task / Pipeline tag:** `text-generation` (used for generative extraction into structured JSON)
- **Intended use:** Structured medical information extraction from clinical-style text for research and internal tooling (NOT for clinical decision-making)
- **Framework versions:** PEFT 0.17.1, Transformers (version used in training available in `pyproject.toml`)
- **Repository:** https://github.com/longhoag/fine-tune-slm


## Training Summary (this project)

### Data
- **Dataset:** Synthetic instruction-tuning dataset located in `synthetic-instruction-tuning-dataset/` (4,500 training examples, 500 validation examples). Each example contains an `instruction`, `input` clinical narrative, and a structured `output` JSON with 7 fields: `cancer_type`, `stage`, `gene_mutation`, `biomarker`, `treatment`, `response`, `metastasis_site`.
- **Max sequence length:** 2048 tokens (configured in training)

### LoRA & Quantization configuration
- **LoRA hyperparameters:**
  - `r = 16`
  - `lora_alpha = 32`
  - `lora_dropout = 0.1`
  - `target_modules = [q_proj, k_proj, v_proj, o_proj]`
  - `bias = none`
- **Quantization:** QLoRA / 4-bit quantization used during training
  - `load_in_4bit: true`
  - `bnb_4bit_compute_dtype: float16`
  - `bnb_4bit_quant_type: nf4`
  - `bnb_4bit_use_double_quant: true`

### Training hyperparameters
(Values come from `config/training_config.yml` used for the run)
- `num_train_epochs = 5`
- `per_device_train_batch_size = 4`
- `per_device_eval_batch_size = 4`
- `gradient_accumulation_steps = 4` (effective batch size = 16)
- `learning_rate = 2.0e-4`
- `warmup_steps = 100`
- `logging_steps = 10`
- `eval_steps = 100`
- `save_steps = 200`
- `save_total_limit = 3`
- `fp16 = true`
- `optimizer = paged_adamw_8bit`

### Infrastructure used for training
- **Compute:** EC2 `g6.2xlarge` (NVIDIA L4 GPU, 24 GB VRAM)
- **Storage for checkpoints:** EBS `gp3` (mounted at `/mnt/training` during training)
- **Artifacts:** Final checkpoints, TensorBoard logs, and exported adapter pushed to S3 and optionally to Hugging Face Hub

### Training run metrics (latest 5-epoch run)
- **Total runtime:** 9,994.9681 seconds (~2.78 hours)
- **Total steps:** ~1,410 steps
- **Training throughput:** 2.251 samples/sec | 0.141 steps/sec
- **Reported per-step loss (final logged step):** 0.5620
- **Aggregated training loss (end summary):** 0.6869197811640746
- **Final evaluation loss:** 0.6551951169967651
- **Best validation loss observed:** 0.652971 (approx; best eval around step ~1100)

Selected progress snapshots (examples taken from training logs):
```
Epoch 0.04: loss=2.6774, lr=1.8e-05 (warmup)
Epoch 0.36: eval_loss=0.8309
Epoch 1.00: loss~0.7003, eval_loss~0.6905
Epoch 3.90: best eval_loss ~0.6535 (checkpoint)
Epoch 5.00: final loss ~0.5620, eval_loss ~0.6552
```

### How overfitting was mitigated
- Frequent evaluations every 100 steps and `load_best_model_at_end=True` (keeps best eval checkpoint)
- LoRA adapters with `lora_dropout=0.1` (adapter-level regularization)
- Small learning rate with warmup steps (100-step warmup)
- Gradient accumulation to increase effective batch size and stabilize updates
- Limited number of epochs (5) and `save_total_limit` to keep checkpoints manageable


## Evaluation

- **Evaluation procedure:** Validation performed every 100 steps using the held-out 500-sample validation split. Metric: `eval_loss` (lower is better).
- **Key metrics:** Final `eval_loss = 0.6552` (21% improvement vs initial eval), best `eval_loss ≈ 0.65297`.
- **Throughput on eval:** ~7.2 samples/sec (varied slightly per run)

> Note: This project focuses on reducing eval loss as a proxy for model generalization for structured extraction. Downstream evaluation involving exact-match / field-level accuracy and manual review are recommended before any production use.


## How to Load & Use the Adapter (Examples)

Below are typical snippets to load the base Llama model + LoRA adapter for inference. Adjust device settings depending on your environment.

Python example using `transformers` + `peft`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", use_fast=False)

# Load base model (if you need 4-bit loading you can pass load_in_4bit options used in training)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=False,
)

# Load the LoRA adapter (replace with your adapter repo or path)
adapter = PeftModel.from_pretrained(base_model, "<your-hf-username>/<your-adapter-repo>")
adapter.eval()

# Simple generation
prompt = "Extract entities: 70-year-old man with widely metastatic cutaneous melanoma..."
inputs = tokenizer(prompt, return_tensors="pt").to(adapter.device)
outputs = adapter.generate(**inputs, max_new_tokens=256)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

If you published the adapter on the Hugging Face Hub as a `PeftModel` adapter, you can load it directly with `PeftModel.from_pretrained()` by providing the HF repo id.


## Intended Use and Limitations

### Intended use
- Research and development of extraction pipelines for oncology-related clinical text
- Prototyping structured information extraction (not clinical decision support)
- Integration into internal QA or annotation-assistant tools (with human-in-the-loop review)

### Out-of-scope / Not recommended
- Clinical diagnosis, triage, or any safety-critical medical decision-making
- Using the model without manual verification for real-world patient care

### Known limitations & risks
- **Synthetic training data:** The dataset is synthetic and may not capture real-world distributional nuances of clinical notes.
- **Hallucinations:** As a generative model, it can produce confident but incorrect outputs.
- **Bias & coverage:** May underperform on demographic groups or clinical contexts not represented in the synthetic dataset.
- **Privacy:** Do not use the model to generate or infer sensitive personal data beyond what is required for annotation tasks.

**Recommendations:**
- Always include human verification for extracted fields before using in downstream pipelines.
- Run additional evaluation on real clinical datasets (with appropriate IRB/permissions) before deploying.
- Monitor outputs and log errors; use conservative acceptance thresholds for any automated ingestion.


## Environmental Impact

Estimate energy and carbon is non-trivial; the run used a single `g6.2xlarge` GPU for ~2.8 hours. If you want a CO2 estimate, use the ML CO2 Impact calculator (https://mlco2.github.io/impact#compute) with:
- Hardware: NVIDIA L4 on EC2
- Hours used: ~2.78
- Region: us-east-1 (select the corresponding grid electricity emission factor)


## Technical Specifications

- **Model architecture:** Meta-Llama 3.1 (decoder-only transformer) + LoRA adapter (rank=16)
- **Compute used:** EC2 `g6.2xlarge` (1× L4 GPU, 24GB VRAM), 8 vCPUs, 32 GiB RAM
- **Software stack:** Python 3.10+, PyTorch, Transformers, PEFT 0.17.1, bitsandbytes (4-bit quantization), loguru for logging


## How this adapter was published

Artifacts produced during training (checkpoints, TensorBoard logs, exported adapter weights) were saved to an S3 bucket and the best adapter checkpoint was optionally pushed to the Hugging Face Hub under the configured repo name. See `scripts/finetune/push_to_hf.py` for the exact publish command.


## Responsible Release / Licensing

- This repository is released under the MIT license. The base model (`meta-llama/Meta-Llama-3.1-8B`) is governed by its own license and access restrictions—check the model page for licensing and usage conditions.
- When publishing adapters to Hugging Face, ensure you comply with the base model's license and any data-use restrictions.


## Contact

- **Maintainer:** Long Hoang (@longhoag)
- **Repository:** https://github.com/longhoag/fine-tune-slm


---

If you'd like edits (more evaluation tables, per-field accuracy numbers, examples of correct vs incorrect predictions, or a shorter public-facing summary), tell me which sections to expand and I'll update `model_card_readme.md` accordingly.
