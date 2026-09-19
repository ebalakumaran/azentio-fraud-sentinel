# Fraud Sentinel: Data Validation and QLoRA Fine-Tuning

## Objective
To build a Python pipeline that cleans and joins banking data, prepares safe transaction evidence, and produces validated JSON fraud-risk assessments using an open-weight language model under 3 billion parameters.

---

## Why Qwen2.5-1.5B-Instruct?
- **Meets the constraint:** approximately 1.54 billion parameters.
- **Accessible:** public Hugging Face weights without gated-access approval.
- **Practical for our hardware:** successfully loaded on a free Colab Tesla T4.
- **Suitable starting point:** instruction-tuned with structured-output capabilities.

Llama was an example in the brief, rather than a mandatory model. We selected Qwen for accessibility and resource constraints—not because we established that it outperforms Llama.

---

## My Approach

> **Three CSVs → Clean and normalize → Validate relational joins → Prepare allowlisted evidence → Record baseline outputs → Fine-tune with QLoRA → Compare before/after → Validate JSON and supporting reasons**

1. **Clean:** parse amounts, dates, and booleans; preserve missing-value flags.
2. **Join:** connect transactions, accounts, and customers without losing or multiplying transactions.
3. **Prepare evidence:** calculate past-only behavioral features and exclude raw PII and arbitrary text from model prompts.
4. **Establish a baseline:** save raw outputs, validation results, and runtime.
5. **Fine-tune:** train LoRA adapters over the frozen 4-bit model.
6. **Compare:** evaluate the same held-out inputs with adapters disabled and enabled.
7. **Validate:** enforce output types and reject unsupported reason codes. Application code renders the final explanation from validated reasons.

---

## QLoRA: Memory-Efficient Fine-Tuning

We used **4-bit NF4 quantization with double quantization**, FP16 computation, rank-8 LoRA adapters, and gradient checkpointing.

The base weights remain frozen; only the adapters are trained.

| Measured item | Result |
|---|---:|
| Trainable parameters | 9,232,384 — **0.5945%** |
| Quantized model footprint before adapter preparation | 1.045 GiB |
| Prepared model footprint with adapters | 1.514 GiB |
| Peak PyTorch GPU allocation during training | 2.623 GiB |
| Training examples | 160 synthetic examples |
| Training epochs | 2 |
| Training time | 241.9 seconds |

*These are measured footprints, not a claim of total memory savings against a separately benchmarked full fine-tuning run.*

---

## Before vs After Fine-Tuning

| Evaluation | Before | After |
|---|:---:|:---:|
| Schema-valid outputs on six real records | 6/6 | 6/6 |
| Supported reason codes on six real records | 5/6 | 6/6 |
| Policy agreement on 12 unseen synthetic cases | 10/12 | 11/12 |

The synthetic comparison showed **two corrected cases and one regression**. All six final guarded sample predictions passed validation, and all ten implemented deterministic guardrail checks passed.

---

## Guardrails
- Validated joins and original-row mapping.
- Explicit missing, invalid, and ambiguous-data flags.
- Allowlisted evidence fields and categorical values.
- Raw notes, merchant names, and customer PII excluded from prompts.
- Strict JSON types, confidence bounds, and supported-reason validation.
- Failed predictions routed to review rather than silently marked safe.

---

## Scope and Limitations
The supplied files contained **no verified fraud labels**. Fine-tuning used explicitly synthetic screening-policy examples. Results demonstrate output compliance and policy learning—not verified real-world fraud accuracy. Confidence scores are uncalibrated.

The pipeline processed 1,000 source transaction rows, retaining 988 unique transactions and a mapping to every original row. Model inference was demonstrated on selected records; full-dataset predictions are not included.
