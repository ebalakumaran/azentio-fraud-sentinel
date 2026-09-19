Paste this into your GitHub **`README.md`**:

````markdown
# Fraud Sentinel: Data Validation and QLoRA Fine-Tuning

## Objective

Build a Python pipeline that cleans and joins banking data, prepares safe
transaction evidence, and produces validated JSON fraud-risk assessments
using an open-weight language model under 3 billion parameters.

## Why Qwen2.5-1.5B-Instruct?

- **Meets the constraint:** approximately 1.54 billion parameters.
- **Accessible:** public Hugging Face weights without gated-access approval.
- **Practical for the hardware:** successfully loaded on a free Colab Tesla T4.
- **Suitable starting point:** instruction-tuned with structured-output capabilities.

Llama was an example in the brief, rather than a mandatory model.
I selected Qwen for accessibility and resource constraints. No comparative
benchmark against Llama was performed.

## My Approach

1. **Clean:** parse amounts, dates, and booleans; preserve missing-value flags.
2. **Join:** connect transactions, accounts, and customers without losing or
   multiplying transactions.
3. **Prepare evidence:** calculate past-only behavioral features and exclude
   raw PII and arbitrary text from model prompts.
4. **Establish a baseline:** save raw outputs, validation results, and runtime.
5. **Fine-tune:** train LoRA adapters over the frozen 4-bit model.
6. **Compare:** evaluate the same held-out inputs with adapters disabled and enabled.
7. **Validate:** enforce output types and reject unsupported reason codes.
   Application code renders the final explanation from validated reasons.

## Pipeline

```mermaid
flowchart TD
    A["Transactions, accounts, customers"] --> B["Clean and normalize"]
    B --> C["Validate joins and prepare safe evidence"]
    C --> D["Baseline: adapters disabled"]
    C --> E["After training: adapters enabled"]
    T["Synthetic training examples"] --> F["QLoRA fine-tuning"]
    F --> E
    D --> G["Compare the same held-out inputs"]
    E --> G
    G --> H["Validate JSON and supporting reasons"]
    H --> I["Predictions and review audit"]
````

## QLoRA: Memory-Efficient Fine-Tuning

I used **4-bit NF4 quantization with double quantization**, FP16 computation,
rank-8 LoRA adapters, and gradient checkpointing.

The base weights remain frozen; only the adapters are trained.

| Measured item                                        |                  Result |
| ---------------------------------------------------- | ----------------------: |
| Trainable parameters                                 | 9,232,384 — **0.5945%** |
| Quantized model footprint before adapter preparation |               1.045 GiB |
| Prepared model footprint with adapters               |               1.514 GiB |
| Peak PyTorch GPU allocation during training          |               2.623 GiB |
| Training examples                                    |  160 synthetic examples |
| Training epochs                                      |                       2 |
| Training time                                        |           241.9 seconds |

These are measured footprints. A full fine-tuning memory baseline was not
separately benchmarked.

## Before vs After Fine-Tuning

| Evaluation                                                 | Before | After |
| ---------------------------------------------------------- | -----: | ----: |
| Schema-valid outputs on six supplied transaction records   |    6/6 |   6/6 |
| Supported reason codes on six supplied transaction records |    5/6 |   6/6 |
| Policy agreement on 12 unseen synthetic cases              |  10/12 | 11/12 |

The synthetic comparison showed **two corrected cases and one regression**.
All six final guarded sample predictions passed validation, and all ten
implemented deterministic guardrail checks passed.

The same model, test inputs, prompts, and generation settings were used
within each paired comparison. Adapters were disabled for the baseline
and enabled for the fine-tuned evaluation.

## Guardrails

* Validated joins and original-row mapping.
* Explicit missing, invalid, and ambiguous-data flags.
* Allowlisted evidence fields and categorical values.
* Raw notes, merchant names, and customer PII excluded from prompts.
* Strict JSON types, confidence bounds, and supported-reason validation.
* Failed predictions routed to review rather than silently marked safe.

These checks test specific defenses; they do not establish resistance
to every possible attack.

## Output Format

Each successful prediction contains:

* `transaction_id`: identifier bound to the input by application code.
* `is_fraud`: model classification as a boolean.
* `confidence`: uncalibrated confidence in the selected classification.
* `justification`: a sentence rendered from validated model-selected reasons.

Raw responses, validation errors, and review flags are stored separately
in the audit output.

## Running in Google Colab

1. Open `fraud_sentinel.ipynb` in Google Colab.
2. Enable a GPU runtime; the implementation was run on a Tesla T4.
3. Upload `transactions.csv`, `accounts.csv`, and `customers.csv` to `/content`.
4. Install the notebook dependencies. Restart the session if required
   before importing the model libraries.
5. Run the implementation cells in order.
6. Download the outputs from `/content/fraud_sentinel_outputs`.

The base model is downloaded from Hugging Face.
The saved QLoRA adapter requires that base model and is not a standalone model.
Package versions are recorded in `environment.json`.

## Scope and Limitations

The supplied files contained **no verified fraud labels**. Fine-tuning used
explicitly synthetic screening-policy examples. Results demonstrate output
compliance and policy learning—not verified real-world fraud accuracy.

The synthetic test examples were unseen but generated from the same
illustrative policy as the training examples. The evaluation is small,
and confidence scores are uncalibrated.

The pipeline processed 1,000 source transaction rows, retaining 988 unique
transactions and a mapping to every original row. Model inference was
demonstrated on selected records; full-dataset predictions are not included.

Historical features reflect only the available transaction data.
Ambiguous dates are flagged, and date-format assumptions are documented.

```
```
