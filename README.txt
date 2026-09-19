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
