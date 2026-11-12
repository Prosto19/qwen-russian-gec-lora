---
language:
- ru
base_model: Qwen/Qwen2.5-1.5B-Instruct
datasets:
- alexanderpl/ru_gec_v1
pipeline_tag: text-generation
tags:
- gguf
- qwen2
- lora
- grammatical-error-correction
---

# Qwen Russian GEC — Q4_K_M GGUF

**Experimental / DON'T SHIP.** A Qwen2.5-1.5B-Instruct derivative fine-tuned with LoRA for Russian grammatical error correction, merged and exported with Soup CLI. The Q4_K_M GGUF artifact is provided for experimental local inference. Recorded evaluation is archival, not a fresh benchmark.

[Code, notebook, results and reports](https://github.com/Prosto19/qwen-russian-gec-lora)

## Artifact

- Filename: `qwen_gec_q4.gguf`
- Format: GGUF v3, Q4_K_M; architecture `qwen2`, 338 tensors, 28 blocks.
- Size: 986,045,248 bytes (940.4 MiB).
- SHA-256: `eff44839a9e6a0e568afaaa6bfd33d019ecd61116769fe72eacbd3b6dc649b95`
- Recorded base revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

The GGUF header and file digest were checked locally. Model inference was not rerun for this publication preparation. No original export-time checksum is available to establish a cryptographic chain from the historical evaluation to this file.

## Training

Soup CLI 0.77.2 on Colab Tesla T4, 2 epochs / 338 steps, FP16 LoRA r=16, alpha=32, dropout=0.05, learning rate 1e-4, batch size 1, gradient accumulation 8, maximum sequence length 512, AdamW, gradient checkpointing, seed 42.

The preparation report describes 1,500 training, 150 validation and 150 test rows from `alexanderpl/ru_gec_v1`. Soup split the training file internally (`val_split: 0.1`), using 1,350 training and 150 validation rows. The separate validation file was not used. The complete environment lock and data-preparation code are not included in the public project, so exact end-to-end retraining is not guaranteed.

## Recorded evaluation

| Artifact | Exact match |
|---|---:|
| Base / Transformers | 0/20 |
| LoRA / Transformers | 6/20 |
| Merged FP16 / Transformers | 6/20 |
| F16 GGUF / llama.cpp | 6/20 |
| Q4_K_M GGUF / llama.cpp | 3/20 |

Metric: stripped prediction equals stripped reference, without case or punctuation normalization. The same 20 test IDs are used. Prompt wording and engines differ across stages; the experiment does not isolate every source of change. F16 and Q4 outputs agree on 16/20 examples. A tiny sample and noisy references limit interpretation; exact match is not human semantic accuracy.

## Local use

Use a compatible llama.cpp installation. After downloading the GGUF:

```sh
llama-server -m qwen_gec_q4.gguf -c 2048 -ngl 0 -t 2 --host 127.0.0.1 --port 18080
```

Send chat requests to `/v1/chat/completions` with temperature 0 and max_tokens 384. Historical GGUF prompt: `Исправь ошибки в тексте. Верни только исправленный текст, сохранив исходный смысл`, followed by two newlines and the input. The notebook contains the complete request. This command follows the historical invocation; it has not been retested on this machine.

## Limitations and intended use

For learning, reproducibility research and inspection of model conversion failures. Not a reliable production corrector. The model can change meaning, preserve mistakes, or introduce new errors. Assistant-only label masking needs further validation. Training loss and adapter activation do not prove generalization. Use human review and a larger, cleaner evaluation before deployment.

## Attribution and licensing

Base model: Qwen Team, [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Dataset: [alexanderpl/ru_gec_v1](https://huggingface.co/datasets/alexanderpl/ru_gec_v1). Their current cards identify Apache-2.0. Preserve applicable upstream license and attribution obligations when distributing derivative weights. The included LICENSE reproduces the Apache-2.0 license from the recorded base-model revision. See NOTICE for attribution and modification details. No warranty is provided.

Training/export: Soup CLI; GGUF runtime: llama.cpp. The technical reports describe the experimenter's work and AI assistance. No affiliation or endorsement is implied.
