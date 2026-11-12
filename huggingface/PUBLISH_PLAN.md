# Publication plan

The Hugging Face account is being created. Do not invent a namespace or live URL.

Upload to a model repository named `qwen-russian-gec-lora`:

- The existing `qwen_gec_q4.gguf` from Downloads (986,045,248 bytes).
- This directory's `README.md` as the model repository's root Model Card, after replacing its draft-status wording.
- `model-manifest.json` and applicable upstream license / attribution files, once finalized.

Do not upload dataset text, logs, notebook outputs, optimizer state or intermediate checkpoints. Keep GGUF out of GitHub. The final LoRA adapter is available in `soup_training_backup.zip`: `soup_qwen_gec_output/adapter_model.safetensors` (8,731,128 bytes) and `adapter_config.json`. Offer it as a separate adapter repository with the matching tokenizer files; do not confuse it with merged GGUF weights.

After publication, verify the remote file size/hash and the model page, then replace the pending-publication text in the GitHub README with the verified URL. The Model Card already links back to the existing GitHub repository. No publication has been performed by preparing these files.
