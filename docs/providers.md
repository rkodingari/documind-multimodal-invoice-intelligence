# Extraction providers

Every provider consumes the same page images, OCR/PDF tokens, and bounding boxes and returns the same
Pydantic schema. Check runtime availability at `GET /api/v1/providers`.

## Rules baseline

```bash
DOCUMIND_EXTRACTION_PROVIDER=rules
```

No model or credential. This is the explainable fallback and comparison baseline.

## Learned CPU ranker

Generate training data and train the committed scikit-learn pipeline:

```bash
python scripts/generate_invoices.py --count 180 --scan-every 0 --seed 17 \
  --output training/generated
python scripts/train_ranker.py
```

The saved artifact is `models/candidate_ranker.joblib`. The classifier ranks field candidates using
label proximity, semantic type, relative page position, OCR confidence, and text/layout features.

## OpenAI paid multimodal API

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

Set current input/output prices explicitly if cost estimation is required:

```bash
DOCUMIND_OPENAI_INPUT_COST_PER_MILLION=...
DOCUMIND_OPENAI_OUTPUT_COST_PER_MILLION=...
```

Pages are sent as base64 image inputs to the Responses API with a strict JSON schema. Cost remains
`null` unless prices are explicitly configured because provider pricing changes over time.

## Ollama local vision model

```bash
ollama pull gemma3:4b
ollama serve
DOCUMIND_OLLAMA_MODEL=gemma3:4b
```

The adapter calls the local `/api/chat` endpoint with page images, temperature zero, and the shared
JSON schema. Change `DOCUMIND_OLLAMA_BASE_URL` for a non-default server.

## Hugging Face pretrained VLM

```bash
pip install -e ".[gpu]"
DOCUMIND_HUGGINGFACE_VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
```

The model is loaded through the Transformers `image-text-to-text` pipeline. A CUDA-capable GPU is
recommended. Model weights are downloaded from Hugging Face on first use and are not committed.

## Fine-tuned LayoutLMv3

```bash
pip install -e ".[gpu,finetune]"
python scripts/finetune_layoutlm.py --epochs 3
```

The training pipeline aligns ground-truth values to OCR/PDF words, normalizes boxes to LayoutLM's
0–1000 coordinate space, and fine-tunes `LayoutLMv3ForTokenClassification`. The resulting checkpoint
is read from `models/layoutlmv3-finetuned`. Training and evaluation results must only be reported after
running on actual GPU hardware.

References: [OpenAI vision inputs](https://developers.openai.com/api/docs/guides/images-vision),
[OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[Ollama vision](https://docs.ollama.com/capabilities/vision),
[Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs),
[Hugging Face Qwen2.5-VL](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl), and
[Hugging Face LayoutLMv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3).

