#!/usr/bin/env python3
import argparse
import io
import json
import re
from pathlib import Path

from PIL import Image

from documind.extraction.document import extract_document_text
from documind.extraction.normalization import detect_currency, normalize_date, normalize_number
from documind.ml.layoutlm import normalize_box
from documind.ml.ranker import TARGET_FIELDS

LABELS = ["O"] + [prefix + field.upper() for field in TARGET_FIELDS for prefix in ("B-", "I-")]


def token_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def matching_indices(tokens, field: str, truth: object) -> list[int]:
    if field in {"subtotal", "tax", "total"}:
        return [
            index
            for index, token in enumerate(tokens)
            if normalize_number(token.text) == normalize_number(truth)
        ][:1]
    if field == "invoice_date":
        return [
            index for index, token in enumerate(tokens) if normalize_date(token.text) == str(truth)
        ][:1]
    if field == "currency":
        return [
            index for index, token in enumerate(tokens) if detect_currency(token.text)[0] == truth
        ][:1]
    target = [token_key(part) for part in str(truth).split() if token_key(part)]
    words = [token_key(token.text) for token in tokens]
    for start in range(len(words) - len(target) + 1):
        if words[start : start + len(target)] == target:
            return list(range(start, start + len(target)))
    return []


def build_examples(dataset: Path):
    records = json.loads((dataset / "ground_truth.json").read_text())
    examples = []
    for record in records:
        artifact = extract_document_text(dataset / record["filename"], "application/pdf")
        for page in artifact.pages:
            labels = ["O"] * len(page.tokens)
            for field in TARGET_FIELDS:
                indices = matching_indices(page.tokens, field, record["ground_truth"][field])
                for position, index in enumerate(indices):
                    labels[index] = ("B-" if position == 0 else "I-") + field.upper()
            examples.append(
                {
                    "image": Image.open(io.BytesIO(page.image_png)).convert("RGB"),
                    "words": [token.text for token in page.tokens],
                    "boxes": [
                        normalize_box(token.bbox, page.width, page.height) for token in page.tokens
                    ],
                    "labels": [LABELS.index(label) for label in labels],
                }
            )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 for invoice fields")
    parser.add_argument("--dataset", type=Path, default=Path("training/generated"))
    parser.add_argument("--base-model", default="microsoft/layoutlmv3-base")
    parser.add_argument("--output", type=Path, default=Path("models/layoutlmv3-finetuned"))
    parser.add_argument("--epochs", type=float, default=3.0)
    args = parser.parse_args()
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install .[gpu,finetune] before fine-tuning LayoutLMv3") from exc

    examples = build_examples(args.dataset)
    processor = LayoutLMv3Processor.from_pretrained(args.base_model, apply_ocr=False)
    id2label = dict(enumerate(LABELS))
    label2id = {label: index for index, label in id2label.items()}
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    class InvoiceDataset(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            item = examples[index]
            encoding = processor(
                item["image"],
                item["words"],
                boxes=item["boxes"],
                word_labels=item["labels"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            return {key: value.squeeze(0) for key, value in encoding.items()}

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=InvoiceDataset())
    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    (args.output / "documind_training_metadata.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "examples": len(examples),
                "epochs": args.epochs,
                "labels": LABELS,
            },
            indent=2,
        )
    )
    print(f"Saved fine-tuned LayoutLMv3 checkpoint to {args.output}")


if __name__ == "__main__":
    main()
