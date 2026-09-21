import io
import shutil
from dataclasses import dataclass
from pathlib import Path

import fitz
import pytesseract
from PIL import Image, ImageOps

SUPPORTED_CONTENT_TYPES = {"application/pdf", "image/png", "image/jpeg"}
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}


@dataclass(slots=True)
class DocumentToken:
    text: str
    page: int
    bbox: tuple[float, float, float, float]
    confidence: float
    block: int = 0
    line: int = 0


@dataclass(slots=True)
class PageArtifact:
    page: int
    width: float
    height: float
    image_png: bytes
    tokens: list[DocumentToken]
    text: str


@dataclass(slots=True)
class DocumentArtifact:
    text: str
    method: str
    page_count: int
    pages: list[PageArtifact]
    path: Path
    content_type: str


# Backwards-compatible name used by existing tests and integrations.
DocumentText = DocumentArtifact


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def _prepare_image(image: Image.Image) -> Image.Image:
    if not tesseract_available():
        raise RuntimeError("Tesseract is required for scanned documents but is not installed")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) < 1800:
        scale = 1800 / max(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    return image


def _ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(_prepare_image(image), config="--psm 6")


def _ocr_image_with_tokens(
    image: Image.Image, page_number: int, page_width: float, page_height: float
) -> tuple[str, list[DocumentToken]]:
    prepared = _prepare_image(image)
    data = pytesseract.image_to_data(
        prepared, config="--psm 6", output_type=pytesseract.Output.DICT
    )
    x_scale = page_width / prepared.width
    y_scale = page_height / prepared.height
    tokens: list[DocumentToken] = []
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        confidence = float(data["conf"][index])
        if not text or confidence < 0:
            continue
        x = float(data["left"][index]) * x_scale
        y = float(data["top"][index]) * y_scale
        width = float(data["width"][index]) * x_scale
        height = float(data["height"][index]) * y_scale
        tokens.append(
            DocumentToken(
                text=text,
                page=page_number,
                bbox=(x, y, x + width, y + height),
                confidence=max(0.0, min(1.0, confidence / 100)),
                block=int(data["block_num"][index]),
                line=int(data["line_num"][index]),
            )
        )
    grouped: dict[tuple[int, int], list[str]] = {}
    for token in tokens:
        grouped.setdefault((token.block, token.line), []).append(token.text)
    text = "\n".join(" ".join(words) for words in grouped.values())
    return text, tokens


def _page_image(page: fitz.Page, scale: float = 2.0) -> tuple[bytes, Image.Image]:
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image_png = pix.tobytes("png")
    return image_png, Image.open(io.BytesIO(image_png)).convert("RGB")


def _pdf_tokens(page: fitz.Page, page_number: int) -> list[DocumentToken]:
    tokens: list[DocumentToken] = []
    for word in page.get_text("words", sort=True):
        x0, y0, x1, y1, text, block, line, _word = word
        tokens.append(
            DocumentToken(
                text=str(text),
                page=page_number,
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                confidence=1.0,
                block=int(block),
                line=int(line),
            )
        )
    return tokens


def extract_document_text(path: Path, content_type: str) -> DocumentArtifact:
    if content_type == "application/pdf" or path.suffix.lower() == ".pdf":
        document = fitz.open(path)
        try:
            pages: list[PageArtifact] = []
            used_ocr = False
            used_text = False
            for page_index, page in enumerate(document, start=1):
                image_png, image = _page_image(page)
                text = page.get_text("text", sort=True).strip()
                if len("".join(text.split())) >= 20:
                    tokens = _pdf_tokens(page, page_index)
                    used_text = True
                else:
                    text, tokens = _ocr_image_with_tokens(
                        image, page_index, float(page.rect.width), float(page.rect.height)
                    )
                    used_ocr = True
                pages.append(
                    PageArtifact(
                        page=page_index,
                        width=float(page.rect.width),
                        height=float(page.rect.height),
                        image_png=image_png,
                        tokens=tokens,
                        text=text,
                    )
                )
            method = "hybrid" if used_ocr and used_text else "ocr" if used_ocr else "text"
            return DocumentArtifact(
                text="\n".join(page.text for page in pages).strip(),
                method=method,
                page_count=document.page_count,
                pages=pages,
                path=path,
                content_type=content_type,
            )
        finally:
            document.close()

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        text, tokens = _ocr_image_with_tokens(
            image, page_number=1, page_width=float(image.width), page_height=float(image.height)
        )
        page = PageArtifact(
            1, float(image.width), float(image.height), buffer.getvalue(), tokens, text
        )
        return DocumentArtifact(text, "ocr", 1, [page], path, content_type)
