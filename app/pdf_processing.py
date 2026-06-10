"""Procesado de PDFs con PyMuPDF: texto, imágenes embebidas y render de página."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

# Ignora imágenes diminutas (iconos, líneas decorativas).
MIN_IMAGE_SIDE = 64


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class ExtractedImage:
    page: int
    path: Path


def extract_text(pdf_path: Path) -> list[PageText]:
    pages: list[PageText] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(PageText(page=i + 1, text=text))
    return pages


def extract_images(pdf_path: Path, out_dir: Path) -> list[ExtractedImage]:
    """Extrae imágenes embebidas significativas y las guarda en ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images: list[ExtractedImage] = []
    stem = pdf_path.stem
    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_index, full=True)):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.width < MIN_IMAGE_SIDE or pix.height < MIN_IMAGE_SIDE:
                    pix = None
                    continue
                if pix.colorspace is None or pix.colorspace.name not in ("DeviceGray", "DeviceRGB"):
                    # CMYK, Indexed, Lab, máscaras de imagen, etc. -> a RGB para poder guardar como PNG
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                out_path = out_dir / f"{stem}_p{page_index + 1}_{img_index}.png"
                pix.save(out_path)
                pix = None
                images.append(ExtractedImage(page=page_index + 1, path=out_path))
    return images


def render_first_page(pdf_path: Path, out_dir: Path, zoom: float = 2.0) -> Path:
    """Renderiza la primera página a PNG (para la detección de formulario)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_path.stem}_page1.png"
    with fitz.open(pdf_path) as doc:
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        pix.save(out_path)
    return out_path
