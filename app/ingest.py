"""CLI de ingesta: docs/*.pdf -> Chroma (texto + imágenes) + extracción de formularios.

Uso:
    docker compose run --rm app python -m app.ingest
    docker compose run --rm app python -m app.ingest --reset
"""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from langchain_experimental.text_splitter import SemanticChunker
from tqdm import tqdm

from .config import settings
from .embeddings import get_embeddings
from .extraction import process_form
from .llm import describe_image
from .pdf_processing import extract_images, extract_text, render_first_page
from .vectorstore import get_vectorstore


def ingest_text(pdf_path: Path, store, splitter: SemanticChunker) -> int:
    pages = list(extract_text(pdf_path))
    chunks, metadatas, ids = [], [], []
    for page in tqdm(pages, desc="  páginas", leave=False, unit="pág"):
        for chunk in splitter.split_text(page.text):
            chunks.append(chunk)
            metadatas.append({"type": "text", "source": pdf_path.name, "page": page.page})
            ids.append(str(uuid.uuid4()))
    if chunks:
        store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


def ingest_images(pdf_path: Path, store) -> int:
    images = list(extract_images(pdf_path, settings.images_dir))
    count = 0
    for img in tqdm(images, desc="  imágenes", leave=False, unit="img"):
        try:
            description = describe_image(img.path)
        except Exception as exc:  # noqa: BLE001 - una imagen ilegible no debe parar la ingesta
            tqdm.write(f"  ! No se pudo describir {img.path.name}: {exc}")
            continue
        store.add_texts(
            texts=[description],
            metadatas=[{
                "type": "image",
                "source": pdf_path.name,
                "page": img.page,
                "image_path": str(img.path),
            }],
            ids=[str(uuid.uuid4())],
        )
        count += 1
    return count


def ingest_pdf(pdf_path: Path, store, splitter: SemanticChunker) -> None:
    n_text = ingest_text(pdf_path, store, splitter)
    tqdm.write(f"  texto: {n_text} chunks indexados")

    n_img = ingest_images(pdf_path, store)
    tqdm.write(f"  imágenes: {n_img} descripciones indexadas")

    first_page = render_first_page(pdf_path, settings.images_dir)
    try:
        form_json = process_form(pdf_path.stem, first_page)
    except Exception as exc:  # noqa: BLE001 - un fallo del LLM no debe parar la ingesta
        tqdm.write(f"  ! No se pudo analizar el formulario: {exc}")
        return
    if form_json:
        tqdm.write(f"  formulario detectado -> {form_json}")
    else:
        tqdm.write("  no es formulario (sin extracción estructurada)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta de PDFs para el RAG.")
    parser.add_argument("--reset", action="store_true", help="Vacía la colección antes de ingestar")
    args = parser.parse_args()

    store = get_vectorstore()
    if args.reset:
        print("Reseteando la colección...")
        store.reset_collection()

    pdfs = sorted(settings.docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No se encontraron PDFs en {settings.docs_dir}")
        return

    splitter = SemanticChunker(get_embeddings(), breakpoint_threshold_type="percentile")

    for pdf_path in tqdm(pdfs, desc="PDFs", unit="pdf"):
        tqdm.write(f"\n== {pdf_path.name} ==")
        ingest_pdf(pdf_path, store, splitter)

    print(f"\nIngesta completada: {len(pdfs)} PDF(s).")


if __name__ == "__main__":
    main()
