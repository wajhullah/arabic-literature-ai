"""Extract clean Classical Arabic text from PDFs under data/raw_pdfs/.

Pipeline step 1 — Ingestion:
    1. Read every PDF in data/raw_pdfs/.
    2. Extract text page by page with pdfplumber (handles complex RTL
       layouts and line wrapping better than PyPDF2).
    3. Clean the extracted text: collapse excessive whitespace while
       preserving Arabic characters (saved as UTF-8).
    4. Optionally fall back to easyocr for scanned/image-only pages.
    5. Write one plain-text file per PDF into data/processed_text/,
       named after the original PDF (same stem, ``.txt`` suffix).

Usage:
    python extract_text.py                      # digital text only
    python extract_text.py --ocr                # also OCR scanned pages (150 DPI)
    python extract_text.py --ocr --ocr-dpi 300  # OCR at higher quality (slower)

Dependencies:
    pdfplumber  (required)
    easyocr     (optional; only imported when ``--ocr`` is used)
    PyMuPDF     (optional; renders pages for easyocr when ``--ocr`` is used)
"""

from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path

import pdfplumber

# Paths are resolved relative to this file so the script works regardless of
# the current working directory (i.e. it points at ../data/... from scripts/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PDFS_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
PROCESSED_TEXT_DIR = PROJECT_ROOT / "data" / "processed_text"

# Unicode characters treated as "horizontal whitespace" during cleanup.
# This includes the non-breaking space and other space variants that PDF
# extractors commonly emit inside Arabic text.
_EXTRA_SPACES = (
    "\u00a0"   # no-break space
    "\u2000"   # en quad
    "\u2001"   # em quad
    "\u2002"   # en space
    "\u2003"   # em space
    "\u2004"   # three-per-em space
    "\u2005"   # four-per-em space
    "\u2006"   # six-per-em space
    "\u2007"   # figure space
    "\u2008"   # punctuation space
    "\u2009"   # thin space
    "\u200a"   # hair space
    "\u200b"   # zero-width space
    "\u202f"   # narrow no-break space
    "\u205f"   # medium mathematical space
    "\u3000"   # ideographic space
    "\ufeff"   # zero-width no-break space (BOM)
)

# Unicode bidirectional/format characters that are preserved during cleanup.
# Arabic text relies on these marks to render correctly, and stripping them
# can corrupt mixed RTL/LTR ordering. All other format characters (soft
# hyphen, ZWJ/ZWNJ, etc.) are removed as whitespace/format noise.
_PRESERVED_FORMAT_CHARS = frozenset(
    {
        "\u061c",  # ARABIC LETTER MARK (ALM)
        "\u200e",  # LEFT-TO-RIGHT MARK (LRM)
        "\u200f",  # RIGHT-TO-LEFT MARK (RLM)
        "\u2066",  # LEFT-TO-RIGHT ISOLATE
        "\u2067",  # RIGHT-TO-LEFT ISOLATE
        "\u2068",  # FIRST STRONG ISOLATE
        "\u2069",  # POP DIRECTIONAL ISOLATE
    }
)


def setup_logging() -> None:
    """Configure standard-library logging with concise, readable output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def clean_text(text: str) -> str:
    """Normalize whitespace in extracted text without damaging Arabic.

    The goal is *whitespace* hygiene only — no aggressive Unicode
    normalization (e.g. NFKC) is applied, because that can rewrite Arabic
    presentation forms and ligatures (e.g. lam-alef) in unintended ways.

    Steps:
        1. Normalize line endings and tabs.
        2. Map exotic Unicode spaces to a single regular space.
        3. Drop control characters (except newlines).
        4. Collapse runs of spaces to one space.
        5. Strip per-line padding and remove blank lines.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    for space in _EXTRA_SPACES:
        text = text.replace(space, " ")

    # Remove control characters while preserving newlines and the bidi marks
    # that Arabic RTL text depends on.
    text = "".join(
        ch
        for ch in text
        if ch == "\n"
        or ch in _PRESERVED_FORMAT_CHARS
        or not unicodedata.category(ch).startswith("C")
    )

    lines: list[str] = []
    for raw_line in text.split("\n"):
        # Collapse any run of spaces/tabs into a single space, then trim.
        line = " ".join(raw_line.split())
        if line:
            lines.append(line)

    return "\n".join(lines)


def extract_page_text(page) -> str:
    """Extract and clean the text of a single pdfplumber page.

    Returns the cleaned text, or an empty string if the page produced
    nothing usable (e.g. it is a scanned image or otherwise corrupted).
    """
    # ``extract_text`` can raise on malformed content streams; isolate that
    # so a single bad page does not abort the whole PDF.
    raw = page.extract_text() or ""
    return clean_text(raw)


def build_ocr_reader(lang: str):
    """Lazily create an easyocr Reader (heavy import + model load)."""
    import os

    import torch

    # Use every CPU core for the detection/recognition networks; torch
    # defaults to 2 threads, which is needlessly slow on multicore machines.
    torch.set_num_threads(os.cpu_count() or 1)

    import easyocr  # imported here so the optional dependency stays optional

    return easyocr.Reader([lang], gpu=False, verbose=False)


def ocr_page(fitz_page, reader, dpi: int = 150) -> str:
    """OCR a scanned/image-only page with easyocr.

    Rasterizes the page with PyMuPDF (fitz) and returns the recognized
    text joined by newlines. A grayscale raster is passed to easyocr,
    which it consumes natively without any color-channel ambiguity.

    ``dpi`` defaults to 150: on CPU this is roughly twice as fast as 300
    DPI with little measurable quality loss for classical Arabic print
    (see the 300/200/150 DPI benchmark in the project history).
    """
    import numpy as np
    import pymupdf as fitz  # PyMuPDF — optional, only needed for OCR

    pix = fitz_page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 1:
        image = np.squeeze(image, axis=-1)  # easyocr handles 2D grayscale natively

    # batch_size=8 batches per-line recognition; measured ~10% faster on CPU
    # than the default of 1, with byte-identical output.
    results = reader.readtext(image, detail=0, paragraph=True, batch_size=8)
    return "\n".join(results)


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    use_ocr: bool,
    ocr_lang: str,
    ocr_dpi: int,
) -> Path:
    """Extract one PDF into a cleaned UTF-8 ``.txt`` file.

    Returns the path of the written output file.
    """
    out_path = output_dir / f"{pdf_path.stem}.txt"
    page_blocks: list[str] = []

    # Lazy-loaded OCR resources — created only if a page actually needs them.
    reader = None
    fitz_doc = None
    scanned_page_count = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logging.info("Opened %s (%d pages)", pdf_path.name, total_pages)

            for index, page in enumerate(pdf.pages, start=1):
                logging.info("Processing Page %d of %d...", index, total_pages)

                try:
                    text = extract_page_text(page)
                except Exception as exc:  # noqa: BLE001 — isolate per-page failures
                    logging.warning(
                        "Failed to extract text from page %d of %s: %s",
                        index,
                        pdf_path.name,
                        exc,
                    )
                    continue

                if text:
                    page_blocks.append(text)
                    continue

                # No digital text — likely a scanned image. Warn and optionally OCR.
                if not use_ocr:
                    logging.warning(
                        "Page %d of %s contains no extractable text (likely scanned). "
                        "Re-run with --ocr to enable easyocr fallback.",
                        index,
                        pdf_path.name,
                    )
                    continue

                if reader is None:
                    logging.info("Initializing easyocr (language=%s)...", ocr_lang)
                    reader = build_ocr_reader(ocr_lang)

                if fitz_doc is None:
                    import pymupdf as fitz  # PyMuPDF — optional, only needed for OCR

                    logging.info("Initializing PyMuPDF renderer...")
                    fitz_doc = fitz.open(str(pdf_path))

                try:
                    text = clean_text(ocr_page(fitz_doc[index - 1], reader, dpi=ocr_dpi))
                except Exception as exc:  # noqa: BLE001
                    logging.warning(
                        "OCR failed for page %d of %s: %s",
                        index,
                        pdf_path.name,
                        exc,
                    )
                    continue

                if text:
                    scanned_page_count += 1
                    page_blocks.append(text)
                else:
                    logging.warning(
                        "OCR returned no text for page %d of %s.", index, pdf_path.name
                    )
    finally:
        if fitz_doc is not None:
            fitz_doc.close()

    full_text = "\n\n".join(page_blocks).strip()
    if not full_text:
        logging.warning(
            "No text could be extracted from %s — output will be empty.", pdf_path.name
        )

    # UTF-8 by construction: ``encoding="utf-8"`` preserves Arabic code points.
    out_path.write_text(full_text + "\n", encoding="utf-8")

    logging.info(
        "Saved %s (%d chars, %d scanned pages OCR'd)",
        out_path.name,
        len(full_text),
        scanned_page_count,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract clean Arabic text from PDFs in data/raw_pdfs/."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_PDFS_DIR,
        help="Directory containing PDFs (default: data/raw_pdfs/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_TEXT_DIR,
        help="Directory for extracted .txt files (default: data/processed_text/).",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable easyocr fallback for scanned/image-only pages.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="ar",
        help="Language code passed to easyocr (default: ar).",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=150,
        help="Rasterization DPI for OCR (default: 150; use 300 for max quality).",
    )
    args = parser.parse_args()

    setup_logging()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        logging.error("Input directory does not exist: %s", input_dir)
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        logging.warning("No PDF files found in %s", input_dir)
        return

    logging.info("Found %d PDF(s) in %s", len(pdf_paths), input_dir)
    logging.info("Output directory: %s", output_dir)
    logging.info("OCR fallback: %s", "enabled" if args.ocr else "disabled")

    for pdf_path in pdf_paths:
        try:
            process_pdf(
                pdf_path,
                output_dir,
                use_ocr=args.ocr,
                ocr_lang=args.ocr_lang,
                ocr_dpi=args.ocr_dpi,
            )
        except Exception as exc:  # noqa: BLE001 — keep the pipeline going per file
            logging.error("Could not process %s: %s", pdf_path.name, exc)

    logging.info("Done.")


if __name__ == "__main__":
    main()
