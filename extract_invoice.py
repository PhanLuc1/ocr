"""
Extract invoice number from Vietnamese receipt/invoice photos.

Pipeline:
    image -> RapidOCR (en, ONNX) -> ASCII normalize -> fuzzy label match -> regex value

Design choice: use English/Latin OCR + unidecode normalization. We don't need
Vietnamese diacritics because:
  - Invoice numbers / IDs are pure [A-Z0-9/]
  - Labels can be fuzzy-matched on ASCII form ("ma kh" instead of "Mã KH")
  - Dropping diacritics shrinks the error surface AND speeds up OCR

Usage:
    python extract_invoice.py samples/tgdd_receipt.jpg
"""
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from rapidfuzz import fuzz
from rapidocr_onnxruntime import RapidOCR
from unidecode import unidecode

MAX_SIDE = 1600  # resize long side before OCR


@dataclass
class OcrLine:
    text: str
    bbox: list
    score: float = 1.0

    @property
    def text_ascii(self) -> str:
        # unidecode: "Số" → "So", "đ" → "d", etc. No letter-digit split so values
        # like "PB06040002688" stay contiguous.
        return re.sub(r"\s+", " ", unidecode(self.text).lower()).strip()


@dataclass
class Extraction:
    doc_type: str
    invoice_number: Optional[str]
    extra: dict = field(default_factory=dict)


_OCR_SINGLETON: Optional[RapidOCR] = None


def get_ocr() -> RapidOCR:
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        _OCR_SINGLETON = RapidOCR()
    return _OCR_SINGLETON


def _resize_for_ocr(image_path: str) -> str:
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) <= MAX_SIDE:
        return image_path
    scale = MAX_SIDE / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    out = Path(image_path).with_suffix(".resized.jpg")
    img.convert("RGB").save(out, "JPEG", quality=90)
    return str(out)


def run_ocr(image_path: str) -> List[OcrLine]:
    small = _resize_for_ocr(image_path)
    result, _elapsed = get_ocr()(small)
    return [
        OcrLine(text=text.strip(), bbox=bbox, score=float(score))
        for bbox, text, score in (result or [])
    ]


def _ascii(s: str) -> str:
    return re.sub(r"\s+", " ", unidecode(s).lower()).strip()


TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_config() -> Tuple[dict, list, dict, dict, list]:
    """Load TEMPLATES / GENERIC_FIELDS / PRIMARY_FIELD / GENERIC_PRIORITY and
    classifier triggers from YAML files in templates/. Called at import time.

    Returns (templates, triggers, primary_field, generic_fields, generic_priority).
    """
    import yaml

    templates: Dict[str, Dict[str, Tuple[List[str], str]]] = {}
    triggers: List[Tuple[str, List[str]]] = []
    primary_field: Dict[str, str] = {}
    generic_fields: Dict[str, Tuple[List[str], str]] = {}
    generic_priority: List[str] = []

    for path in sorted(TEMPLATES_DIR.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        fields = {
            name: (cfg["labels"], cfg["value_regex"])
            for name, cfg in (data.get("fields") or {}).items()
        }
        if path.name.startswith("_generic"):
            generic_fields = fields
            generic_priority = data.get("priority", [])
            continue
        doc_type = data["doc_type"]
        templates[doc_type] = fields
        triggers.append((doc_type, data.get("triggers", [])))
        if "primary_field" in data:
            primary_field[doc_type] = data["primary_field"]
    return templates, triggers, primary_field, generic_fields, generic_priority


TEMPLATES, _TRIGGERS, PRIMARY_FIELD, GENERIC_FIELDS, GENERIC_PRIORITY = _load_config()


def classify_document(lines: List[OcrLine]) -> str:
    """Trigger-based classification. YAML files define the trigger keywords —
    first doc type whose any trigger appears in the (ascii) text wins."""
    full = _ascii(" ".join(l.text for l in lines))
    compact = full.replace(" ", "")  # OCR often concatenates words
    for doc_type, keywords in _TRIGGERS:
        for kw in keywords:
            # Match against full (with spaces) OR compact — handles OCR concat
            if kw in full or kw.replace(" ", "") in compact:
                return doc_type
    return "unknown"


def _build_label_regex(variants: List[str]) -> re.Pattern:
    """Compile variants into a non-letter-boundary regex. Longer variants first
    so 'ma khach hang' matches before 'ma kh' at the same position.

    Non-letter boundary (?:^|[^a-z]) + (?=[^a-z]|$) is stricter than \\b:
      - matches 'MKH 106026' and 'MKH106026' (digit is non-letter)
      - rejects 'MA KH' inside 'MA KHACH' (next char 'a' is a letter)
    """
    sorted_variants = sorted(variants, key=len, reverse=True)
    patterns = []
    for v in sorted_variants:
        parts = v.split()
        inner = r"\s*".join(re.escape(p) for p in parts)
        pat = r"(?:^|[^a-z])" + inner + r"(?=[^a-z]|$)"
        patterns.append(pat)
    return re.compile("|".join(patterns), re.IGNORECASE)


def _value_near_label(
    lines: List[OcrLine],
    label_variants: List[str],
    value_regex: str,
    exclude_lines: Optional[set] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """Find a line containing one of `label_variants`, then take the first value
    that matches `value_regex` either:
      (a) inline, same line after the label — covers 'MKH 106026', 'Số: 494538'
      (b) on one of the next 1-3 lines — covers 'Phiếu Y/C:' + '00360SO...' split
    Returns (value, line_index_of_label) so callers can mark lines as consumed.
    """
    exclude_lines = exclude_lines or set()
    label_re = _build_label_regex(label_variants)
    value_pat = re.compile(value_regex)
    body = value_regex.lstrip("^").rstrip("$")
    inline_val = re.compile(body)

    for i, line in enumerate(lines):
        if i in exclude_lines:
            continue
        ascii_line = line.text_ascii
        m = label_re.search(ascii_line)
        if not m:
            continue
        # (a) inline: strip whitespace from rest so "PB 06040002688" captures together
        rest = re.sub(r"\s+", "", ascii_line[m.end():]).upper()
        hit = inline_val.search(rest)
        if hit and value_pat.match(hit.group(0)):
            return hit.group(0), i
        # (b) next lines in reading order
        for j in range(i + 1, min(i + 4, len(lines))):
            if j in exclude_lines:
                continue
            cand = lines[j].text.strip().replace(" ", "").upper()
            if value_pat.match(cand):
                return cand, i
    return None, None


def _fields_for(doc_type: str) -> Dict[str, Tuple[List[str], str]]:
    """Merge GENERIC_FIELDS with the doc-type template. Generic goes first so
    more-specific labels (ma so thue) consume their line before shorter ones (so)."""
    combined: Dict[str, Tuple[List[str], str]] = dict(GENERIC_FIELDS)
    combined.update(TEMPLATES.get(doc_type, {}))
    return combined


def extract_from_lines(lines: List[OcrLine]) -> Extraction:
    """Core extraction logic. Separate from OCR so callers (e.g. the UI) can run
    OCR once and reuse lines for display + extraction."""
    doc_type = classify_document(lines)
    fields = _fields_for(doc_type)

    used: set = set()
    extracted: dict = {}
    for key, (variants, regex) in fields.items():
        value, idx = _value_near_label(lines, variants, regex, exclude_lines=used)
        extracted[key] = value
        if idx is not None:
            used.add(idx)

    primary_key = PRIMARY_FIELD.get(doc_type)
    invoice_number = extracted.get(primary_key) if primary_key else None
    if not invoice_number:
        invoice_number = next(
            (extracted[k] for k in list(GENERIC_PRIORITY) + list(fields.keys()) if extracted.get(k)),
            None,
        )

    extra = {k: v for k, v in extracted.items() if v}
    if doc_type == "unknown":
        extra["raw_lines"] = [l.text for l in lines]
    return Extraction(doc_type=doc_type, invoice_number=invoice_number, extra=extra)


def extract(image_path: str) -> Extraction:
    return extract_from_lines(run_ocr(image_path))


# Backward-compat helpers for app.py (uses EXTRACTORS[doc_type](lines))
def _extract_wrap(lines: List[OcrLine], doc_type: str) -> Extraction:
    fields = _fields_for(doc_type)
    used: set = set()
    extracted: dict = {}
    for key, (variants, regex) in fields.items():
        value, idx = _value_near_label(lines, variants, regex, exclude_lines=used)
        extracted[key] = value
        if idx is not None:
            used.add(idx)
    primary_key = PRIMARY_FIELD.get(doc_type)
    invoice_number = extracted.get(primary_key) if primary_key else None
    if not invoice_number:
        invoice_number = next((v for v in extracted.values() if v), None)
    return Extraction(doc_type=doc_type, invoice_number=invoice_number,
                      extra={k: v for k, v in extracted.items() if v})


extract_tgdd = lambda lines: _extract_wrap(lines, "tgdd_receipt")
extract_vat_invoice = lambda lines: _extract_wrap(lines, "vat_invoice")
extract_water_sms = lambda lines: _extract_wrap(lines, "water_sms")

EXTRACTORS = {
    "tgdd_receipt": extract_tgdd,
    "vat_invoice": extract_vat_invoice,
    "water_sms": extract_water_sms,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_invoice.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        sys.exit(1)
    print(f"\nProcessing: {image_path}")
    print("-" * 60)
    r = extract(image_path)
    print(f"Document type   : {r.doc_type}")
    print(f"Invoice number  : {r.invoice_number}")
    print(f"Extra fields    : {r.extra}")


if __name__ == "__main__":
    main()
