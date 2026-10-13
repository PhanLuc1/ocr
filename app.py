"""Streamlit UI to try the invoice-number extractor on arbitrary receipts.

Run with:
    .venv/bin/streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from extract_invoice import extract_from_lines, get_ocr, run_ocr


st.set_page_config(page_title="Invoice Number Extractor", layout="wide")


@st.cache_resource(show_spinner="Loading PaddleOCR model (one-time)...")
def warm_up_ocr():
    return get_ocr()


def draw_boxes(image: Image.Image, lines) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for line in lines:
        pts = [(p[0], p[1]) for p in line.bbox]
        draw.polygon(pts, outline="red", width=2)
    return img


st.title("Invoice Number Extractor — PoC")
st.caption("Upload hoá đơn / biên nhận, hệ thống tự nhận diện loại và trích xuất số hoá đơn.")

warm_up_ocr()

uploaded = st.file_uploader(
    "Chọn ảnh hoá đơn (jpg/png/webp)",
    type=["jpg", "jpeg", "png", "webp"],
)

sample_col1, sample_col2 = st.columns(2)
with sample_col1:
    if st.button("Dùng mẫu: TGDD receipt"):
        uploaded = Path("samples/tgdd_receipt.jpg").read_bytes()
        st.session_state["sample"] = ("tgdd_receipt.jpg", uploaded)
with sample_col2:
    if st.button("Dùng mẫu: HEPCO VAT invoice"):
        uploaded = Path("samples/hepco_invoice.png").read_bytes()
        st.session_state["sample"] = ("hepco_invoice.png", uploaded)

if uploaded is None and "sample" in st.session_state:
    name, data = st.session_state["sample"]
    file_bytes = data
    filename = name
elif uploaded is not None and hasattr(uploaded, "read"):
    file_bytes = uploaded.read()
    filename = uploaded.name
elif uploaded is not None:
    file_bytes = uploaded
    filename = "uploaded"
else:
    st.info("Hãy upload 1 ảnh hoặc chọn 1 mẫu ở trên để bắt đầu.")
    st.stop()

suffix = Path(filename).suffix or ".jpg"
with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    tmp.write(file_bytes)
    tmp_path = tmp.name

col_img, col_out = st.columns([3, 2])

with st.spinner("OCR + extracting..."):
    lines = run_ocr(tmp_path)
    result = extract_from_lines(lines)

original = Image.open(tmp_path)
annotated = draw_boxes(original, lines)

with col_img:
    st.subheader("Ảnh gốc + bounding boxes")
    st.image(annotated, use_container_width=True)

with col_out:
    st.subheader("Kết quả")
    st.metric("Document type", result.doc_type)
    if result.invoice_number:
        st.metric("Số hoá đơn", result.invoice_number)
    else:
        st.warning("Không tìm được số hoá đơn trong ảnh.")
    st.write("**Các field khác:**")
    st.json({k: v for k, v in result.extra.items() if k != "raw_lines"})

    with st.expander("Raw OCR output"):
        for line in lines:
            st.text(line.text)
