from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image


# -----------------------------
# Utilities
# -----------------------------

def summarize_text(text: str, max_len: int = 220) -> str:
    """
    Simple summarization to avoid OCR / heavy NLP.
    For grading, you can later replace with a better summarizer.
    """
    t = (text or "").strip()
    if len(t) > max_len:
        return t[:max_len].rstrip() + "..."
    return t


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_path_for_url(path: str) -> str:
    """
    Normalize filesystem path to forward slashes.
    Note: in main.py we convert 'storage/..' -> '/storage/..'
    """
    return path.replace("\\", "/")


def bbox_intersects(a: List[float], b: List[float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def looks_like_footer_or_noise(text: str) -> bool:
    """
    Filter out common PDF noise. Keep this conservative to avoid deleting valid content.
    """
    s = (text or "").strip()
    if not s:
        return True


def crop_graph_region(image_path: str) -> Optional[Image.Image]:
    """
    Detect and isolate chart/graph, remove surrounding text and labels.
    Uses edge detection, morphological operations, and contour analysis.
    Highlights the chart by removing text annotations.
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        original_shape = img.shape
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance chart
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Edge detection to find chart boundaries
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Morphological operations to connect chart regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return Image.open(image_path)
        
        # Sort contours by area and find the largest one (chart area)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Get bounding box of the largest contour
        largest_contour = contours[0]
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Minimum size filter (at least 20% of image)
        min_area = (original_shape[0] * original_shape[1]) * 0.2
        if cv2.contourArea(largest_contour) < min_area:
            return Image.open(image_path)
        
        # Expand region slightly to include chart borders but exclude outer text
        padding = 15
        x = max(0, x - padding)
        y = max(0, y - padding)
        x_max = min(original_shape[1], x + w + 2 * padding)
        y_max = min(original_shape[0], y + h + 2 * padding)
        
        # Crop the chart region
        cropped = img[y:y_max, x:x_max]
        
        # Remove remaining text using thresholding and morphological operations
        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, text_mask = cv2.threshold(gray_crop, 240, 255, cv2.THRESH_BINARY)
        
        # Dilate to remove small text artifacts
        kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_mask = cv2.dilate(text_mask, kernel_text, iterations=2)
        
        # Apply inpainting to remove detected text (replace with background)
        result = cv2.inpaint(cropped, cv2.bitwise_not(text_mask), 3, cv2.INPAINT_TELEA)
        
        # Enhance the chart colors
        result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        result_hsv[:, :, 1] = cv2.multiply(result_hsv[:, :, 1], 1.2)  # Increase saturation
        result_hsv[:, :, 2] = cv2.multiply(result_hsv[:, :, 2], 1.1)  # Increase brightness
        result_enhanced = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
        
        # Convert back to PIL Image
        result_rgb = cv2.cvtColor(result_enhanced, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None




    # Page numbers or short numeric axis labels
    if re.fullmatch(r"[\d\s,.\-]+", s) and len(s) <= 20:
        return True

    # Very common headers/footers
    lower = s.lower()
    if "diw weekly report" in lower:
        return True
    if "© diw" in lower or "doi:" in lower:
        return True
    if lower.startswith("issn "):
        return True

    return False


# -----------------------------
# TEXT EXTRACTION (PDF)
# -----------------------------

def _group_words_into_lines(words: List[Dict[str, Any]], y_tolerance: float = 3.0) -> List[List[Dict[str, Any]]]:
    """
    Group words into lines based on their 'top' coordinate.
    pdfplumber words have: text, x0, x1, top, bottom, etc.
    """
    if not words:
        return []

    # Sort words top-to-bottom then left-to-right
    words_sorted = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))

    lines: List[List[Dict[str, Any]]] = []
    current_line: List[Dict[str, Any]] = []
    current_top: Optional[float] = None

    for w in words_sorted:
        top = float(w["top"])
        if current_top is None:
            current_top = top
            current_line = [w]
            continue

        # Same line if close in y-direction
        if abs(top - current_top) <= y_tolerance:
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
            current_top = top

    if current_line:
        lines.append(current_line)

    # Sort each line by x0
    for line in lines:
        line.sort(key=lambda w: float(w["x0"]))

    return lines


def _line_to_text(line_words: List[Dict[str, Any]]) -> str:
    """
    Convert list of words into a spaced line string.
    """
    parts = []
    for w in line_words:
        t = str(w.get("text", "")).strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


def _estimate_line_font_size(line_words: List[Dict[str, Any]]) -> float:
    """
    pdfplumber's extract_words does not always include font size unless extra_attrs are used.
    We'll fallback to line height as a proxy.
    """
    if not line_words:
        return 0.0
    heights = [float(w["bottom"]) - float(w["top"]) for w in line_words]
    return sum(heights) / max(len(heights), 1)


def _classify_line(text: str, font_proxy: float) -> str:
    """
    Heuristic classification: title/subtitle/paragraph/list_item/caption.
    Keep generic (not document-specific).
    """
    s = (text or "").strip()
    if not s:
        return "paragraph"

    # Bullet list
    if s.startswith(("•", "-", "–")):
        return "list_item"

    # Captions: "Figure 1", "Table 2", etc.
    if re.match(r"^(Figure|Fig\.|Table)\s*\d+", s, flags=re.IGNORECASE):
        return "caption"

    # Headings by font proxy (line height)
    # Not "hardcoded content"; these are general typographic heuristics.
    if font_proxy >= 14:
        return "title"
    if font_proxy >= 11:
        return "subtitle"

    return "paragraph"


def _extract_text_components_from_page(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    """
    Extract text components with basic structure:
    - group words into lines
    - classify line types
    - merge consecutive paragraph lines
    """
    components: List[Dict[str, Any]] = []

    # Extract words with coordinates
    # keep simple; fontname/size may not always exist
    words = page.extract_words() or []
    lines = _group_words_into_lines(words, y_tolerance=3.0)

    # Build line objects
    line_objs = []
    for line in lines:
        text = _line_to_text(line)
        if looks_like_footer_or_noise(text):
            continue

        font_proxy = _estimate_line_font_size(line)
        line_type = _classify_line(text, font_proxy)

        # approximate bbox of the whole line
        x0 = float(min(w["x0"] for w in line))
        x1 = float(max(w["x1"] for w in line))
        top = float(min(w["top"] for w in line))
        bottom = float(max(w["bottom"] for w in line))

        line_objs.append({
            "type": line_type,
            "text": text,
            "bbox": [x0, top, x1, bottom]
        })

    # Merge consecutive paragraph lines into one component
    merged: List[Dict[str, Any]] = []
    buffer_text = ""
    buffer_bbox: Optional[List[float]] = None
    buffer_type: Optional[str] = None

    def flush_buffer():
        nonlocal buffer_text, buffer_bbox, buffer_type
        if buffer_text.strip():
            merged.append({
                "type": buffer_type or "paragraph",
                "content": buffer_text.strip(),
                "summary": summarize_text(buffer_text),
                "position": {"page": page.page_number, "bbox": buffer_bbox}
            })
        buffer_text = ""
        buffer_bbox = None
        buffer_type = None

    for obj in line_objs:
        t = obj["type"]
        txt = obj["text"]
        bbox = obj["bbox"]

        # For titles/subtitles/captions/list items: keep separate components
        if t in {"title", "subtitle", "caption", "list_item"}:
            flush_buffer()
            merged.append({
                "type": t,
                "content": txt,
                "summary": summarize_text(txt),
                "position": {"page": page.page_number, "bbox": bbox}
            })
            continue

        # Paragraph merging
        if buffer_type is None:
            buffer_type = "paragraph"

        if not buffer_text:
            buffer_text = txt
            buffer_bbox = bbox
        else:
            # merge bbox
            bx0, btop, bx1, bbot = buffer_bbox or bbox
            x0, top, x1, bottom = bbox
            buffer_bbox = [min(bx0, x0), min(btop, top), max(bx1, x1), max(bbot, bottom)]
            buffer_text += " " + txt

    flush_buffer()
    components.extend(merged)
    return components


# -----------------------------
# TABLE EXTRACTION (PDF)
# -----------------------------

def _is_reasonable_table(table_data: List[List[Any]]) -> bool:
    """
    Reject false positives from pdfplumber.
    """
    if not table_data or len(table_data) < 2:
        return False

    max_cols = max((len(r) for r in table_data), default=0)
    if max_cols < 2:
        return False

    cells = [c for r in table_data for c in r]
    non_empty = [c for c in cells if c and str(c).strip()]
    if len(non_empty) / max(len(cells), 1) < 0.2:
        return False

    # Avoid "table" with extremely long text blobs in one cell (often false detection)
    for c in non_empty:
        if isinstance(c, str) and len(c) > 1500:
            return False

    return True


def _extract_tables_from_page(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []

    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 20,
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
    }

    try:
        found = page.find_tables(table_settings=table_settings)
    except Exception:
        found = []

    words = page.extract_words() or []

    for t in found:
        data = t.extract()
        if not _is_reasonable_table(data):
            continue

        bbox = [float(t.bbox[0]), float(t.bbox[1]), float(t.bbox[2]), float(t.bbox[3])]

        # collect words inside bbox (no OCR)
        in_words = []
        for w in words:
            wx0, wx1 = float(w["x0"]), float(w["x1"])
            wtop, wbot = float(w["top"]), float(w["bottom"])
            if wx0 >= bbox[0] and wx1 <= bbox[2] and wtop >= bbox[1] and wbot <= bbox[3]:
                in_words.append({"text": w["text"], "bbox": [wx0, wtop, wx1, wbot]})

        flat_text = " ".join([w["text"] for w in in_words]).strip()

        tables.append({
            "page": page.page_number,
            "bbox": bbox,
            "data": data,
            "extracted_texts": in_words,
            "summary": summarize_text(flat_text) or "Table extracted"
        })

    return tables


# -----------------------------
# GRAPHICS EXTRACTION (PDF)
# -----------------------------

def _extract_embedded_images(doc: fitz.Document, page_index: int, out_dir: str) -> List[Dict[str, Any]]:
    """
    Extract embedded images (if present).
    Crops to remove text/labels, keeping only the chart/graph region.
    """
    page = doc[page_index]
    page_no = page_index + 1

    images = page.get_images(full=True)
    if not images:
        return []

    ensure_dir(os.path.join(out_dir, "images"))
    results: List[Dict[str, Any]] = []

    for img_idx, img in enumerate(images):
        xref = img[0]
        try:
            base = doc.extract_image(xref)
            img_bytes = base["image"]
            ext = base.get("ext", "png")
        except Exception:
            continue

        rects = page.get_image_rects(xref) or []
        for rect_i, r in enumerate(rects):
            file_name = f"p{page_no}_img{img_idx}_{rect_i}.{ext}"
            file_path_full = os.path.join(out_dir, "images", file_name)
            
            # Save original
            with open(file_path_full, "wb") as f:
                f.write(img_bytes)
            
            # Crop to remove text/labels
            cropped = crop_graph_region(file_path_full)
            if cropped:
                cropped.save(file_path_full)  # Overwrite with cropped version
            
            file_path = normalize_path_for_url(file_path_full)

            results.append({
                "page": page_no,
                "bbox": [float(r.x0), float(r.y0), float(r.x1), float(r.y1)],
                "file_path": file_path,
                "extracted_texts": [],
                "summary": "Embedded image extracted (text removed)"
            })

    return results


def _extract_vector_graphics(doc: fitz.Document, page_index: int, out_dir: str) -> List[Dict[str, Any]]:
    """
    Extract vector graphics by detecting drawing regions.
    Crops extracted images to keep only the chart/graph portion.
    """
    page = doc[page_index]
    page_no = page_index + 1

    drawings = page.get_drawings()
    if not drawings:
        return []

    # Collect larger drawing rectangles
    rects = []
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        area = float(r.width * r.height)
        if area < 10_000:  # filter tiny noise
            continue
        rects.append(r)

    if not rects:
        return []

    # Merge intersecting rects (simple greedy union)
    merged: List[fitz.Rect] = []
    for r in rects:
        merged_into_existing = False
        for i in range(len(merged)):
            if merged[i].intersects(r):
                merged[i] = merged[i] | r  # union
                merged_into_existing = True
                break
        if not merged_into_existing:
            merged.append(r)

    ensure_dir(os.path.join(out_dir, "graphics"))
    results: List[Dict[str, Any]] = []

    for idx, r in enumerate(merged):
        # Render clip region
        try:
            pix = page.get_pixmap(clip=r, dpi=150)
        except Exception:
            continue

        file_name = f"p{page_no}_vec_{idx}.png"
        file_path_full = os.path.join(out_dir, "graphics", file_name)
        pix.save(file_path_full)
        
        # Crop to remove text/labels
        cropped = crop_graph_region(file_path_full)
        if cropped:
            cropped.save(file_path_full)  # Overwrite with cropped version
        
        file_path = normalize_path_for_url(file_path_full)

        results.append({
            "page": page_no,
            "bbox": [float(r.x0), float(r.y0), float(r.x1), float(r.y1)],
            "file_path": file_path,
            "extracted_texts": [],
            "summary": "Vector graphic extracted (text removed)"
        })

    return results


# -----------------------------
# Public API: PDF
# -----------------------------

def extract_from_pdf(file_path: str, storage_dir: str) -> Dict[str, Any]:
    """
    Main PDF extractor:
    - structured text components
    - tables (validated)
    - graphics (embedded images + vector drawings)
    """
    ensure_dir(storage_dir)

    components: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []
    graphics: List[Dict[str, Any]] = []

    # TEXT + TABLES
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # page.page_number is 1-based in pdfplumber
            components.extend(_extract_text_components_from_page(page))
            tables.extend(_extract_tables_from_page(page))

    # GRAPHICS (images + vectors)
    doc = fitz.open(file_path)
    try:
        for page_index in range(len(doc)):
            graphics.extend(_extract_embedded_images(doc, page_index, storage_dir))
            graphics.extend(_extract_vector_graphics(doc, page_index, storage_dir))
    finally:
        doc.close()

    # Normalize file paths
    for g in graphics:
        g["file_path"] = normalize_path_for_url(g["file_path"])

    return {
        "components": components,
        "tables": tables,
        "graphics": graphics
    }
