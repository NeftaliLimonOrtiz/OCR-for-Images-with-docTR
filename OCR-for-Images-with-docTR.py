import os
import cv2
import re
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

PDF_PATH = "prueba"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def merge_close(values, gap=10):
    if not values:
        return []
    values = sorted(values)
    groups = [[values[0]]]
    for v in values[1:]:
        if abs(v - np.mean(groups[-1])) <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(np.mean(g)) for g in groups]


def find_interval(value, bounds):
    for i in range(len(bounds) - 1):
        if bounds[i] <= value < bounds[i + 1]:
            return i
    return max(0, len(bounds) - 2)

def collect_words(page, img_w, img_h):
    words = []
    for block in page["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                text = word["value"].strip()
                if not text:
                    continue
                (x1, y1), (x2, y2) = word["geometry"]
                x1p, y1p = int(x1 * img_w), int(y1 * img_h)
                x2p, y2p = int(x2 * img_w), int(y2 * img_h)
                words.append({
                    "text": text,
                    "x1": x1p, "y1": y1p, "x2": x2p, "y2": y2p,
                    "xc": (x1p + x2p) / 2, "yc": (y1p + y2p) / 2,
                    "w": x2p - x1p,        "h":  y2p - y1p,
                })
    return words

def get_lines_mask(gray, h_kernel, v_kernel):
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 8)
    H = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, h_kernel))
    V = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, v_kernel))
    return H, V


def count_lines(mask, axis, min_ratio):
    total = mask.shape[1] if axis == "h" else mask.shape[0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(
        1 for c in contours
        if cv2.boundingRect(c)[2 if axis == "h" else 3] > total * min_ratio
    )

def detect_table_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    H, V = get_lines_mask(gray, (max(30, w // 25), 1), (1, max(8, h // 60)))

    grid = cv2.dilate(cv2.add(H, V),
                      cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)

    regions = []
    for cnt in cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww < w * 0.15 or hh < h * 0.03:
            continue
        h_lines = count_lines(H[y:y+hh, x:x+ww], "h", 0.45)
        v_lines = count_lines(V[y:y+hh, x:x+ww], "v", 0.30)
        if (h_lines >= 3 and v_lines >= 3) or (h_lines >= 2 and v_lines >= 2):
            pad = 8
            regions.append((max(0, x-pad), max(0, y-pad),
                            min(w, x+ww+pad) - max(0, x-pad),
                            min(h, y+hh+pad) - max(0, y-pad)))

    return merge_overlapping_regions(regions)


def merge_overlapping_regions(regions, margin=0):
    if not regions:
        return []
    merged = []
    for x, y, w, h in sorted(regions, key=lambda r: (r[1], r[0])):
        x2, y2 = x + w, y + h
        for m in merged:
            if not (x > m[2]+margin or x2 < m[0]-margin or
                    y > m[3]+margin or y2 < m[1]-margin):
                m[:] = [min(m[0],x), min(m[1],y), max(m[2],x2), max(m[3],y2)]
                break
        else:
            merged.append([x, y, x2, y2])
    return [(x1, y1, x2-x1, y2-y1) for x1, y1, x2, y2 in merged]


def detect_grid(region_img):
    gray = cv2.cvtColor(region_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    H, V = get_lines_mask(gray, (max(20, w // 18), 1), (1, max(8, h // 10)))

    def extract_coords(mask, axis, min_ratio):
        coords = []
        for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, ww, hh = cv2.boundingRect(cnt)
            dim = ww if axis == "h" else hh
            total = w if axis == "h" else h
            if dim > total * min_ratio:
                coords += ([y, y+hh] if axis == "h" else [x, x+ww])
        return coords

    def to_bounds(coords, size):
        coords = merge_close(coords, gap=10)
        if not coords:
            return [0, size]
        return sorted(set(([0] if coords[0] > 5 else []) + coords +
                          ([size] if coords[-1] < size - 5 else [])))

    return (to_bounds(extract_coords(V, "v", 0.30), w),
            to_bounds(extract_coords(H, "h", 0.30), h))

def has_dark_header(region_img, threshold=90, ratio=0.25, top=0.22):
    gray = cv2.cvtColor(region_img, cv2.COLOR_RGB2GRAY)
    top_h = max(1, int(gray.shape[0] * top))
    row_means = np.mean(gray[:top_h, :], axis=1)
    return np.mean(row_means < threshold) >= ratio


def infer_columns(words, region_bbox, tolerance=20, top_portion=0.35):
    rx, ry, rw, rh = region_bbox
    limit = ry + int(rh * top_portion)
    xcs = sorted(w["xc"] - rx for w in words
                 if rx <= w["xc"] <= rx + rw and ry <= w["yc"] <= limit)
    if len(xcs) < 2:
        return [0, rw]
    groups = [[xcs[0]]]
    for x in xcs[1:]:
        if abs(x - np.mean(groups[-1])) <= tolerance:
            groups[-1].append(x)
        else:
            groups.append([x])
    centers = [int(np.mean(g)) for g in groups if g]
    if len(centers) < 2:
        return [0, rw]
    bounds = [0] + [int((centers[i]+centers[i+1])/2) for i in range(len(centers)-1)] + [rw]
    return sorted(set(merge_close(bounds, gap=30)))


def group_words_by_lines(words, factor=0.6):
    lines = []
    for w in sorted(words, key=lambda x: x["yc"]):
        for line in lines:
            ly = np.mean([lw["yc"] for lw in line])
            lh = np.mean([lw["h"] for lw in line])
            if abs(w["yc"] - ly) <= factor * max(lh, w["h"]):
                line.append(w)
                break
        else:
            lines.append([w])
    return [sorted(l, key=lambda x: x["x1"]) for l in
            sorted(lines, key=lambda l: np.mean([w["yc"] for w in l]))]


def cell_text(words):
    return "\n".join(" ".join(w["text"] for w in line)
                     for line in group_words_by_lines(words, 0.5)).strip()

def assign_to_cells(words, region_bbox, x_lines, y_lines):
    rx, ry, rw, rh = region_bbox
    nr, nc = len(y_lines)-1, len(x_lines)-1
    table = [[[] for _ in range(nc)] for _ in range(nr)]

    for w in words:
        if not (rx <= w["xc"] <= rx+rw and ry <= w["yc"] <= ry+rh):
            continue
        ci = find_interval(w["xc"]-rx, x_lines)
        ri = find_interval(w["yc"]-ry, y_lines)
        if 0 <= ri < nr and 0 <= ci < nc:
            table[ri][ci].append(w)

    table = next((table[i:] for i, row in enumerate(table)
                  if sum(1 for c in row if c) >= 2), table)

    nr = len(table)  

    return [[cell_text(table[r][c]) for c in range(nc)] for r in range(nr)]

def table_to_text(table):
    if not table:
        return []
    nc = max(len(r) for r in table)
    widths = [max((max(len(p) for p in (c or "").split("\n")) if c else 0)
                  for row in table for c in [row[i] if i < len(row) else ""])
              for i in range(nc)]
    lines = []
    for ri, row in enumerate(table):
        row = row + [""] * (nc - len(row))
        cells = [c.split("\n") for c in row]
        height = max(len(c) for c in cells)
        cells = [c + [""] * (height - len(c)) for c in cells]
        for si in range(height):
            lines.append(" | ".join(cells[c][si].ljust(widths[c]) for c in range(nc)))
        if ri < len(table) - 1:
            lines.append("-+-".join("-" * w for w in widths))
    return lines

def count_valid_words(words):
    return sum(
        1 for w in words
        if re.search(r"[A-Za-zÀ-ÿ]{2,}", w["text"])
    )

def process_pdf(pdf_path, model, min_words=50):
    print(f"\nProcesando: {pdf_path}")
    doc = DocumentFile.from_pdf(pdf_path)
    all_text = []
    for page_idx, img in enumerate(doc, 1):
        page_data = model([img]).export()["pages"][0]
        h, w = img.shape[:2]
        words = collect_words(page_data, w, h)

        if count_valid_words(words) < 10:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            page_data_rot = model([rotated]).export()["pages"][0]
            h2, w2 = rotated.shape[:2]
            words_rot = collect_words(page_data_rot, w2, h2)

            # 👇 comparar calidad
            if count_valid_words(words_rot) > count_valid_words(words):
                img = rotated
                page_data = page_data_rot
                words = words_rot

        regions = detect_table_regions(img)
        blocks, valid_regions = [], []

        for region in regions:
            rx, ry, rw, rh = region
            region_img = img[ry:ry+rh, rx:rx+rw]
            x_lines, y_lines = detect_grid(region_img)

            if has_dark_header(region_img):
                inferred = infer_columns(words, region)
                if len(inferred) >= 3:
                    x_lines = inferred

            x_lines = sorted(set(merge_close(x_lines, gap=30)))
            if len(x_lines) < 2 or len(y_lines) < 2:
                continue

            table = assign_to_cells(words, region, x_lines, y_lines)
            if not table or len(table) < 2 or len(table[0]) < 2:
                continue

            blocks.append({"type": "table", "y": ry, "content": table_to_text(table)})
            valid_regions.append(region)

        free_words = [w for w in words if not any(
            rx-2 <= w["xc"] <= rx+rw+2 and ry-2 <= w["yc"] <= ry+rh+2
            for rx, ry, rw, rh in valid_regions
        )]
        for line in group_words_by_lines(free_words):
            text = " ".join(w["text"] for w in line).strip()
            if text:
                blocks.append({"type": "text", "y": min(w["y1"] for w in line), "content": [text]})

        all_text.append(f"\n===== PAGE {page_idx} =====")
        for block in sorted(blocks, key=lambda b: b["y"]):
            if block["type"] == "table":
                all_text += ["", *block["content"], ""]
            else:
                all_text += block["content"]

    out = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(pdf_path))[0] + ".txt")
    open(out, "w", encoding="utf-8").write("\n".join(all_text))
    print(f"Save: {out}")

def main():
    pdfs = ([PDF_PATH] if os.path.isfile(PDF_PATH)
            else sorted(f for f in (os.path.join(PDF_PATH, n)
                        for n in os.listdir(PDF_PATH)) if f.endswith(".pdf")))
    if not pdfs:
        return
    model = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
    for pdf in pdfs:
        try:
            process_pdf(pdf, model)
        except Exception as e:
            print(f"Error {pdf}: {e}")

if __name__ == "__main__":
    main()