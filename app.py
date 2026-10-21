"""
TomaTech - Classificador de qualidade de tomates (backend Flask).

Tudo o que é análise visual considera SOMENTE a região do tomate:
o fundo é removido por segmentação (cor + GrabCut) antes de qualquer cálculo.

Execução:
    pip install -r requirements.txt
    python app.py
    -> abra http://127.0.0.1:5000
"""
import base64
import os
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

try:
    from skimage.feature import local_binary_pattern
    SKIMAGE = True
except ImportError:
    SKIMAGE = False

try:
    import serial
    import serial.tools.list_ports
    SERIAL = True
except ImportError:
    SERIAL = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Pasta do dataset (pode ser alterada pela variável de ambiente TOMATO_DATASET)
DATASET_DIR = os.environ.get("TOMATO_DATASET", os.path.join(BASE_DIR, "dataset"))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
SIZE = 400  # todas as imagens são normalizadas para 400x400

# ---- Parâmetros de qualidade (ajuste aqui conforme o seu dataset) ----
SIZE_MIN_OK, SIZE_MAX_OK = 18000, 50000   # área padrão do tomate (px, imagem 400x400)
SPOT_OK_PCT = 1.0          # abaixo disso: saudável
SPOT_REJECT_PCT = 2.0      # a partir disso: reprovado (manchas/podre)
SPOT_BIG_PCT = 1.0         # uma única mancha maior que isso (% do tomate) reprova
SPOT_DARKNESS = 0.60       # pixel é mancha se < 60% do brilho local do tomate
REJECT_SMALL = True        # tomate "Pequeno" vai para a Cesta B
REJECT_UNRIPE = True       # tomate verde vai para a Cesta B

app = Flask(__name__, static_folder=None)


# ----------------------------------------------------------------------------
# Utilidades de imagem
# ----------------------------------------------------------------------------
def to_data_url(img, rgb=True, ext=".png"):
    """Converte imagem numpy em data URL base64."""
    if img.ndim == 3 and img.shape[2] == 3 and rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[2] == 4 and rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(ext, img)
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf).decode()


def decode_data_url(data_url):
    raw = data_url.split(",", 1)[1] if "," in data_url else data_url
    arr = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagem inválida.")
    return img


def prepare_image(img_bgr, return_frame=False):
    """Redimensiona para caber em SIZE x SIZE SEM deformar (letterbox).
    Antes a imagem era esticada para 400x400, o que distorcia formato e proporção."""
    h, w = img_bgr.shape[:2]
    s = SIZE / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r = cv2.resize(img_bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)
    rh, rw = r.shape[:2]
    frame_px = rh * rw  # área útil da foto (sem as bordas de preenchimento)
    if (rh, rw) == (SIZE, SIZE):
        return (r, frame_px) if return_frame else r
    border = np.concatenate([r[0], r[-1], r[:, 0], r[:, -1]])
    pad = tuple(int(v) for v in np.median(border, axis=0))
    top, left = (SIZE - rh) // 2, (SIZE - rw) // 2
    out = cv2.copyMakeBorder(r, top, SIZE - rh - top, left, SIZE - rw - left,
                             cv2.BORDER_CONSTANT, value=pad)
    return (out, frame_px) if return_frame else out


def local_dark_spots(gray, region):
    """Manchas = pixels bem mais escuros que o brilho LOCAL do tomate.
    Comparar com a média global confundia sombra/curvatura com doença."""
    g = gray.astype(np.float32)
    reg = (region > 0)
    valid = reg.astype(np.float32)
    spots = np.zeros(gray.shape, bool)
    for _ in range(2):  # 2ª passada ignora as manchas ao calcular o brilho local
        num = cv2.GaussianBlur(g * valid, (0, 0), 15)
        den = cv2.GaussianBlur(valid, (0, 0), 15)
        local = num / np.maximum(den, 1e-3)
        spots = (g < local * SPOT_DARKNESS) & reg
        valid = (reg & ~spots).astype(np.float32)
    return spots.astype(np.uint8) * 255


def safe_dataset_path(rel):
    full = os.path.abspath(os.path.join(DATASET_DIR, rel))
    if not full.startswith(os.path.abspath(DATASET_DIR) + os.sep):
        raise ValueError("Caminho inválido.")
    if not os.path.isfile(full):
        raise FileNotFoundError("Arquivo não encontrado no dataset.")
    return full


def generate_synthetic_tomato(kind="perfect"):
    img = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 30
    center, radius = (200, 200), 120

    def sphere(rad, bgr_fn):
        for r in range(rad, 0, -2):
            cv2.circle(img, center, r, bgr_fn(int(120 + 135 * (1 - r / rad))), -1)

    def stem(color=(40, 150, 40), y=75):
        cv2.circle(img, (200, y), 10, color, -1)
        cv2.line(img, (200, y), (200, y + 20), color, 4)

    red = lambda v: (20, 20, v)
    if kind == "perfect":
        sphere(radius, red)
        stem()
    elif kind == "green":
        for r in range(radius, 0, -2):
            cv2.circle(img, center, r, (20, int(80 + 120 * (1 - r / radius)), 20), -1)
        stem((20, 90, 20))
    elif kind == "diseased":
        sphere(radius, red)
        cv2.circle(img, (150, 180), 12, (10, 15, 12), -1)
        cv2.circle(img, (240, 220), 16, (8, 12, 10), -1)
        cv2.circle(img, (180, 250), 9, (12, 18, 15), -1)
        stem()
    elif kind == "cracked":
        sphere(radius, red)
        pts = [(130, 130), (160, 160), (180, 170), (210, 160), (250, 140)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (10, 10, 10), 3)
        pts2 = [(220, 220), (240, 250), (260, 280)]
        for a, b in zip(pts2, pts2[1:]):
            cv2.line(img, a, b, (10, 10, 10), 4)
        stem()
    elif kind == "undersized":
        small = 65
        for r in range(small, 0, -2):
            cv2.circle(img, center, r, red(int(120 + 135 * (1 - r / small))), -1)
        stem(y=130)
    else:
        raise ValueError("Tipo sintético desconhecido.")
    return img


def load_image(src):
    """Carrega uma imagem BGR a partir de um descritor enviado pelo frontend."""
    if not src:
        raise ValueError("Nenhuma imagem informada.")
    kind = src.get("kind")
    if kind == "synthetic":
        return generate_synthetic_tomato(src.get("type", "perfect"))
    if kind == "dataset":
        img = cv2.imread(safe_dataset_path(src["file"]))
        if img is None:
            raise ValueError("Não foi possível ler a imagem do dataset.")
        return img
    if kind in ("upload", "camera"):
        return decode_data_url(src["image"])
    raise ValueError("Fonte de imagem inválida.")


# ----------------------------------------------------------------------------
# Segmentação: isola APENAS o tomate (remove o fundo)
# ----------------------------------------------------------------------------
RED1 = ((0, 70, 50), (10, 255, 255))
RED2 = ((170, 70, 50), (180, 255, 255))
GREEN = ((35, 40, 40), (85, 255, 255))
YELLOW = ((11, 50, 50), (34, 255, 255))


def color_masks(hsv):
    rng = lambda r: cv2.inRange(hsv, np.array(r[0]), np.array(r[1]))
    red = cv2.bitwise_or(rng(RED1), rng(RED2))
    return red, rng(GREEN), rng(YELLOW)


def _largest_filled(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(mask)
    c = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [c], -1, 255, -1)
    return out


def segment_tomato_ex(img_bgr):
    """Retorna (máscara 0/255, método usado: grabcut | cor | fallback)."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red, green, yellow = color_masks(hsv)
    color = cv2.bitwise_or(cv2.bitwise_or(red, green), yellow)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, k5)
    color = cv2.morphologyEx(color, cv2.MORPH_OPEN, k5)
    color_fill = _largest_filled(color)
    total = h * w

    mask = None
    method = "grabcut"
    sure = cv2.erode(color_fill, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    if cv2.countNonZero(sure) > 100:
        gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        gc[color > 0] = cv2.GC_PR_FGD
        gc[sure > 0] = cv2.GC_FGD
        m = max(2, int(0.03 * min(h, w)))
        gc[:m, :] = cv2.GC_BGD
        gc[-m:, :] = cv2.GC_BGD
        gc[:, :m] = cv2.GC_BGD
        gc[:, -m:] = cv2.GC_BGD
        try:
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(img_bgr, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
            mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
            mask = _largest_filled(mask)
        except cv2.error:
            mask = None

    # Validação: GrabCut pode falhar (vazio ou invadir a imagem toda)
    if mask is None or not (0.015 * total < cv2.countNonZero(mask) < 0.92 * total):
        mask = color_fill
        method = "cor"
    if not (0.015 * total < cv2.countNonZero(mask) < 0.98 * total):
        # último recurso: elipse central (NÃO é uma segmentação real)
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.3), int(h * 0.3)), 0, 0, 360, 255, -1)
        method = "fallback"
    return mask, method


def segment_tomato(img_bgr):
    return segment_tomato_ex(img_bgr)[0]


def lbp_map(gray):
    if SKIMAGE:
        return local_binary_pattern(gray, 24, 3, method="uniform")
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(sx, sy)


# ----------------------------------------------------------------------------
# Análise de qualidade (somente dentro da máscara do tomate)
# ----------------------------------------------------------------------------
def analyze(img_bgr, min_score=80):
    img, frame_px = prepare_image(img_bgr, return_frame=True)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    mask, seg_method = segment_tomato_ex(img)
    area = max(1, cv2.countNonZero(mask))
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    if cv2.countNonZero(inner) < 200:
        inner = mask

    reasons, warnings = [], []
    if seg_method == "fallback":
        reasons.append("não foi possível isolar o tomate na imagem")

    # Reflexo (brilho estourado) não é cor nem defeito: sai dos cálculos de cor
    highlight = ((hsv[..., 2] >= 235) & (hsv[..., 1] < 70)).astype(np.uint8) * 255
    valid = cv2.bitwise_and(mask, cv2.bitwise_not(highlight))
    valid_inner = cv2.bitwise_and(inner, cv2.bitwise_not(highlight))
    if cv2.countNonZero(valid_inner) < 200:
        valid_inner = inner
    valid_area = max(1, cv2.countNonZero(valid))

    # 1) Cor / maturação (percentuais relativos ao tomate, sem reflexos)
    red, green, yellow = color_masks(hsv)
    pct = lambda m: cv2.countNonZero(cv2.bitwise_and(m, valid)) / valid_area * 100
    pct_red, pct_green, pct_yellow = pct(red), pct(green), pct(yellow)

    if pct_red > 50:
        ripeness_class, ripeness_score = "Vermelho (Maduro)", 20
    elif pct_red > 20:
        ripeness_class, ripeness_score = "Rosado / Vermelho claro", 18
    elif pct_yellow > 30:
        ripeness_class, ripeness_score = "Em transição", 15
    elif pct_green > 40:
        ripeness_class, ripeness_score = "Verde (Imaturo)", 10
    else:
        ripeness_class, ripeness_score = "Indeterminado", 12

    # Nota de cor pelo canal a* do Lab (verde -> vermelho), mais estável com a luz
    # do que contar pixels por faixa de matiz.
    a_mean = cv2.mean(lab, mask=valid_inner)[1] - 128
    redness = max(0.0, min(1.0, a_mean / 40.0))
    color_score = int(round(5 + 20 * redness))

    # 2) Tamanho / geometria
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    # tamanho = fração da foto que o tomate ocupa (equivalente a um quadro 400x400);
    # assim não depende da proporção/resolução da câmera, só da distância fixa.
    size_px = cv2.contourArea(c) * (SIZE * SIZE) / frame_px
    x, y, bw, bh = cv2.boundingRect(c)
    aspect = bw / bh if bh else 1.0
    if SIZE_MIN_OK <= size_px <= SIZE_MAX_OK:
        size_score, size_class = 15, "Padrão"
    elif size_px < SIZE_MIN_OK:
        size_score, size_class = max(5, int(15 * size_px / SIZE_MIN_OK)), "Pequeno"
    else:
        size_score, size_class = 12, "Grande"
    m_edge = 2
    if x <= m_edge or y <= m_edge or x + bw >= SIZE - m_edge or y + bh >= SIZE - m_edge:
        warnings.append("o tomate toca a borda da imagem: tamanho pode estar subestimado")

    # 3) Textura (LBP só dentro do tomate)
    lbp = lbp_map(gray)
    vals = lbp[inner > 0]
    roughness = float(np.std(vals)) if vals.size else 0.0
    lo, hi = (4.5, 7.0) if SKIMAGE else (25.0, 60.0)
    if roughness < lo:
        texture_score, texture_class = 15, "Lisa / Premium"
    elif roughness < hi:
        texture_score, texture_class = 12, "Normal"
    else:
        texture_score, texture_class = 7, "Áspera / Opaca"

    # 4) Saúde e defeitos (manchas escuras em relação ao brilho local)
    spot = local_dark_spots(gray, valid_inner)
    spot = cv2.morphologyEx(spot, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    spot_pct = cv2.countNonZero(spot) / area * 100

    has_crack = False
    biggest_spot_pct = 0.0
    sc, _ = cv2.findContours(spot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for s_ in sc:
        biggest_spot_pct = max(biggest_spot_pct, cv2.contourArea(s_) / area * 100)
        if cv2.contourArea(s_) < 15 and len(s_) < 15:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(s_)
        long_, short_ = max(rw, rh), max(1.0, min(rw, rh))
        if long_ > 35 and long_ / short_ > 4:
            has_crack = True

    if has_crack:
        health_score, health_class = 8, "Rachado / Danificado"
        defect_score, defect_class = 1, "Rachaduras"
    elif spot_pct < SPOT_OK_PCT and biggest_spot_pct < SPOT_BIG_PCT:
        health_score, health_class = 20, "Saudável"
        defect_score, defect_class = 5, "Sem defeitos"
    elif spot_pct < SPOT_REJECT_PCT and biggest_spot_pct < SPOT_BIG_PCT:
        health_score, health_class = 14, "Pequenas manchas"
        defect_score, defect_class = 4, "Leve machucado"
    else:
        health_score, health_class = 6, "Doente / Podre"
        defect_score, defect_class = 1, "Manchas profundas"

    total = color_score + texture_score + size_score + ripeness_score + health_score + defect_score

    # Defeito grave REPROVA mesmo com nota alta nos outros critérios
    # (antes tomate rachado/manchado/pequeno somava >= 80 e ia para a Cesta A).
    if has_crack:
        reasons.append("rachaduras")
    if health_class == "Doente / Podre":
        reasons.append("manchas escuras / sinais de podridão")
    if REJECT_SMALL and size_class == "Pequeno":
        reasons.append("tamanho abaixo do padrão")
    if REJECT_UNRIPE and ripeness_class == "Verde (Imaturo)":
        reasons.append("tomate verde (imaturo)")
    if total < min_score:
        reasons.append(f"nota {int(total)} abaixo do mínimo ({int(min_score)})")
    decision = "B" if reasons else "A"

    # Imagens de apoio (tudo restrito ao tomate)
    cutout = np.full_like(img_rgb, 24)
    cutout[mask > 0] = img_rgb[mask > 0]

    overlay = np.zeros_like(img_rgb)
    overlay[(red > 0) & (mask > 0)] = [230, 50, 50]
    overlay[(green > 0) & (mask > 0)] = [50, 200, 50]
    overlay[(yellow > 0) & (mask > 0)] = [230, 220, 50]
    color_img = cutout.copy()
    keep = mask > 0
    color_img[keep] = cv2.addWeighted(cutout, 0.6, overlay, 0.4, 0)[keep]

    contour_img = img_rgb.copy()
    cv2.drawContours(contour_img, [c], -1, (0, 255, 0), 2)
    cv2.rectangle(contour_img, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
    cv2.putText(contour_img, f"{int(size_px)}px", (x, max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    lbp_disp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lbp_disp[mask == 0] = 0

    defect_img = cutout.copy()
    defect_img[spot > 0] = (0.4 * defect_img[spot > 0] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)

    return {
        "metrics": {
            "color": {"red_pct": pct_red, "green_pct": pct_green, "yellow_pct": pct_yellow, "score": color_score},
            "size": {"area_px": float(size_px), "width": bw, "height": bh, "aspect": aspect,
                     "class": size_class, "score": size_score},
            "texture": {"roughness": roughness, "class": texture_class, "score": texture_score},
            "ripeness": {"class": ripeness_class, "score": ripeness_score},
            "health": {"class": health_class, "spot_pct": spot_pct, "score": health_score},
            "defects": {"class": defect_class, "score": defect_score},
        },
        "total_score": int(total),
        "basket": decision,
        "reasons": reasons,
        "warnings": warnings,
        "segmentation": seg_method,
        "images": {
            "original": to_data_url(img_rgb),
            "tomato_only": to_data_url(cutout),
            "color_split": to_data_url(color_img),
            "contours": to_data_url(contour_img),
            "texture": to_data_url(lbp_disp, rgb=False),
            "defects": to_data_url(defect_img),
        },
    }


# ----------------------------------------------------------------------------
# Comparação entre dois tomates (somente a região do tomate)
# ----------------------------------------------------------------------------
def tomato_signature(img_bgr):
    img = prepare_image(img_bgr)
    mask = segment_tomato(img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = max(1.0, cv2.arcLength(c, True))
    hull_area = max(1.0, cv2.contourArea(cv2.convexHull(c)))
    x, y, w, h = cv2.boundingRect(c)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    if cv2.countNonZero(inner) < 200:
        inner = mask

    hist = cv2.calcHist([hsv], [0, 1], inner, [18, 8], [0, 180, 0, 256])
    hist = hist / max(1e-9, hist.sum())

    lbp = lbp_map(gray)
    if SKIMAGE:
        lh, _ = np.histogram(lbp[inner > 0], bins=26, range=(0, 26))
    else:
        lh, _ = np.histogram(lbp[inner > 0], bins=26)
    lh = (lh / max(1, lh.sum())).astype(np.float32)

    mean_lab = np.array(cv2.mean(lab, mask=inner)[:3])
    spot = local_dark_spots(gray, inner)
    spot_pct = cv2.countNonZero(spot) / max(1, cv2.countNonZero(mask)) * 100

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, mask])[y:y + h, x:x + w]
    return {
        "contour": c, "area": area, "circ": 4 * np.pi * area / perim ** 2,
        "solidity": area / hull_area, "aspect": w / h if h else 1.0,
        "hist": hist.astype(np.float32), "lbp_hist": lh, "lab": mean_lab,
        "spot_pct": spot_pct, "cutout": rgba,
    }


def compare_tomatoes(img_a, img_b):
    a, b = tomato_signature(img_a), tomato_signature(img_b)

    bh = cv2.compareHist(a["hist"], b["hist"], cv2.HISTCMP_BHATTACHARYYA)
    delta_e = float(np.linalg.norm(a["lab"] - b["lab"]))
    color = 0.6 * (1 - bh) + 0.4 * max(0.0, 1 - delta_e / 60.0)

    texture = 1 - cv2.compareHist(a["lbp_hist"], b["lbp_hist"], cv2.HISTCMP_BHATTACHARYYA)

    ms = cv2.matchShapes(a["contour"], b["contour"], cv2.CONTOURS_MATCH_I1, 0)
    geo = (abs(a["circ"] - b["circ"]) + abs(a["solidity"] - b["solidity"])
           + abs(np.log(max(a["aspect"], 1e-3) / max(b["aspect"], 1e-3)))) / 3
    shape = 0.5 * float(np.exp(-5 * ms)) + 0.5 * max(0.0, 1 - geo * 4)

    size = min(a["area"], b["area"]) / max(a["area"], b["area"], 1.0)
    defects = 1 - min(1.0, abs(a["spot_pct"] - b["spot_pct"]) / 5.0)

    parts = {"color": color, "texture": texture, "shape": shape, "size": size, "defects": defects}
    parts = {k: float(max(0.0, min(1.0, v))) for k, v in parts.items()}
    weights = {"color": 0.40, "texture": 0.20, "shape": 0.20, "size": 0.10, "defects": 0.10}
    overall = sum(parts[k] * weights[k] for k in parts) * 100

    if overall >= 85:
        verdict = "Tomates muito semelhantes"
    elif overall >= 70:
        verdict = "Tomates semelhantes"
    elif overall >= 50:
        verdict = "Semelhança parcial"
    else:
        verdict = "Tomates diferentes"

    return {
        "overall": round(overall, 1),
        "verdict": verdict,
        "parts": {k: round(v * 100, 1) for k, v in parts.items()},
        "details": {
            "delta_e": round(delta_e, 1),
            "area_a": int(a["area"]), "area_b": int(b["area"]),
            "spot_a": round(a["spot_pct"], 2), "spot_b": round(b["spot_pct"], 2),
        },
        "cutouts": {"a": to_data_url(a["cutout"]), "b": to_data_url(b["cutout"])},
    }


# ----------------------------------------------------------------------------
# ESP32 (serial)
# ----------------------------------------------------------------------------
class Esp32:
    def __init__(self):
        self.ser = None
        self.lock = threading.Lock()

    def connect(self, port, baud):
        with self.lock:
            self.disconnect_nolock()
            self.ser = serial.Serial(port, baud, timeout=1)

    def disconnect_nolock(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def disconnect(self):
        with self.lock:
            self.disconnect_nolock()

    def send(self, ch):
        with self.lock:
            if self.ser and self.ser.is_open:
                try:
                    self.ser.write(ch.encode())
                    return f"Enviado '{ch}' ao ESP32. Aguardando acionamento do servo..."
                except Exception as e:
                    return f"Falha ao enviar ao ESP32: {e}"
            return f"[SIMULAÇÃO] Comando serial '{ch}' enviado ao ESP32 simulado."

    @property
    def connected(self):
        return bool(self.ser and self.ser.is_open)


esp = Esp32()


# ----------------------------------------------------------------------------
# Rotas
# ----------------------------------------------------------------------------
def err(msg, code=400):
    return jsonify({"error": msg}), code


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/style.css")
def css():
    return send_from_directory(BASE_DIR, "style.css")


@app.route("/script.js")
def js():
    return send_from_directory(BASE_DIR, "script.js")


@app.route("/api/ports")
def ports():
    lst = [p.device for p in serial.tools.list_ports.comports()] if SERIAL else []
    return jsonify({"serial_available": SERIAL, "ports": lst, "connected": esp.connected})


@app.route("/api/connect", methods=["POST"])
def connect():
    if not SERIAL:
        return err("pyserial não está instalado.")
    d = request.get_json(force=True)
    try:
        esp.connect(d["port"], int(d.get("baud", 115200)))
        return jsonify({"connected": True, "message": f"Conectado a {d['port']}"})
    except Exception as e:
        return err(f"Erro ao conectar: {e}")


@app.route("/api/disconnect", methods=["POST"])
def disconnect():
    esp.disconnect()
    return jsonify({"connected": False})


@app.route("/api/dataset")
def dataset_list():
    if not os.path.isdir(DATASET_DIR):
        return jsonify({"exists": False, "path": DATASET_DIR, "files": []})
    files = []
    for root, _, names in os.walk(DATASET_DIR):
        for n in names:
            if n.lower().endswith(IMG_EXT):
                files.append(os.path.relpath(os.path.join(root, n), DATASET_DIR).replace("\\", "/"))
    return jsonify({"exists": True, "path": DATASET_DIR, "files": sorted(files)})


@app.route("/api/dataset/image")
def dataset_image():
    try:
        full = safe_dataset_path(request.args.get("file", ""))
    except Exception as e:
        return err(str(e), 404)
    return send_from_directory(os.path.dirname(full), os.path.basename(full))


@app.route("/api/synthetic/<kind>")
def synthetic(kind):
    try:
        return jsonify({"image": to_data_url(generate_synthetic_tomato(kind), rgb=False)})
    except ValueError as e:
        return err(str(e), 404)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    d = request.get_json(force=True)
    try:
        img = load_image(d.get("source"))
        res = analyze(img, int(d.get("min_score", 80)))
    except Exception as e:
        return err(str(e))
    res["serial_log"] = esp.send(res["basket"]) if d.get("send", True) else ""
    res["esp_connected"] = esp.connected
    return jsonify(res)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    d = request.get_json(force=True)
    try:
        return jsonify(compare_tomatoes(load_image(d.get("a")), load_image(d.get("b"))))
    except Exception as e:
        return err(str(e))


if __name__ == "__main__":
    print(f"Dataset: {DATASET_DIR}")
    print("Abra http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)