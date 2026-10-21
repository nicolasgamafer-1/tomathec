"""
TomaTech - monta o dataset buscando fotos de tomates na internet, por categoria.

Categorias (pastas dentro de dataset/):
    verde   -> imaturo
    maduro  -> vermelho, bom (Cesta A)
    podre   -> estragado, mofo, manchas necróticas
    bixado  -> furos de larvas/brocas, muito podre

Uso:
    pip install -r requirements-dataset.txt
    python baixar_dataset.py                          # 60 fotos por categoria
    python baixar_dataset.py -n 100 --categorias verde maduro
    python baixar_dataset.py --so-livres              # só licença Creative Commons (filtro do Bing)
    python baixar_dataset.py --so-processar           # não baixa; só limpa o que já está em _download_tmp
    python baixar_dataset.py --revisao                # só gera revisao_dataset.html

O que o script faz com cada foto baixada:
    1. descarta arquivos corrompidos e imagens com menos de 200 px de lado menor
    2. descarta duplicadas (inclusive a mesma foto em tamanhos diferentes)
    3. (verde e maduro) manda para dataset_suspeitas/ as fotos cuja cor não bate
       com a categoria, usando a mesma segmentação do app.py
    4. salva em dataset/<categoria>/<categoria>_0001.jpg

IMPORTANTE: busca de imagens traz erros (tomate maduro em pasta "podre" etc.).
Depois de baixar, abra revisao_dataset.html e apague à mão as fotos erradas.
As fotos pertencem aos seus autores: use só para estudo/treino interno e,
para publicar ou vender algo, use --so-livres e confira a licença de cada uma.
"""
import argparse
import importlib.util
import logging
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
DATASET = Path(os.environ.get("TOMATO_DATASET", BASE / "dataset"))
RAW = BASE / "_download_tmp"          # fora do dataset, para não aparecer no painel
SUSPEITAS = BASE / "dataset_suspeitas"
REVISAO = BASE / "revisao_dataset.html"
MIN_LADO = 200
MAX_LADO = 1280

CATEGORIAS = {
    "verde": [
        "tomate verde imaturo",
        "unripe green tomato",
        "green tomato close up fruit",
    ],
    "maduro": [
        "tomate maduro vermelho",
        "ripe red tomato fruit",
        "fresh red tomato close up",
    ],
    "podre": [
        "tomate podre",
        "rotten tomato",
        "tomato fruit rot mold",
        "tomate estragado mofo",
    ],
    "bixado": [
        "tomate bichado",
        "tomato fruit borer damage hole",
        "Tuta absoluta tomato fruit damage",
        "Helicoverpa armigera tomato fruit larva",
    ],
}


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------
def baixar(categoria, consultas, n, so_livres):
    from icrawler.builtin import BingImageCrawler  # import tardio: --so-processar não precisa

    por_busca = math.ceil(n * 1.6 / len(consultas))  # baixa a mais: a limpeza descarta parte
    filtros = {"license": "creativecommons"} if so_livres else None
    for i, consulta in enumerate(consultas):
        pasta = RAW / categoria / f"busca{i}"
        pasta.mkdir(parents=True, exist_ok=True)
        print(f"  [{categoria}] buscando: {consulta!r} (até {por_busca})")
        crawler = BingImageCrawler(
            storage={"root_dir": str(pasta)},
            downloader_threads=4,
            log_level=logging.WARNING,
        )
        crawler.crawl(keyword=consulta, max_num=por_busca,
                      min_size=(MIN_LADO, MIN_LADO), filters=filtros)


# ----------------------------------------------------------------------------
# Limpeza
# ----------------------------------------------------------------------------
def dhash(img):
    """Hash perceptual de 64 bits: fotos iguais em tamanhos diferentes dão o mesmo hash."""
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def assinatura(img):
    """(hash de forma, cor média RGB): duas fotos só são 'iguais' se forma E cor batem."""
    return dhash(img), np.asarray(img.resize((16, 16))).reshape(-1, 3).mean(axis=0)


def parecida(sig, vistos, limite=4, dif_cor=30):
    h, cor = sig
    return any(bin(h ^ vh).count("1") <= limite and np.abs(cor - vc).max() < dif_cor
               for vh, vc in vistos)


def carregar_app():
    """Reaproveita a segmentação e as faixas de cor do app.py."""
    caminho = BASE / "app.py"
    if not caminho.exists():
        return None
    spec = importlib.util.spec_from_file_location("tomatech_app", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def checar_cor(img_pil, categoria, app):
    """Retorna o motivo da suspeita (str) ou None se a cor combina com a categoria."""
    if categoria not in ("verde", "maduro"):
        return None  # podre/bixado não têm cor característica confiável
    bgr = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    prep = app.prepare_image(bgr) if hasattr(app, "prepare_image") else cv2.resize(bgr, (app.SIZE, app.SIZE))
    mask, metodo = app.segment_tomato_ex(prep)
    if metodo == "fallback":
        return "tomate não localizado na foto"
    hsv = cv2.cvtColor(prep, cv2.COLOR_BGR2HSV)
    red, green, yellow = app.color_masks(hsv)
    area = max(1, cv2.countNonZero(mask))
    pct = lambda m: cv2.countNonZero(cv2.bitwise_and(m, mask)) / area * 100
    r, g, y = pct(red), pct(green), pct(yellow)
    if categoria == "maduro" and r < 45:
        return f"pouco vermelho ({r:.0f}%)"
    if categoria == "verde" and (g + y < 40 or r > 25):
        return f"cor não é de tomate verde (verde+amarelo {g + y:.0f}%, vermelho {r:.0f}%)"
    return None


def proximo_indice(pasta, prefixo):
    nums = []
    for f in pasta.glob(f"{prefixo}_*.jpg"):
        try:
            nums.append(int(f.stem.split("_")[-1].lstrip("s")))
        except ValueError:
            pass
    return max(nums, default=0) + 1


def processar(categoria, n, filtro_cor, app):
    destino = DATASET / categoria
    destino.mkdir(parents=True, exist_ok=True)

    vistos = []
    for f in list(destino.glob("*.jpg")) + list((SUSPEITAS / categoria).glob("*.jpg")):
        try:
            with Image.open(f) as im:
                vistos.append(assinatura(im.convert("RGB")))
        except Exception:
            pass

    prox = proximo_indice(destino, categoria)
    prox_s = proximo_indice(SUSPEITAS / categoria, categoria)
    aceitas = len(list(destino.glob("*.jpg")))
    stats = Counter()

    bruto = RAW / categoria
    arquivos = sorted(p for p in bruto.rglob("*") if p.is_file()) if bruto.exists() else []
    for f in arquivos:
        if aceitas >= n:
            break
        try:
            with Image.open(f) as im:
                im.load()
                img = im.convert("RGB")
        except Exception:
            stats["corrompida"] += 1
            continue
        if min(img.size) < MIN_LADO:
            stats["pequena"] += 1
            continue
        sig = assinatura(img)
        if parecida(sig, vistos):
            stats["duplicada"] += 1
            continue
        vistos.append(sig)
        if max(img.size) > MAX_LADO:
            img.thumbnail((MAX_LADO, MAX_LADO), Image.LANCZOS)

        motivo = checar_cor(img, categoria, app) if (filtro_cor and app) else None
        if motivo:
            pasta_s = SUSPEITAS / categoria
            pasta_s.mkdir(parents=True, exist_ok=True)
            img.save(pasta_s / f"{categoria}_s{prox_s:04d}.jpg", quality=92)
            prox_s += 1
            stats["suspeita"] += 1
        else:
            img.save(destino / f"{categoria}_{prox:04d}.jpg", quality=92)
            prox += 1
            aceitas += 1
            stats["aceita"] += 1
    return stats, aceitas


# ----------------------------------------------------------------------------
# Página de revisão
# ----------------------------------------------------------------------------
def gerar_revisao():
    def secao(titulo, pasta):
        fotos = sorted(pasta.glob("*.jpg"))
        if not fotos:
            return ""
        itens = "".join(
            f'<figure><img loading="lazy" src="{quote(os.path.relpath(f, BASE).replace(os.sep, "/"))}">'
            f"<figcaption>{f.name}</figcaption></figure>"
            for f in fotos
        )
        return f"<h2>{titulo} ({len(fotos)})</h2><div class='grid'>{itens}</div>"

    corpo = ""
    for cat in CATEGORIAS:
        corpo += secao(f"dataset/{cat}", DATASET / cat)
        corpo += secao(f"dataset_suspeitas/{cat} (cor não bateu: confira antes de usar)", SUSPEITAS / cat)
    REVISAO.write_text(f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Revisão do dataset TomaTech</title><style>
body{{font:15px system-ui,sans-serif;margin:24px;background:#fffae9;color:#23221e}}
h2{{margin:28px 0 10px;color:#a43032}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}}
figure{{margin:0;background:#fff;border:1px solid #d2e5c5;border-radius:8px;overflow:hidden}}
img{{display:block;width:100%;height:150px;object-fit:cover}} figcaption{{padding:4px 8px;font-size:12px;color:#6b655b}}
</style></head><body><h1>Revisão do dataset</h1>
<p>Apague (no Explorador de Arquivos) as fotos que estiverem na categoria errada. O nome do arquivo aparece embaixo de cada foto.</p>
{corpo or "<p>Nenhuma foto ainda.</p>"}</body></html>""", encoding="utf-8")
    print(f"Página de revisão: {REVISAO}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Baixa e organiza fotos de tomates por categoria.")
    ap.add_argument("-n", "--por-categoria", type=int, default=60, help="fotos aceitas por categoria (padrão 60)")
    ap.add_argument("--categorias", nargs="+", choices=list(CATEGORIAS), default=list(CATEGORIAS))
    ap.add_argument("--so-livres", action="store_true", help="só imagens Creative Commons (filtro do Bing)")
    ap.add_argument("--sem-filtro-cor", action="store_true", help="não separa fotos suspeitas de verde/maduro")
    ap.add_argument("--so-processar", action="store_true", help="não baixa, só limpa o que já está em _download_tmp")
    ap.add_argument("--manter-bruto", action="store_true", help="não apaga _download_tmp no fim")
    ap.add_argument("--revisao", action="store_true", help="só gera revisao_dataset.html")
    a = ap.parse_args()

    if a.revisao:
        return gerar_revisao()

    app = None if a.sem_filtro_cor else carregar_app()
    if app is None and not a.sem_filtro_cor:
        print("Aviso: app.py não encontrado ao lado do script; filtro de cor desligado.")

    for cat in a.categorias:
        print(f"\n== {cat} ==")
        if not a.so_processar:
            baixar(cat, CATEGORIAS[cat], a.por_categoria, a.so_livres)
        stats, total = processar(cat, a.por_categoria, not a.sem_filtro_cor, app)
        print(f"  novas: {stats['aceita']} | suspeitas: {stats['suspeita']} | duplicadas: {stats['duplicada']} | "
              f"pequenas: {stats['pequena']} | corrompidas: {stats['corrompida']} | total em dataset/{cat}: {total}")

    if not (a.manter_bruto or a.so_processar) and RAW.exists():
        shutil.rmtree(RAW, ignore_errors=True)
    gerar_revisao()
    print("\nPronto. Abra revisao_dataset.html, apague as fotos erradas e rode: python app.py")


if __name__ == "__main__":
    main()
