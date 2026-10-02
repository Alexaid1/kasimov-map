#!/usr/bin/env python3
"""
Три карты границ исследования: г. Касимов (Рязанская область).

  1. map1_okrug.png        — Уровень 1: округ (административная граница)
  2. map2_historic_settlement.png — Уровень 2: историческое поселение (координаты Приказа № 2969, прил. 1)
  3. slide20_core.png — Уровень 3: граница проекта между оврагами (Никольский / Успенский и Ямской),
     овраги — из data/ovragi.gpkg (создаётся из схемы data/Ovrag.png + OSM)
  + out/boundaries.gpkg    — все контуры и объекты для QGIS

Запуск:   python make_maps.py            (первый раз скачает OSM и сложит в data/)
          python make_maps.py --okn my.csv   (свой слой точек: колонки lat,lon[,name,category])
Данные:   data/prikaz_2969_points.csv, data/prikaz_2969_objects.csv — переписаны из Приказа;
          выгрузка ЕГРОКН data-69-structure-16.csv (opendata.mkrf.ru) — в корне проекта или в data/;
          data/objects_geocoded.csv — объекты с координатами, создаётся при первом запуске, можно править.
Требует:  geopandas osmnx scipy matplotlib pyproj
"""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pyproj

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, Point, Polygon, box
from shapely.affinity import translate
from shapely.ops import linemerge, nearest_points, polylabel, substring, unary_union

warnings.filterwarnings("ignore")

# ───────────────────────── НАСТРОЙКИ ─────────────────────────
CRS_M = "EPSG:32637"          # UTM 37N: метры, Касимов (41.4° в.д.)
OUT = Path("out")
DATA = Path("data")
# Вёрстка под Figma: слайд 1920×1080; размеры карт — в пикселях макета (1×)
PPI = 133                      # px макета на дюйм фигуры: 1 pt шрифта ≈ 1,85 px на слайде
EXPORT_SCALE = 2               # PNG в 2× для чёткости в Figma
SLIDE_RIGHT = (1120, 1080)     # слайды 18–20: правая часть, x = 800…1920, на всю высоту
SLIDE_LEFT = (1120, 900)       # слайды 34–35: левая часть под заголовком, x = 0…1120, y = 180…1080

# Историческое поселение (Приказ Минкультуры № 2969 от 07.12.2015)
ORDER_POINTS = DATA / "prikaz_2969_points.csv"     # прил. 1: координаты 40 характерных точек (МСК-62)
ORDER_OBJECTS = DATA / "prikaz_2969_objects.csv"   # прил. 2: исторически ценные градоформирующие объекты
EGROKN_FULL = [Path("data-69-structure-16.csv"), DATA / "data-69-structure-16.csv"]  # выгрузка opendata.mkrf.ru
EGROKN_CITY = DATA / "egrokn_kasimov.csv"          # выборка ЕГРОКН по городу (создаётся автоматически)
OBJ_CACHE = DATA / "objects_geocoded.csv"          # объекты с координатами (создаётся, можно править)
# МСК-62 зона 2 — приближение; точная привязка — подобием по перекрёсткам ORDER_CONTROL
MSK62_GUESS = ("+proj=tmerc +lat_0=0 +lon_0=41.48333333333 +k=1 +x_0=2250000 +y_0=-5714743.504 "
               "+ellps=krass +towgs84=23.57,-140.95,-79.8,0,0.35,0.79,-0.22 +units=m +no_defs")
# точки Приказа, лежащие на перекрёстках осей улиц (из текстового описания границы)
ORDER_CONTROL = {2: ("Начальная", "Комсомольская"), 3: ("Комсомольская", "Пролетарская"),
                 4: ("Пролетарская", "50 лет ВЛКСМ"), 5: ("50 лет ВЛКСМ", "Володарского"),
                 11: ("Гагарина", "Горького"), 13: ("Ленина", "Горького"),
                 19: ("Фёдоровой", "Чижова"), 20: ("Чижова", "Большакова")}
# улицы, по которым идёт граница (для подписей на карте 2)
STREETS = [
    "Начальная", "Комсомольская", "Пролетарская", "50 лет ВЛКСМ", "Володарского", "Урицкого",
    "Игашова", "Полевой", "Гагарина", "Горького", "Ленина", "Фёдоровой", "Чижова", "Большакова",
    "Старопосадская",
]

# Граница проекта (Уровень 3): между оврагами. Запад — Никольский овраг, восток — Успенский и Ямской,
# юго-запад — Ока, север — улицы между вершинами оврагов. Овраги остаются снаружи (граница по ближней бровке).
OVRAG_PNG = DATA / "Ovrag.png"     # схема «Касимовские овраги»; привязывается по границе исторического поселения
OVRAGI = DATA / "ovragi.gpkg"      # овраги с координатами (создаётся при первом запуске, можно править в QGIS)
# чёрные точки выносок на схеме, px: по ним выбираются контуры оврагов
OVRAG_DOTS = {"Никольский овраг": (201.25, 260.75), "Ямской овраг": (254.5, 261.25), "Успенский овраг": (241.75, 285)}
# начальное приближение привязки: № точки Приказа → px на схеме (уточняется по всему контуру)
OVRAG_GCP = {1: (125, 242), 2: (162.5, 207.5), 4: (211.75, 191.25), 7: (240, 191),
             10: (301, 185), 11: (313, 200), 12: (265, 245)}
OSM_RAVINE = "Успенский овраг"     # в OSM так подписана вся система: верхняя часть — Ямской овраг
NORTH_START = "Академика"          # северная граница начинается на этой улице у вершины Никольского оврага
NORTH_VIA = [("Академика В.Ф.Уткина", "Карла Маркса"), ("Карла Либкнехта", "Комарова"), ("Карла Либкнехта", "Советская")]  # и идёт через перекрёстки
FALLBACK_CENTRE = (54.9356, 41.3911)   # Соборная пл. (lat, lon), если не найдена в OSM

# Палитра
C_BG = "#F6F4EE"; C_WATER = "#BCD2E0"; C_ROAD = "#CFCBC1"; C_TEXT = "#26262B"
C_L1 = "#1F3A5F"; C_L2 = "#B5532E"; C_L3 = "#7A1F2B"; C_OKN = "#3B2F2F"
plt.rcParams.update({"font.family": ["Gilroy", "DejaVu Sans"], "figure.facecolor": "white",
                     "svg.fonttype": "none"})


# ───────────────────────── ДАННЫЕ ─────────────────────────
def norm(s):
    s = str(s).lower().replace("ё", "е")
    s = re.sub(r"\b(улица|ул\.?|переулок|пер\.?|проспект|пр-т|площадь|пл\.?|набережная)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def cached(name, fn):
    DATA.mkdir(exist_ok=True)
    p = DATA / f"{name}.gpkg"
    if p.exists():
        return gpd.read_file(p)
    gdf = fn()
    if len(gdf):
        gdf = gdf.reset_index(drop=True)
        for c in gdf.columns:  # gpkg не любит списки
            if gdf[c].map(lambda v: isinstance(v, (list, dict))).any():
                gdf[c] = gdf[c].astype(str)
        gdf.to_file(p, driver="GPKG")
    return gdf


def geocode_first(queries):
    import osmnx as ox
    for q in queries:
        try:
            g = ox.geocode_to_gdf(q)
            if len(g) and g.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon"):
                print(f"  геокод: {q}")
                return g.to_crs(CRS_M)
        except Exception as e:
            print(f"  не нашёл «{q}»: {e}")
    sys.exit("Не удалось получить границу по запросам: " + "; ".join(queries))


def load_all():
    import osmnx as ox
    ox.settings.use_cache = True
    print("Границы…")
    okrug = cached("okrug", lambda: geocode_first([
        "Касимовский муниципальный округ, Рязанская область, Россия",
        "Касимовский район, Рязанская область, Россия"]))
    city = cached("city", lambda: geocode_first(["Касимов, Рязанская область, Россия"]))
    oblast = cached("oblast", lambda: geocode_first(["Рязанская область, Россия"]))

    okrug_ll = okrug.to_crs(4326).geometry.iloc[0]
    city_buf = city.geometry.iloc[0].buffer(2500)
    city_ll = gpd.GeoSeries([city_buf], crs=CRS_M).to_crs(4326).iloc[0]

    print("Реки (округ)…")
    rivers = cached("rivers", lambda: ox.features_from_polygon(
        okrug_ll, {"waterway": ["river"], "natural": ["water"], "water": ["river", "lake"]}).to_crs(CRS_M))
    print("Вода (город)…")
    water = cached("water_city", lambda: ox.features_from_polygon(
        city_ll, {"natural": ["water"], "waterway": ["river", "stream", "canal"], "water": True}).to_crs(CRS_M))
    print("Улицы (город)…")

    def get_edges():
        G = ox.graph_from_polygon(city_ll, network_type="all", simplify=True)
        return ox.graph_to_gdfs(G, nodes=False).to_crs(CRS_M).reset_index()
    edges = cached("edges_city", get_edges)

    print("Ор-объекты, овраги…")
    heritage = cached("heritage", lambda: ox.features_from_polygon(
        city_ll, {"heritage": True, "historic": True, "tourism": ["museum"]}).to_crs(CRS_M))
    named = cached("named_city", lambda: ox.features_from_polygon(
        city_ll, {"name": True}).to_crs(CRS_M))
    ravines = cached("ravines", lambda: ox.features_from_polygon(
        city_ll, {"natural": ["valley", "ravine", "gully"], "waterway": ["ditch", "stream"]}).to_crs(CRS_M))
    print("Здания (город)…")

    def get_buildings():
        b = ox.features_from_polygon(city_ll, {"building": True}).to_crs(CRS_M)
        b = b[b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        cols = [c for c in ("building", "name", "heritage", "historic", "addr:street", "addr:housenumber",
                            "building:levels") if c in b]
        return b[cols + ["geometry"]]
    buildings = cached("buildings_city", get_buildings)
    return dict(okrug=okrug, city=city, oblast=oblast, rivers=rivers, water=water,
                edges=edges, heritage=heritage, named=named, ravines=ravines, buildings=buildings)


# ───────────────────────── ГРАНИЦА ПО ПРИКАЗУ (УРОВЕНЬ 2) ─────────────────────────
def edge_names(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        return re.findall(r"'([^']+)'", v)
    return [v] if isinstance(v, str) else []


def river_geoms(water):
    """Акватория Оки (полигоны, через которые идёт линия «Ока») и линия Бабёнки."""
    wname = water["name"].astype(str) if "name" in water else pd.Series([""] * len(water))
    oka_line = unary_union(water[wname.str.fullmatch("Ока")].geometry)
    wpoly = water[water.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    oka = unary_union(wpoly[wpoly.intersects(oka_line)].geometry) if not oka_line.is_empty else None
    if oka is None or oka.is_empty:
        sys.exit("Не найдена акватория Оки в data/water_city.gpkg")
    oka_parts = list(oka.geoms) if hasattr(oka, "geoms") else [oka]
    bab = linemerge(unary_union(water[wname.str.contains("Бабенка|Бабёнка")].geometry))
    if hasattr(bab, "geoms"):
        bab = max(bab.geoms, key=lambda l: l.length)
    return oka_parts, (None if bab.is_empty else bab)


def shore_arc(oka_parts, a, b):
    """Более короткая дуга берега Оки между проекциями точек a и b, ориентированная от a к b."""
    part = min(oka_parts, key=lambda g: g.distance(a) + g.distance(b))
    sl = part.exterior
    da, db = sl.project(a), sl.project(b)
    lo, hi = min(da, db), max(da, db)
    inner = list(substring(sl, lo, hi).coords)
    outer = list(substring(sl, hi, sl.length).coords) + list(substring(sl, 0, lo).coords)[1:]
    arc = inner if LineString(inner).length <= LineString(outer).length else outer[::-1]
    return arc if Point(arc[0]).distance(a) <= Point(arc[-1]).distance(a) else arc[::-1]


def street_crossing(edges, a, b, near=None):
    """Точка пересечения двух улиц по OSM (общий узел или середина ближайших точек).
    near: если улицы пересекаются несколько раз — берётся пересечение, ближайшее к этой точке."""
    def sel(key):
        k = norm(key)
        return edges[edges["name"].map(lambda v: any(k == norm(n) for n in edge_names(v)))]
    ea, eb = sel(a), sel(b)
    if not len(ea) or not len(eb):
        return None
    ga, gb = unary_union(ea.geometry), unary_union(eb.geometry)
    x = ga.intersection(gb)
    if not x.is_empty:
        if near is not None and hasattr(x, "geoms"):
            x = min(x.geoms, key=lambda g: g.distance(near))
        return np.array([x.centroid.x, x.centroid.y])
    p, q = nearest_points(ga, gb)
    return np.array([(p.x + q.x) / 2, (p.y + q.y) / 2]) if p.distance(q) < 40 else None


def fit_similarity(A, B):
    """Подобие B ≈ s·R·A + t (Гельмерт 2D) по МНК."""
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ U.T
    s = S.sum() / ((A - ca) ** 2).sum()
    return s, R, cb - s * R @ ca


def order_boundary(d):
    """Контур исторического поселения по координатам характерных точек Приказа № 2969 (МСК-62).

    Параметры МСК-62 в открытых источниках известны неточно, поэтому точки сначала
    проецируются приближённой ТМ (Красовский, осевой 41°29′), а затем привязываются к OSM
    подобием по перекрёсткам, которые названы в текстовом описании границы.
    """
    pts = pd.read_csv(ORDER_POINTS)
    tr = pyproj.Transformer.from_crs(pyproj.CRS.from_proj4(MSK62_GUESS), CRS_M, always_xy=True)
    P = {int(r.n): np.array(tr.transform(r.y, r.x)) for r in pts.itertuples()}  # X — север, Y — восток

    A, B, ids = [], [], []
    for n, (a, b) in ORDER_CONTROL.items():
        q = street_crossing(d["edges"], a, b)
        if q is not None:
            A.append(P[n]); B.append(q); ids.append(n)
    A, B = np.array(A), np.array(B)
    s, R, t = fit_similarity(A, B)
    res = np.hypot(*(B - (s * (R @ A.T).T + t)).T)
    ok = res < 25                                        # точка не на оси перекрёстка — отбрасываем
    s, R, t = fit_similarity(A[ok], B[ok])
    res = np.hypot(*(B - (s * (R @ A.T).T + t)).T)
    print(f"  привязка МСК-62 → OSM по {ok.sum()} перекрёсткам: "
          + ", ".join(f"т.{n} {r:.0f} м" for n, r in zip(ids, res)))
    Q = {n: s * R @ p + t for n, p in P.items()}

    oka_parts, bab = river_geoms(d["water"])
    coords = []
    for n in range(1, len(Q) + 1):
        coords.append(tuple(Q[n]))
        if n == 25 and bab is not None:                 # 25–26: по левому берегу Бабёнки
            a, b = Point(Q[25]), Point(Q[26])
            if a.distance(bab) < 80 and b.distance(bab) < 80:
                da, db = bab.project(a), bab.project(b)
                seg = list(substring(bab, min(da, db), max(da, db)).coords)
                coords += seg if da <= db else seg[::-1]
    # 40–1: на север по береговой линии Оки
    coords += shore_arc(oka_parts, Point(Q[len(Q)]), Point(Q[1]))
    poly = Polygon(coords).buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    print(f"  площадь: {poly.area/1e4:.1f} га (в Приказе — 357,5 га)")
    points = gpd.GeoDataFrame({"n": list(Q)}, geometry=[Point(q) for q in Q.values()], crs=CRS_M)
    return poly, points


# ───────────────────────── ОБЪЕКТЫ: ПРИКАЗ (ПРИЛ. 2) + ЕГРОКН ─────────────────────────
def street_key(s):
    """(тип, порядковый №, опорное слово) — «ул. Академика В.Ф.Уткина» ~ «улица Академика Уткина»."""
    s = str(s).lower().replace("ё", "е")
    typ = ("пер" if re.search(r"переулок|\bпер\.", s) else "пл" if re.search(r"площад|\bпл\.", s)
           else "пр" if "проезд" in s else "ул")
    m = re.search(r"\b(\d+)-[йя]\b", s)
    s = re.sub(r"\b\d+-[йя]\b|улица|переулок|площадь|проезд|\bул\.|\bпер\.|\bпл\.", " ", s)
    words = re.findall(r"[а-яa-z0-9]+", s)
    return typ, (m.group(1) if m else ""), (words[-1] if words else "")


def same_street(a, b):
    return a[0] == b[0] and a[2] == b[2] and (a[1] == b[1] or {a[1], b[1]} <= {"", "1"})


def house_norm(h):
    h = str(h).lower().replace("ё", "е").translate(str.maketrans("abvgde", "абвгде"))  # латиница в литерах
    h = re.sub(r"\b(дом|д\.|здание|стр\.)", "", h)
    return re.sub(r"\s+", "", h).strip(",.")


def house_num(h):
    m = re.match(r"(\d+)", house_norm(h))
    return int(m.group(1)) if m else None


def egrokn_city():
    """Объекты ЕГРОКН в г. Касимове: из выгрузки opendata.mkrf.ru (кэшируется в data/)."""
    if EGROKN_CITY.exists():
        return pd.read_csv(EGROKN_CITY, dtype=str)
    src = next((p for p in EGROKN_FULL if p.exists()), None)
    if src is None:
        print("  ⚠ выгрузка ЕГРОКН не найдена — используется только перечень Приказа")
        return pd.DataFrame(columns=["name", "street", "house", "category", "lat", "lon"])
    print(f"  разбор {src} …")
    cols = ["Объект", "Полный адрес", "На карте", "Категория историко-культурного значения"]
    parts = []
    for ch in pd.read_csv(src, usecols=cols, dtype=str, chunksize=50000):
        parts.append(ch[ch["Полный адрес"].str.contains(r"г\.?\s*Касимов|город Касимов", na=False, regex=True)])
    df = pd.concat(parts)
    rows = []
    for r in df.itertuples(index=False):
        addr = re.split(r"(?:г\.|город)\s*Касимов,?", r[1])[-1]
        bits = [b.strip() for b in addr.split(",") if b.strip()]
        lat = lon = None
        if isinstance(r[2], str):
            lon, lat = json.loads(r[2])["coordinates"][:2]
        rows.append(dict(name=r[0], street=bits[0] if bits else "", house=house_norm(bits[1]) if len(bits) > 1 else "",
                         category=r[3], lat=lat, lon=lon))
    out = pd.DataFrame(rows)
    out.to_csv(EGROKN_CITY, index=False, encoding="utf-8-sig")
    return pd.read_csv(EGROKN_CITY, dtype=str)


def nominatim(street, house):
    """Последний шаг привязки: Nominatim, только если найден именно дом."""
    import time, urllib.parse, urllib.request
    q = urllib.parse.urlencode({"q": f"{street}, {house}, Касимов, Рязанская область", "format": "json",
                                "addressdetails": 1, "limit": 1})
    req = urllib.request.Request(f"https://nominatim.openstreetmap.org/search?{q}",
                                 headers={"User-Agent": "kasimov-heritage-maps/1.0 (study project)"})
    time.sleep(1.1)
    try:
        js = json.loads(urllib.request.urlopen(req, timeout=20).read())
    except Exception:
        return None
    if js and js[0].get("address", {}).get("house_number"):
        return float(js[0]["lat"]), float(js[0]["lon"])
    return None


MKD_XLSX = DATA / "floor.xlsx"               # реестр МКД с dom.mingkh.ru: Адрес, Год, Этажей
MKD_CACHE = DATA / "mkd_geocoded.csv"        # МКД с привязкой к зданиям OSM (создаётся, можно править)
TALL_LEVELS = 4                              # «многоэтажная» для города с 1–2-этажной исторической застройкой


def mkd_buildings(d):
    """Здания OSM с этажностью: OSM building:levels + реестр МКД (mingkh), привязанный по адресу.
    Возвращает GeoDataFrame зданий с колонками levels, year, src."""
    b = d["buildings"].copy()
    b["levels"] = pd.to_numeric(b.get("building:levels"), errors="coerce")
    b["year"] = np.nan
    b["src"] = np.where(b["levels"].notna(), "OSM", None)
    if not MKD_XLSX.exists():
        return b
    if MKD_CACHE.exists():
        m = pd.read_csv(MKD_CACHE)
    else:
        print("  привязка реестра МКД к зданиям (один раз → data/mkd_geocoded.csv)…")
        x = pd.read_excel(MKD_XLSX)
        x = x.rename(columns={c: str(c).strip() for c in x.columns})
        to_ll = pyproj.Transformer.from_crs(CRS_M, 4326, always_xy=True)
        ba = b[b["addr:street"].notna() & b["addr:housenumber"].notna()]
        idx = {}
        for i, s, h in zip(ba.index, ba["addr:street"], ba["addr:housenumber"]):
            for hh in {house_norm(h)} | set(house_norm(h).split("/")):
                idx.setdefault((street_key(s)[0], street_key(s)[2], hh), []).append(i)
        rows = []
        for r in x.itertuples(index=False):
            addr = str(r[2]); street, _, house = addr.rpartition(",")
            sk, hn = street_key(street), house_norm(house)
            levels = pd.to_numeric(r[5], errors="coerce"); year = pd.to_numeric(r[4], errors="coerce")
            hit = idx.get((sk[0], sk[2], hn)) or idx.get(next(((t, w, h) for (t, w, h) in idx
                                                               if w == sk[2] and h == hn), None), [])
            lat = lon = None; how = "не найден"
            if hit:
                c = b.loc[hit[0]].geometry.centroid
                lon, lat = to_ll.transform(c.x, c.y); how = "OSM: адрес"
            elif levels >= TALL_LEVELS:          # многоэтажки ищем и геокодером — их важно не потерять
                ll = nominatim(street.strip(), house.strip())
                if ll:
                    lat, lon = ll; how = "Nominatim"
                else:                             # ближайший номер той же чётности на этой улице (±4)
                    n = house_num(house)
                    near = [(abs(house_num(h) - n), i) for (t, w, h), ii in idx.items() for i in ii
                            if n is not None and w == sk[2] and house_num(h) is not None
                            and house_num(h) % 2 == n % 2 and abs(house_num(h) - n) <= 4]
                    if near:
                        dn, i = min(near)
                        c = b.loc[i].geometry.centroid
                        lon, lat = to_ll.transform(c.x, c.y); how = f"соседний дом (±{dn})"
            rows.append(dict(address=addr, levels=levels, year=year, lat=lat, lon=lon, method=how))
        m = pd.DataFrame(rows)
        m.to_csv(MKD_CACHE, index=False, encoding="utf-8-sig")
        print("  МКД:", m["method"].value_counts().to_dict(),
              f"| {TALL_LEVELS}+ эт.: {(m['levels'] >= TALL_LEVELS).sum()}, из них не найдено:",
              ((m["levels"] >= TALL_LEVELS) & m["lat"].isna()).sum())
    m = m[m["lat"].notna()]
    pts = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m["lon"], m["lat"]), crs=4326).to_crs(CRS_M)
    # точка → здание, в которое она попадает, иначе ближайшее в пределах 25 м
    j = gpd.sjoin_nearest(pts, b[["geometry"]], how="left", max_distance=25, distance_col="dist")
    j = j[j["index_right"].notna()]
    for _, r in j.iterrows():
        i = r["index_right"]
        b.loc[i, "levels"] = r["levels"]; b.loc[i, "year"] = r["year"]; b.loc[i, "src"] = "МКД (mingkh)"
    return b


def geocode_objects(d):
    """Перечень Приказа (186 ценных объектов) + ЕГРОКН → точки с категорией и способом привязки.

    Результат кэшируется в data/objects_geocoded.csv — его можно поправить вручную
    (колонки lat, lon) и перезапустить скрипт.
    """
    if OBJ_CACHE.exists():
        df = pd.read_csv(OBJ_CACHE, dtype={"n": str})
        df = df[df["lat"].notna()]
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326).to_crs(CRS_M)

    print("  привязка объектов к зданиям (один раз, результат → data/objects_geocoded.csv)…")
    order = pd.read_csv(ORDER_OBJECTS, dtype=str).fillna("")
    egr = egrokn_city()
    to_ll = pyproj.Transformer.from_crs(CRS_M, 4326, always_xy=True)

    # индекс адресов OSM: (ключ улицы, дом) → центроид здания
    b = d["buildings"]
    b = b[b["addr:street"].notna() & b["addr:housenumber"].notna()]
    osm = [(street_key(s), house_norm(h), g.centroid) for s, h, g in zip(b["addr:street"], b["addr:housenumber"], b.geometry)]

    def by_osm(skey, house):
        hs = {house_norm(house)} | set(house_norm(house).split("/"))
        hit = [c for k, h, c in osm if same_street(k, skey) and h in hs]
        if hit:
            return hit[0], "OSM: здание по адресу"
        # тип улицы в OSM бывает другим («переулок Губарева» вместо «улица») — сверяем только название
        hit = [c for k, h, c in osm if k[2] == skey[2] and h in hs]
        return (hit[0], "OSM: здание по адресу (тип улицы в OSM другой)") if len(hit) == 1 else (None, None)

    def by_interp(skey, house):
        n = house_num(house)
        if n is None:
            return None, None
        side = [(house_num(h), c) for k, h, c in osm if same_street(k, skey) and house_num(h) is not None
                and house_num(h) % 2 == n % 2]
        lo = max((x for x in side if x[0] < n), default=None, key=lambda x: x[0])
        hi = min((x for x in side if x[0] > n), default=None, key=lambda x: x[0])
        if lo and hi and hi[0] - lo[0] <= 12:
            t = (n - lo[0]) / (hi[0] - lo[0])
            return Point(lo[1].x + t * (hi[1].x - lo[1].x), lo[1].y + t * (hi[1].y - lo[1].y)), \
                f"интерполяция между д. {lo[0]} и {hi[0]}"
        near = min([x for x in (lo, hi) if x], default=None, key=lambda x: abs(x[0] - n))
        if near and abs(near[0] - n) <= 4:
            return near[1], f"соседний дом {near[0]}"
        return None, None

    def by_linear(skey, house):
        """Номер → положение вдоль оси улицы по линейной модели известных домов (+ сторона по чётности)."""
        n = house_num(house)
        if n is None:
            return None, None
        es = d["edges"][d["edges"]["name"].map(lambda v: any(same_street(street_key(x), skey) for x in edge_names(v)))]
        if not len(es):
            return None, None
        line = linemerge(unary_union(es.geometry))
        if hasattr(line, "geoms"):
            line = max(line.geoms, key=lambda l: l.length)
        known = [(house_num(h), c) for k, h, c in osm if same_street(k, skey) and house_num(h) is not None]
        if len(known) < 3:
            return None, None
        nums = np.array([k[0] for k in known]); pos = np.array([line.project(k[1]) for k in known])
        if np.corrcoef(nums, pos)[0, 1] ** 2 < 0.8:
            return None, None
        a, b0 = np.polyfit(nums, pos, 1)
        dist = float(np.clip(a * n + b0, 0, line.length))
        p = line.interpolate(dist); q = line.interpolate(min(dist + 5, line.length))
        tx, ty = q.x - p.x, q.y - p.y; tn = max(np.hypot(tx, ty), 1e-6)
        side = [(c.x - line.interpolate(line.project(c)).x) * -ty / tn + (c.y - line.interpolate(line.project(c)).y) * tx / tn
                for k, c in known if k % 2 == n % 2]
        off = float(np.mean(side)) if side else 0.0
        return Point(p.x - ty / tn * off, p.y + tx / tn * off), "линейная модель нумерации улицы"

    named = d["named"]
    landmark = re.compile(r"церк|мечет|текие|собор|монастыр|училищ|гимнази|застав|колокольн", re.I)

    def by_name(name):
        """Культовые и общественные здания — по названию в OSM («Ильинская церковь» и т. п.)."""
        if not landmark.search(name):
            return None, None
        stem = next((w[:6] for w in re.findall(r"[А-ЯЁ][а-яё]{4,}", name) if not landmark.search(w)), None)
        if stem is None:
            return None, None
        m = named[named["name"].astype(str).str.contains(stem, case=False, na=False)
                  & named["name"].astype(str).str.contains(landmark, na=False)]
        return (m.geometry.iloc[0].centroid, f"OSM: по названию «{m['name'].iloc[0]}»") if len(m) else (None, None)

    egr_keys = [(street_key(s), house_norm(h)) for s, h in zip(egr["street"], egr["house"])]
    used = set()
    rows = []

    def locate(street, house, egr_i=None, name=""):
        if egr_i is not None:
            lat, lon = pd.to_numeric(egr.iloc[egr_i][["lat", "lon"]], errors="coerce")
            if pd.notna(lat) and pd.notna(lon):
                return float(lat), float(lon), "ЕГРОКН: координаты реестра"
        sk = street_key(street)
        steps = [lambda: by_osm(sk, house), lambda: by_name(name), lambda: by_interp(sk, house)]
        for f in steps:
            p, how = f()
            if p is not None:
                lon, lat = to_ll.transform(p.x, p.y)
                return lat, lon, how
        if house:
            ll = nominatim(street, house)
            if ll:
                return ll[0], ll[1], "Nominatim"
        p, how = by_linear(sk, house)
        if p is not None:
            lon, lat = to_ll.transform(p.x, p.y)
            return lat, lon, how
        return None, None, "не найден"

    for r in order.itertuples():
        sk, hn = street_key(r.street), house_norm(r.house)
        ei = next((i for i, (k, h) in enumerate(egr_keys) if hn and same_street(k, sk) and h == hn), None)
        if ei is None and not hn:                        # без номера дома — по названию
            ei = next((i for i, nm in enumerate(egr["name"].fillna("")) if len(nm) > 8 and nm.lower()[:20] in r.name.lower()), None)
        cat = egr.iloc[ei]["category"] if ei is not None else "Ценный градоформирующий объект"
        if ei is not None:
            used.add(ei)
        lat, lon, how = locate(r.street, r.house, ei, r.name)
        rows.append(dict(n=r.n, name=r.name, address=f"{r.street}, {r.house}".strip(", "),
                         category=cat, source="Приказ № 2969, прил. 2" + (" + ЕГРОКН" if ei is not None else ""),
                         method=how, lat=lat, lon=lon))
    for i, e in egr.iterrows():                          # ОКН из реестра, которых нет в перечне Приказа
        if i in used:
            continue
        lat, lon, how = locate(e["street"], e["house"], i, str(e["name"]))
        rows.append(dict(n=f"Е{i}", name=e["name"], address=f"{e['street']}, {e['house']}".strip(", "),
                         category=e["category"], source="ЕГРОКН", method=how, lat=lat, lon=lon))
    df = pd.DataFrame(rows)
    df.to_csv(OBJ_CACHE, index=False, encoding="utf-8-sig")
    print("  способы привязки:", df["method"].str.extract(r"^(\S+)")[0].value_counts().to_dict())
    miss = df[df["lat"].isna()]
    if len(miss):
        print(f"  ⚠ не привязаны ({len(miss)}): " + "; ".join(miss["address"].head(15)))
    df = df[df["lat"].notna()]
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"].astype(float), df["lat"].astype(float)),
                            crs=4326).to_crs(CRS_M)


# ───────────────────────── ИСТОРИЧЕСКОЕ ЯДРО (УРОВЕНЬ 3) ─────────────────────────
def okn_points(d, okn_csv):
    """Точки для плотности: по умолчанию — перечень Приказа (прил. 2) + ЕГРОКН; либо свой CSV lat,lon."""
    if okn_csv:
        df = pd.read_csv(okn_csv)
        g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326).to_crs(CRS_M)
        if "category" not in g:
            g["category"] = "ОКН"
        return g
    return geocode_objects(d)


def is_okn(cat):
    return cat.astype(str).str.contains("значения", na=False)


def find_named(d, sub):
    n = d["named"]
    m = n[n["name"].astype(str).str.contains(sub, case=False, na=False)]
    return m


def pick_part(geom, pt):
    if geom.is_empty:
        return geom
    parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    hit = [p for p in parts if p.contains(pt)]
    return hit[0] if hit else max(parts, key=lambda p: p.area)


def georef_scheme(img, settlement, order_pts):
    """Привязка схемы-скриншота: аффинное преобразование UTM → px, подобранное так, чтобы контур
    исторического поселения (Приказ № 2969) лёг на красный пунктир схемы. Возвращает Affine px → UTM."""
    from affine import Affine
    from scipy.ndimage import distance_transform_edt, map_coordinates
    from scipy.optimize import minimize
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    red = (R > 100) & (R < 175) & (G < 80) & (B < 70)
    red[int(img.shape[0] * 0.9):, int(img.shape[1] * 0.64):] = False      # легенда схемы
    P = {int(n): (g.x, g.y) for n, g in zip(order_pts["n"], order_pts.geometry)}
    A = np.array([P[k] for k in OVRAG_GCP]); Bp = np.array(list(OVRAG_GCP.values()))
    c0 = A.mean(0)
    M, *_ = np.linalg.lstsq(np.c_[A - c0, np.ones(len(A))], Bp, rcond=None)
    ring = settlement.exterior
    xy = np.array([ring.interpolate(t).coords[0] for t in np.arange(0, ring.length, 10)]) - c0
    dt = distance_transform_edt(~red)

    def to_px(p):
        return np.c_[p[0] * xy[:, 0] + p[1] * xy[:, 1] + p[4], p[2] * xy[:, 0] + p[3] * xy[:, 1] + p[5]]

    def cost(p):
        q = to_px(p)
        return np.mean(np.minimum(map_coordinates(dt, [q[:, 1], q[:, 0]], order=1, mode="nearest"), 8) ** 2)
    p0 = [M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[2, 0], M[2, 1]]
    p = minimize(cost, p0, method="Powell", options=dict(maxiter=20000, xtol=1e-6, ftol=1e-8)).x
    q = to_px(p)
    err = map_coordinates(dt, [q[:, 1], q[:, 0]], order=1)
    Ai = np.linalg.inv(np.array([[p[0], p[1]], [p[2], p[3]]]))
    off = c0 - Ai @ np.array([p[4], p[5]])
    print(f"  привязка схемы оврагов: {1 / np.sqrt(abs(np.linalg.det(np.array([[p[0], p[1]], [p[2], p[3]]])))):.1f} м/px, "
          f"медианное отклонение от границы Приказа {np.median(err):.1f} px")
    return Affine(Ai[0, 0], Ai[0, 1], off[0], Ai[1, 0], Ai[1, 1], off[1])


def scheme_ravine(img, aff, dot):
    """Контур оврага на схеме: связная коричневая заливка у точки выноски, сглаженная и переведённая в UTM."""
    from scipy.ndimage import binary_closing, binary_dilation, binary_opening, label as nd_label
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    br = (R > 170) & (R < 215) & (G > 115) & (G < 160) & (B > 95) & (B < 140) & (R - G > 35)
    dark = (R < 60) & (G < 60) & (B < 60)                     # точки выносок внутри заливки
    m = binary_opening(binary_closing(br | (dark & binary_dilation(br, iterations=2))))
    lab, _ = nd_label(m)
    c, r = int(round(dot[0])), int(round(dot[1]))
    win = lab[r - 3:r + 4, c - 3:c + 4]
    k = np.bincount(win[win > 0]).argmax()
    sm = gaussian_filter(np.pad((lab == k).astype(float), 3), 0.9)
    f, a = plt.subplots()
    segs = a.contour(sm, levels=[0.5]).allsegs[0]
    plt.close(f)
    polys = []
    for s in segs:
        if len(s) >= 4:
            polys.append(Polygon([aff * (x - 3, y - 3) for x, y in s]).buffer(0))
    return max(unary_union(polys).geoms, key=lambda g: g.area) if unary_union(polys).geom_type == "MultiPolygon" \
        else unary_union(polys)


def split_osm_ravine(g, aff):
    """Система «Успенский овраг» в OSM: верхняя Y-образная часть — Ямской овраг (по схеме), нижняя — Успенский.
    Разрез — по самому узкому месту ствола между точками выносок двух оврагов."""
    yy = aff * tuple(np.array(OVRAG_DOTS["Ямской овраг"]) + 0.5)
    yu = aff * tuple(np.array(OVRAG_DOTS["Успенский овраг"]) + 0.5)
    x0, _, x1, _ = g.bounds
    lo, hi = sorted((yu[1], yy[1]))
    cand = [(g.intersection(LineString([(x0 - 1, y), (x1 + 1, y)])).length, y)
            for y in np.arange(lo + 0.25 * (hi - lo), hi, 5)]
    y = min(cand)[1]
    top = pick_part(g.intersection(box(x0 - 1, y, x1 + 1, g.bounds[3] + 1)), Point(yy))
    return top, g.difference(top.buffer(0.01)).buffer(0)


def ovragi(d, settlement, order_pts):
    """Никольский овраг — со схемы data/Ovrag.png; Ямской и Успенский — полигон OSM, разделённый по схеме.
    Результат кэшируется в data/ovragi.gpkg (можно поправить в QGIS и перезапустить)."""
    if OVRAGI.exists():
        return gpd.read_file(OVRAGI)
    from PIL import Image
    img = np.array(Image.open(OVRAG_PNG).convert("RGB")).astype(int)
    aff = georef_scheme(img, settlement, order_pts)
    nik = scheme_ravine(img, aff, OVRAG_DOTS["Никольский овраг"]).simplify(2)
    osm = find_named(d, OSM_RAVINE)
    osm = unary_union(osm[osm.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].geometry)
    if osm.is_empty:                                       # нет в OSM — тоже со схемы
        yam = scheme_ravine(img, aff, OVRAG_DOTS["Ямской овраг"]).simplify(2)
        yam, usp = split_osm_ravine(yam, aff)
        src = "схема «Касимовские овраги»"
    else:
        yam, usp = split_osm_ravine(osm, aff)
        src = "OSM, разделение по схеме «Касимовские овраги»"
    g = gpd.GeoDataFrame({"name": ["Никольский овраг", "Ямской овраг", "Успенский овраг"],
                          "source": ["схема «Касимовские овраги» (data/Ovrag.png)", src, src]},
                         geometry=[nik, yam, usp], crs=CRS_M)
    g.to_file(OVRAGI, driver="GPKG")
    print("  овраги → " + str(OVRAGI) + ": " + ", ".join(f"{n} {a/1e4:.1f} га" for n, a in zip(g["name"], g.area)))
    return g


def street_graph(edges):
    import networkx as nx
    G = nx.Graph()
    minor = re.compile(r"footway|path|service|track|steps|pedestrian|cycleway")
    for u, v, g, hw in zip(edges["u"], edges["v"], edges.geometry, edges["highway"].astype(str)):
        w = g.length * (4 if minor.search(hw) else 1)         # держимся улиц, а не проездов и тропинок
        if not G.has_edge(u, v) or G[u][v]["w"] > w:
            G.add_edge(u, v, w=w, geom=g)
    xy = {}
    for u, v, g in zip(edges["u"], edges["v"], edges.geometry):
        xy[u], xy[v] = g.coords[0], g.coords[-1]
    return G, xy


def street_path(G, xy, a, b):
    """Линия по улицам от узла a до узла b (кратчайший путь)."""
    import networkx as nx
    nodes = nx.shortest_path(G, a, b, weight="w")
    coords = [xy[a]]
    for u, v in zip(nodes, nodes[1:]):
        cs = list(G[u][v]["geom"].coords)
        if Point(cs[0]).distance(Point(coords[-1])) > Point(cs[-1]).distance(Point(coords[-1])):
            cs = cs[::-1]
        coords += cs[1:]
    return coords


def nearest_node(xy, p, allowed=None):
    ids = [n for n in xy if allowed is None or n in allowed]
    return min(ids, key=lambda n: Point(xy[n]).distance(p))


def reach(p, target, extra=10):
    """Отрезок от p до ближайшей точки target, продлённый на extra м (чтобы гарантированно пересечь)."""
    q = nearest_points(p, target)[1]
    v = np.array([q.x - p.x, q.y - p.y]); v = v / max(np.hypot(*v), 1e-6)
    return LineString([(p.x, p.y), (q.x + v[0] * extra, q.y + v[1] * extra)])


def historic_core(d, settlement, order_pts):
    sq = find_named(d, "Соборная площадь")
    centre = unary_union(sq.geometry).centroid if len(sq) else \
        gpd.GeoSeries([Point(FALLBACK_CENTRE[::-1])], crs=4326).to_crs(CRS_M).iloc[0]
    rv = ovragi(d, settlement, order_pts)
    geom = dict(zip(rv["name"], rv.geometry))
    nik, yam, usp = geom["Никольский овраг"], geom["Ямской овраг"], geom["Успенский овраг"]
    east = unary_union([yam, usp])
    oka_parts, _ = river_geoms(d["water"])
    oka = unary_union(oka_parts)

    # вершины и устья: Никольский идёт с юго-запада на северо-восток, Успенский — с севера на юг
    nb = list(nik.exterior.coords)
    nik_head = Point(max(nb, key=lambda c: c[0] + c[1])); nik_mouth = Point(min(nb, key=lambda c: c[0] + c[1]))
    yam_head = Point(max(yam.exterior.coords, key=lambda c: c[1]))
    usp_mouth = Point(min(east.exterior.coords if east.geom_type == "Polygon" else
                          [c for g in east.geoms for c in g.exterior.coords], key=lambda c: c[1]))

    # север: по улицам от вершины Никольского оврага к вершине Ямского
    e = d["edges"]
    G, xy = street_graph(e)
    st = e[e["name"].map(lambda v: any(NORTH_START.lower() in n.lower() for n in edge_names(v)))]
    # первый участок — по NORTH_START от проекции вершины оврага до первого перекрёстка
    via = [Point(*street_crossing(e, a, b, near=nik_head.union(yam_head).centroid)) for a, b in NORTH_VIA]
    sl = linemerge(unary_union(st.geometry))
    sl = min(sl.geoms, key=lambda g: g.distance(via[0])) if hasattr(sl, "geoms") else sl
    ta, tb = sl.project(nik_head), sl.project(via[0])
    north = list(substring(sl, min(ta, tb), max(ta, tb)).coords)
    if Point(north[0]).distance(nik_head) > Point(north[-1]).distance(nik_head):
        north = north[::-1]
    way = [nearest_node(xy, p) for p in via]
    for a, b in zip(way, way[1:]):
        north += street_path(G, xy, a, b)[1:]
    north = LineString(north)

    # юго-восток: от устья Успенского оврага к Оке — по улице, спускающейся к реке
    # (улица в коридоре 80 м вдоль кратчайшего направления к берегу; если её нет — прямой отрезок)
    straight = reach(usp_mouth, oka, extra=0)
    near = e[e.geometry.within(straight.buffer(80))]
    Gs, xys = street_graph(near) if len(near) else (None, None)
    try:
        south = LineString([usp_mouth.coords[0]] + street_path(
            Gs, xys, nearest_node(xys, usp_mouth), nearest_node(xys, Point(straight.coords[-1]))))
    except Exception:
        south = straight
    barrier = unary_union([nik, east, north, south,
                           reach(Point(north.coords[0]), nik, 3), reach(Point(north.coords[-1]), yam, 3),
                           reach(nik_mouth, oka), reach(Point(south.coords[-1]), oka)]).buffer(1)
    area = settlement.difference(oka).difference(barrier)
    core = pick_part(area.buffer(0), centre)
    core = core.buffer(-6).buffer(6)                         # убираем «волоски» вдоль бровок
    core = pick_part(core, centre).simplify(1)
    return dict(core=core, centre=centre, ravines=rv, north=north, south=south, nik=nik, yam=yam, usp=usp)


# ───────────────────────── ОФОРМЛЕНИЕ ─────────────────────────
HALO = [pe.withStroke(linewidth=2.6, foreground="white")]
C_BLD = "#DDD8CC"                       # здания (фон)
C_BLD2 = "#D9A88C"                      # здания в границах историч. поселения
C_BLD3 = "#B9707A"                      # здания в границах ядра
C_BLD_OKN = "#5A2A30"                   # здания-ОКН
C_BLD_VAL = "#A7676F"                   # здания — ценные градоформирующие объекты
# точки объектов: категория (начало строки в ЕГРОКН) → (цвет, размер)
POINT_STYLE = {"Федерального": ("#2B0F14", 26), "Регионального": ("#6E1F2B", 18),
               "Местного": ("#A8434F", 14), "Ценный": ("#6B6157", 7)}
C_RAV = "#6B7F4E"


def scalebar(ax, length_m, loc=(0.03, 0.04)):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * loc[0]; y = y0 + (y1 - y0) * loc[1]
    h = (y1 - y0) * 0.006
    ax.plot([x, x + length_m], [y, y], color=C_TEXT, lw=2.4, solid_capstyle="butt", zorder=30)
    ax.plot([x, x + length_m / 2], [y, y], color="white", lw=1.2, solid_capstyle="butt", zorder=31)
    for xx in (x, x + length_m):
        ax.plot([xx, xx], [y - h, y + h], color=C_TEXT, lw=1.2, zorder=32)
    lab = f"{length_m/1000:g} км" if length_m >= 1000 else f"{length_m:g} м"
    ax.text(x + length_m / 2, y + h * 2, lab, ha="center", va="bottom", fontsize=7.5, color=C_TEXT,
            zorder=32, path_effects=HALO)


def north(ax, loc=(0.955, 0.93)):
    ax.annotate("", xy=loc, xytext=(loc[0], loc[1] - 0.06), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=C_TEXT, lw=1.4), zorder=30)
    ax.text(loc[0], loc[1] + 0.008, "С", transform=ax.transAxes, ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=C_TEXT, path_effects=HALO, zorder=30)


def frame(bounds, size):
    """Фигура ровно в размер блока на слайде (px макета): без рамки, заголовков и полей.
    bounds расширяется до пропорций блока, чтобы карта не искажалась."""
    extent = fit(bounds, size[0] / size[1])
    fig = plt.figure(figsize=(size[0] / PPI, size[1] / PPI))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_BG); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3])
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


def legend_header(text):
    """Заголовок группы в легенде (строка без значка)."""
    return Line2D([], [], linestyle="none", label=text)


def legend(ax, handles, loc="lower left", anchor=(0.02, 0.09), fs=8):
    lg = ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, frameon=True, framealpha=0.95,
                   fontsize=fs, edgecolor="none", fancybox=False, borderpad=0.9, labelspacing=0.45,
                   handlelength=1.8, handletextpad=0.7)
    lg.get_frame().set_facecolor("white")
    for t, h in zip(lg.get_texts(), lg.legend_handles):
        if isinstance(h, Line2D) and h.get_linestyle() in ("None", "none") and h.get_marker() in ("None", "none", None, ""):
            t.set_fontweight("bold"); t.set_color("#555"); t.set_fontsize(fs - 0.5)  # заголовок группы
    lg.set_zorder(40)


def basemap(ax, d, buildings=None):
    """Вода, улицы, здания. buildings: список (GeoDataFrame, цвет) поверх фоновых зданий."""
    w = d["water"]
    if len(w):
        w[w.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].plot(ax=ax, color=C_WATER, lw=0, zorder=2)
        w[w.geometry.geom_type.isin(["LineString", "MultiLineString"])].plot(ax=ax, color=C_WATER, lw=1.4, zorder=2)
    d["edges"].plot(ax=ax, color=C_ROAD, lw=0.5, zorder=3)
    d["buildings"].plot(ax=ax, color=C_BLD, lw=0, zorder=4)
    for g, c in (buildings or []):
        if len(g):
            g.plot(ax=ax, color=c, lw=0, zorder=5)


def bld_within(d, geom, frac=0.5):
    """Здания, у которых не меньше frac площади внутри geom."""
    b = d["buildings"]
    b = b[b.intersects(geom)]
    return b[b.intersection(geom).area >= frac * b.area]


def okn_buildings(d, okn):
    """Здания, на которые попадают точки ОКН, + здания с тегом heritage."""
    b = d["buildings"]
    hit = gpd.sjoin(b, okn[["geometry"]], predicate="contains", how="inner").index.unique()
    m = b.index.isin(hit)
    if "heritage" in b:
        m |= b["heritage"].notna().values
    return b[m]


def save(fig, name):
    import io, time
    OUT.mkdir(exist_ok=True)
    for ext, kw in (("png", dict(dpi=PPI * EXPORT_SCALE)), ("svg", {})):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, facecolor=C_BG, pad_inches=0, **kw)
        # Windows: файл может быть кратко занят просмотрщиком/антивирусом — повторяем
        for attempt in range(6):
            try:
                (OUT / f"{name}.{ext}").write_bytes(buf.getvalue())
                break
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(1)
    plt.close(fig)
    print(f"  → out/{name}.png, .svg")


def pad(b, k):
    return (b[0] - k, b[1] - k, b[2] + k, b[3] + k)


def fit(b, aspect=1.25):
    """Расширяет экстент до заданного соотношения ширины к высоте (без искажения)."""
    w, h = b[2] - b[0], b[3] - b[1]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ───────────────────────── КАРТЫ ─────────────────────────
# Единая шкала кеглей (pt; на слайде 1 pt ≈ 1,85 px)
FS_LEG = 8.2        # легенда
FS_LABEL = 8.6      # подписи объектов
FS_STREET = 7.2     # подписи улиц
FS_SMALL = 6.8      # номера точек, мелкие подписи
FS_BIG = 11         # крупные подписи (город, округ)
C_VIEW = "#2F5D8A"  # видовые связи
C_PLAN = "#4A4A52"  # охраняемая планировка
C_ENS = "#8A5A44"   # ансамбли
C_PROB = "#C8402F"  # проблемы
C_POT = "#2E7D6B"   # проекты и потенциал
C_POT_LIGHT = "#9CCBBE"  # то же, светлый тон для широких линий


def label(ax, x, y, text, dx=0, dy=0, fs=FS_LABEL, color=C_TEXT, weight="normal", ha="center", va="center",
          leader=False, z=20):
    """Подпись с белым ореолом; leader — тонкая выноска к точке."""
    kw = dict(arrowprops=dict(arrowstyle="-", color="#666", lw=0.6, shrinkA=0, shrinkB=2)) if leader else {}
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fs, color=color,
                fontweight=weight, ha=ha, va=va, zorder=z, path_effects=HALO, linespacing=1.15, **kw)


def named_geom(d, subs):
    """Первый объект OSM, в названии которого есть одна из подстрок (по слоям named и heritage)."""
    for s in subs:
        for layer in ("named", "heritage"):
            g = d[layer]
            m = g[g["name"].astype(str).str.contains(s, case=False, na=False, regex=False)]
            if not len(m):
                continue
            areal = m[~m.geometry.geom_type.isin(["LineString", "MultiLineString"])]
            # площадь в OSM бывает только набором линий — тогда объединяем их
            return areal.geometry.iloc[0] if len(areal) else unary_union(m.geometry)
    return None


def anchor(g):
    return g.representative_point() if g.geom_type.endswith("Polygon") else g.centroid


def map1(d, size=SLIDE_RIGHT):
    okrug, city, oblast, rivers = d["okrug"], d["city"], d["oblast"], d["rivers"]
    fig, ax = frame(pad(okrug.total_bounds, 2500), size)
    ext = (*ax.get_xlim(), *ax.get_ylim()); ext = (ext[0], ext[2], ext[1], ext[3])
    if len(rivers):
        rp = rivers[rivers.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        rl = rivers[rivers.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        rp.clip(box(*ext)).plot(ax=ax, color=C_WATER, lw=0, zorder=2)
        rl.clip(box(*ext)).plot(ax=ax, color=C_WATER, lw=1.2, zorder=2)
    okrug.plot(ax=ax, facecolor=C_L1, alpha=0.07, edgecolor="none", zorder=1)
    okrug.boundary.plot(ax=ax, color=C_L1, lw=2.4, zorder=10)
    city.plot(ax=ax, facecolor=C_L2, edgecolor="white", lw=0.6, zorder=11)
    c = city.geometry.iloc[0].centroid
    label(ax, c.x, c.y, "г. Касимов", 18, -18, fs=FS_BIG, weight="bold", ha="left")

    # подпись округа — в «самой глубокой» точке округа вдали от города
    og = okrug.geometry.iloc[0]
    free = og.buffer(-6000).difference(city.geometry.iloc[0].buffer(12000))
    free = pick_part(free, og.centroid) if not free.is_empty else og
    lp = polylabel(free if free.geom_type == "Polygon" else max(free.geoms, key=lambda p: p.area), 100)
    label(ax, lp.x, lp.y, "Касимовский\nмуниципальный округ", fs=FS_BIG + 1, color=C_L1, weight="bold", z=12)

    legend(ax, [Line2D([0], [0], color=C_L1, lw=2.4, label="Граница округа"),
                Patch(fc=C_L2, ec="white", label="Город Касимов"),
                Patch(fc=C_WATER, label="Реки и водоёмы")], fs=FS_LEG)
    ins = ax.inset_axes([0.025, 0.72, 0.24, 0.26])
    oblast.plot(ax=ins, fc="#E7E4DB", ec="#8D8A80", lw=0.6)
    okrug.plot(ax=ins, fc=C_L1, ec=C_L1, lw=0.4)
    ins.set_facecolor("white"); ins.set_xticks([]); ins.set_yticks([]); ins.set_aspect("equal")
    ins.set_xlabel(""); ins.set_ylabel("")
    for s in ins.spines.values():
        s.set_visible(False)
    ins.text(0.5, 0.03, "Рязанская область", transform=ins.transAxes, ha="center", va="bottom",
             fontsize=FS_SMALL, color=C_TEXT, path_effects=HALO)
    scalebar(ax, 10000); north(ax)
    save(fig, "slide18_okrug")


def street_label(ax, g, text, fs=FS_STREET):
    """Подпись вдоль улицы (поворот по направлению отрезка в точке подписи)."""
    p = g.interpolate(0.5, normalized=True)
    a = g.interpolate(max(g.length * 0.5 - 15, 0)); b = g.interpolate(min(g.length * 0.5 + 15, g.length))
    ang = np.degrees(np.arctan2(b.y - a.y, b.x - a.x))
    if ang > 90: ang -= 180
    if ang < -90: ang += 180
    ax.text(p.x, p.y, text, fontsize=fs, color=C_TEXT, ha="center", va="center", rotation=ang,
            rotation_mode="anchor", zorder=15, path_effects=HALO)


def map2(d, poly, pts, size=SLIDE_RIGHT):
    fig, ax = frame(pad(poly.bounds, 200), size)
    basemap(ax, d, [(bld_within(d, poly), C_BLD2)])
    gpd.GeoSeries([poly], crs=CRS_M).plot(ax=ax, facecolor=C_L2, alpha=0.08, edgecolor="none", zorder=6)
    gpd.GeoSeries([poly.exterior], crs=CRS_M).plot(ax=ax, color=C_L2, lw=2.6, zorder=8)
    # характерные точки границы с номерами по Приказу (прил. 1)
    pts.plot(ax=ax, color="white", edgecolor=C_L2, markersize=16, linewidth=1.1, zorder=9)
    cx, cy = poly.centroid.x, poly.centroid.y
    # близкие точки (< 60 м подряд) подписываем одним диапазоном: «5–6», «36–38»
    groups, cur = [], [0]
    for i in range(1, len(pts)):
        if pts.geometry.iloc[i].distance(pts.geometry.iloc[cur[-1]]) < 60:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    for gidx in groups:
        ns = [int(pts["n"].iloc[i]) for i in gidx]
        g = unary_union([pts.geometry.iloc[i] for i in gidx]).centroid
        v = np.array([g.x - cx, g.y - cy]); v = v / max(np.hypot(*v), 1) * 11   # наружу от контура
        text = str(ns[0]) if len(ns) == 1 else f"{ns[0]}–{ns[-1]}"
        label(ax, g.x, g.y, text, *v, fs=FS_SMALL, color=C_L2, weight="bold", z=10)
    # подписи улиц контура: самый длинный отрезок на контуре, сдвинутый внутрь
    edges = d["edges"]
    near = edges[edges.geometry.distance(poly.exterior) < 20]
    for s in STREETS:
        key = norm(s)
        m = near[near["name"].astype(str).map(norm).str.contains(re.escape(key), na=False)]
        if not len(m):
            continue
        # участок контура, идущий по этой улице; короткие (клинья у т. 5–8) не подписываем
        run = poly.exterior.intersection(unary_union(m.geometry).buffer(20))
        run = linemerge(run) if run.geom_type == "MultiLineString" else run
        parts = list(run.geoms) if hasattr(run, "geoms") else [run]
        parts = [p_ for p_ in parts if p_.geom_type == "LineString"]
        if not parts:
            continue
        g = max(parts, key=lambda p_: p_.length)
        if g.length < 200:
            continue
        p = g.interpolate(0.5, normalized=True); q = g.interpolate(min(g.length * 0.5 + 10, g.length))
        nx_, ny_ = -(q.y - p.y), q.x - p.x
        nn = max(np.hypot(nx_, ny_), 1e-6); nx_, ny_ = nx_ / nn, ny_ / nn
        if (p.x - cx) * nx_ + (p.y - cy) * ny_ > 0:          # номера точек — снаружи, улицы — внутри
            nx_, ny_ = -nx_, -ny_
        street_label(ax, translate(g, nx_ * 40, ny_ * 40), s)
    legend(ax, [Line2D([0], [0], color=C_L2, lw=2.6, label="Граница исторического поселения\n(Приказ Минкультуры № 2969)"),
                Line2D([0], [0], marker="o", color="w", mfc="white", mec=C_L2, ms=6,
                       label="Характерные точки границы (прил. 1)"),
                Patch(fc=C_BLD2, label="Застройка в границах поселения"),
                Patch(fc=C_BLD, label="Прочая застройка"),
                Patch(fc=C_WATER, label="Ока и водоёмы")], fs=FS_LEG)
    scalebar(ax, 500); north(ax)
    save(fig, "slide19_settlement")


def poly_patch(geom, **kw):
    """PathPatch из (Multi)Polygon — для отсечения растров и заливок по контуру."""
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    verts, codes = [], []
    for p in (geom.geoms if hasattr(geom, "geoms") else [geom]):
        for ring in [p.exterior] + list(p.interiors):
            cs = list(ring.coords)
            verts += cs; codes += [MPath.MOVETO] + [MPath.LINETO] * (len(cs) - 2) + [MPath.CLOSEPOLY]
    return PathPatch(MPath(verts, codes), **kw)


def plot_line(ax, geom, **kw):
    for part in (geom.geoms if hasattr(geom, "geoms") else [geom]):
        if part.geom_type in ("LineString", "LinearRing"):
            xs, ys = part.xy
            ax.plot(xs, ys, **kw)


def plot_points_by_category(ax, okn):
    okn_mask = is_okn(okn["category"])
    for cat, (c, ms) in POINT_STYLE.items():
        sel = okn[okn["category"].astype(str).str.startswith(cat)] if cat != "Ценный" else okn[~okn_mask]
        if len(sel):
            sel.plot(ax=ax, color=c, markersize=ms * 1.3, zorder=12, edgecolor="white", linewidth=0.5)


def point_handles():
    return [Line2D([0], [0], marker="o", color="w", mfc=c, mec="white", ms=(ms * 1.3) ** 0.5 * 1.25, label=lab)
            for (c, ms), lab in zip(POINT_STYLE.values(),
                                    ["ОКН федерального значения", "ОКН регионального значения",
                                     "ОКН местного значения", "Ценный градоформирующий объект"])]


C_RAV_TXT = "#4F6B3A"   # подписи природных элементов
C_BLD_CORE = "#DDB3AA"  # рядовая застройка в границе проекта


def line_label(ax, g, text, off=0, fs=FS_STREET, color=C_TEXT, at=0.5, weight="normal"):
    """Подпись вдоль линии g (в точке at), сдвинутая на off м по нормали."""
    p = g.interpolate(at, normalized=True)
    a = g.interpolate(max(g.length * at - 20, 0)); b = g.interpolate(min(g.length * at + 20, g.length))
    tx, ty = b.x - a.x, b.y - a.y; n = max(np.hypot(tx, ty), 1e-6)
    ang = np.degrees(np.arctan2(ty, tx))
    ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
    ax.text(p.x - ty / n * off, p.y + tx / n * off, text, fontsize=fs, color=color, ha="center", va="center",
            rotation=ang, rotation_mode="anchor", zorder=15, fontweight=weight, path_effects=HALO)


def map3(d, settlement, okn, r, size=SLIDE_RIGHT):
    core, rv = r["core"], r["ravines"]
    fb = unary_union([core, r["nik"], r["yam"], r["usp"]]).bounds
    # запас слева-снизу — под легенду (там Ока)
    fig, ax = frame((fb[0] - 430, fb[1] - 330, fb[2] + 120, fb[3] + 110), size)

    okn_mask = is_okn(okn["category"])
    basemap(ax, d, [(bld_within(d, settlement), "#E6CDBE"), (bld_within(d, core), C_BLD_CORE),
                    (okn_buildings(d, okn[~okn_mask]), C_BLD_VAL), (okn_buildings(d, okn[okn_mask]), C_BLD_OKN)])
    gpd.GeoSeries([core], crs=CRS_M).plot(ax=ax, facecolor=C_L3, alpha=0.06, edgecolor="none", zorder=5.5)
    plot_line(ax, settlement.exterior, color=C_L2, lw=1.4, ls=(0, (5, 3)), zorder=6, alpha=0.9)
    rv.plot(ax=ax, facecolor=C_RAV, alpha=0.45, edgecolor=C_RAV, lw=0.9, zorder=7)
    plot_line(ax, core.exterior, color="white", lw=5.2, zorder=9.5, solid_joinstyle="round")
    plot_line(ax, core.exterior, color=C_L3, lw=3.0, zorder=10, solid_joinstyle="round")
    plot_points_by_category(ax, okn)

    # подписи оврагов — снаружи контура, курсивом природного цвета
    for g, text, dx, dy in [(r["nik"], "Никольский\nовраг", -62, 22), (r["yam"], "Ямской\nовраг", 52, 18),
                            (r["usp"], "Успенский\nовраг", 58, -26)]:
        p = anchor(g)
        label(ax, p.x, p.y, text, dx, dy, fs=FS_LABEL, color=C_RAV_TXT, weight="bold", leader=True)

    def tag(g, text, dx, dy):
        if g is not None:
            p = anchor(g)
            label(ax, p.x, p.y, text, dx, dy, leader=True)
    tag(named_geom(d, ["Соборная площадь"]), "Соборная пл.", 70, -30)
    m = named_geom(d, ["Ханская мечеть"])
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    if m is not None and x0 < anchor(m).x < x1 and y0 < anchor(m).y < y1:
        tag(m, "Ханская мечеть\n(Татарская слобода)", 40, -36)
    # улицы северной границы
    for key, text in [("Карла Либкнехта", "ул. Карла Либкнехта")]:
        seg = r["north"].intersection(edges_named(d, [key]).buffer(3))
        seg = linemerge(seg) if seg.geom_type == "MultiLineString" else seg
        seg = max(seg.geoms, key=lambda g: g.length) if hasattr(seg, "geoms") else seg
        if not seg.is_empty and seg.length > 120:
            line_label(ax, seg, text, off=30)
    nb = d["edges"][d["edges"]["name"].astype(str).str.contains("Набережная", na=False)]
    nb = nb[nb.geometry.intersects(core.buffer(40)) & (nb.geometry.length > 80)]
    if len(nb):
        line_label(ax, nb.geometry.iloc[int(np.argmax(nb.geometry.length))], "ул. Набережная", off=-16)
    # «р. Ока» — в свободной воде слева, между подписями и легендой
    w_, h_ = x1 - x0, y1 - y0
    oka = unary_union(river_geoms(d["water"])[0]).intersection(
        box(x0, y0 + 0.40 * h_, x0 + 0.30 * w_, y0 + 0.62 * h_))
    if not oka.is_empty:
        q = polylabel(max(oka.geoms, key=lambda g: g.area) if hasattr(oka, "geoms") else oka, 5)
        label(ax, q.x, q.y, "р. Ока", fs=FS_BIG, color="#3F6E93", weight="bold")

    legend(ax, [legend_header("ГРАНИЦА ПРОЕКТА"),
                Line2D([0], [0], color=C_L3, lw=3.0, label="Граница проекта (уровень 3)"),
                Patch(fc=C_RAV, alpha=0.45, ec=C_RAV, label="Овраги: Никольский (запад),\nУспенский и Ямской (восток)"),
                Line2D([0], [0], color=C_L2, lw=1.4, ls=(0, (5, 3)), label="Историческое поселение"),
                legend_header("ОБЪЕКТЫ")]
               + point_handles()
               + [Patch(fc=C_BLD_OKN, label="Здания ОКН"),
                  Patch(fc=C_BLD_VAL, label="Здания ценных объектов"),
                  Patch(fc=C_BLD_CORE, label="Прочая застройка в границе проекта")], fs=FS_LEG, anchor=(0.02, 0.085))
    scalebar(ax, 200); north(ax)
    save(fig, "slide20_core")


# ───────── Сводная схема ценностей (слайд 34) и проблем (слайд 35) ─────────
# Приказ № 2969, прил. 2: охраняемая планировочная структура (раздел 2)
MAIN_STREETS = ["Набережная", "Рязанский", "Советская", "50 лет ВЛКСМ", "Ленина", "Татарская"]
PROTECTED_STREETS = ["Академика", "Карла Маркса", "Большакова", "Илюшкина", "Карла Либкнехта", "Свердлова",
                     "Старопосадская", "Дзержинского", "Урицкого", "Пролетарская", "Нариманова", "Комсомольская",
                     "Володарского", "Воровского", "Октябрьская", "Окская", "Кокорева", "Губарева", "Комарова",
                     "Чижова", "Школьный"]
# раздел 4: высотные доминанты и акценты — (подпись, подстроки названия в OSM)
DOMINANTS = [("Вознесенский собор", ["Вознесенский собор"]), ("Благовещенская ц.", ["Благовещенская"]),
             ("Успенская ц.", ["Успенская церковь"]), ("Троицкая ц.", ["Церковь Троицы"]),
             ("Никольская ц.", ["Никольская церковь"]), ("Георгиевская ц.", ["Богоявленская", "Георгиевск"]),
             ("Ильинская ц.", ["Ильинский храм", "Ильинская"]), ("Старая Татарская\nмечеть", ["Ханская мечеть"])]
SQUARES = [("Соборная пл.", ["Соборная площадь"]), ("пл. Ленина", ["площадь Ленина"]),
           ("пл. Победы", ["площадь Победы"]), ("Никольская пл.", ["площадь Пионеров", "Никольская площадь", "Никольская церковь"])]


def edges_named(d, keys, within=None):
    e = d["edges"]
    m = e["name"].map(lambda v: any(k.lower() in n.lower() and "переулок ленина" not in n.lower()
                                    for n in edge_names(v) for k in keys))
    g = unary_union(e[m].geometry)
    return g.intersection(within) if within is not None else g


def heritage_context(d, okn, settlement, ovragi_gdf=None):
    """Общие для слайдов 34–35 слои: зоны ансамблей, доминанты, площади, берег, овраги."""
    obj = okn.copy()
    sob = named_geom(d, ["Соборная площадь"])
    sob_c = unary_union(sob).centroid if sob is not None else None
    mosque = named_geom(d, ["Ханская мечеть"])
    zones = {}
    if sob_c is not None:
        near = obj[obj.distance(sob_c) < 170]
        zones["Ансамбль\nСоборной площади"] = unary_union(near.geometry).convex_hull.buffer(25)
        cross = street_crossing(d["edges"], "Советская", "50 лет ВЛКСМ")
        sov = edges_named(d, ["Советская"])
        if cross is not None:
            axis = LineString([(sob_c.x, sob_c.y), tuple(cross)])
            parts = [g for g in (sov.geoms if hasattr(sov, "geoms") else [sov]) if g.distance(axis) < 30]
            zones["Линейный ансамбль\nул. Советской"] = unary_union(parts).buffer(45).difference(zones["Ансамбль\nСоборной площади"])
    if mosque is not None:
        near = obj[obj.distance(anchor(mosque)) < 260]
        zones["Татарская\nслобода"] = unary_union(near.geometry).convex_hull.buffer(30)
    nab = obj[obj["address"].astype(str).str.contains("Набережная", na=False)]
    nab = nab[nab["address"].map(lambda a: 21 <= (house_num(a.split(",")[-1]) or 0) <= 36)]
    if len(nab) >= 3:
        zones["Набережная\n(д. 21–36)"] = unary_union(nab.geometry).convex_hull.buffer(35)

    dominants = [(t, anchor(g)) for t, subs in DOMINANTS if (g := named_geom(d, subs)) is not None]
    squares = [(t, anchor(g)) for t, subs in SQUARES if (g := named_geom(d, subs)) is not None]

    oka_parts, bab = river_geoms(d["water"])
    shore = unary_union([p.exterior for p in oka_parts]).intersection(settlement.buffer(30))
    rv = [g for g in d["named"][d["named"]["name"].astype(str).str.contains("овраг", case=False, na=False)].geometry
          if g.geom_type.endswith("Polygon")]
    rv += [g for g in d["ravines"].geometry if g.geom_type.endswith("Polygon")]
    if ovragi_gdf is not None:
        rv += list(ovragi_gdf.geometry)
    ravines = unary_union(rv).intersection(settlement.buffer(150)) if rv else None
    mouth = None
    if bab is not None:
        oka = unary_union(oka_parts)
        mouth = min((bab.interpolate(0), bab.interpolate(bab.length)), key=lambda p: p.distance(oka))
    oka_line = unary_union(d["water"][d["water"]["name"].astype(str).str.fullmatch("Ока")].geometry)
    return dict(zones=zones, dominants=dominants, squares=squares, shore=shore, ravines=ravines, ovragi=ovragi_gdf,
                sob_c=sob_c, mouth=mouth, oka_line=oka_line)


def values_frame(settlement, size):
    b = settlement.bounds
    return frame((b[0] - 250, b[1] - 120, b[2] + 120, b[3] + 120), size)


def map_values(d, settlement, core, okn, ctx, size=SLIDE_LEFT):
    """Сводная схема ценностей: 4 карты-панели (ландшафт · планировка · застройка · силуэт и виды),
    на каждой — одна тема поверх общей приглушённой подложки."""
    W, H = size
    gap = 0                                              # панели встык, разделены тонкими линиями
    pw, ph = (W - gap) / 2, (H - gap) / 2
    fig = plt.figure(figsize=(W / PPI, H / PPI))
    fig.patch.set_facecolor("white")
    b = settlement.bounds
    ext = fit((b[0] - 60, b[1] - 60, b[2] + 60, b[3] + 60), pw / ph)

    okn_mask = is_okn(okn["category"])
    b_okn, b_val = okn_buildings(d, okn[okn_mask]), okn_buildings(d, okn[~okn_mask])
    oka_parts, bab = river_geoms(d["water"])
    wpoly = d["water"][d["water"].geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    wline = d["water"][d["water"].geometry.geom_type.isin(["LineString", "MultiLineString"])]

    def panel(i, title):
        col, row = i % 2, i // 2
        ax = fig.add_axes([col * (pw + gap) / W, 1 - (row + 1) * ph / H - row * gap / H, pw / W, ph / H])
        ax.set_facecolor(C_BG); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(ext[0], ext[2]); ax.set_ylim(ext[1], ext[3])
        for s in ax.spines.values():
            s.set_visible(False)
        wpoly.plot(ax=ax, color=C_WATER, lw=0, zorder=1)
        wline.plot(ax=ax, color=C_WATER, lw=1.0, zorder=1)
        d["edges"].plot(ax=ax, color="#DEDAD1", lw=0.35, zorder=2)
        d["buildings"].plot(ax=ax, color="#E4E0D7", lw=0, zorder=2)
        plot_line(ax, settlement.exterior, color="#A9A398", lw=0.8, ls=(0, (4, 3)), zorder=3)
        ax.text(0.035, 0.955, title, transform=ax.transAxes, ha="left", va="top", fontsize=FS_BIG,
                fontweight="bold", color=C_TEXT, zorder=40, path_effects=HALO)
        return ax

    def sub(ax, text):                                   # пояснение под заголовком панели
        ax.text(0.035, 0.885, text, transform=ax.transAxes, ha="left", va="top", fontsize=FS_SMALL,
                color="#555", zorder=40, linespacing=1.3, path_effects=HALO)

    # 1 — ЛАНДШАФТ
    ax = panel(0, "1  Ландшафт")
    sub(ax, "высокий левый берег Оки,\nрасчленённый оврагами")
    plot_line(ax, ctx["shore"], color=C_RAV, lw=5, solid_capstyle="round", zorder=5)
    if ctx["ravines"] is not None and not ctx["ravines"].is_empty:
        gpd.GeoSeries([ctx["ravines"]], crs=CRS_M).plot(ax=ax, color=C_RAV, alpha=0.55, lw=0, zorder=4)
    if bab is not None:
        plot_line(ax, bab.intersection(box(*ext)), color="#5C8FB5", lw=1.6, zorder=4)
    s1 = max(ctx["shore"].geoms, key=lambda g: g.length) if hasattr(ctx["shore"], "geoms") else ctx["shore"]
    p = s1.interpolate(0.5, normalized=True)
    label(ax, p.x, p.y, "склон берега\nОки, >2,5 км", -48, -6, fs=FS_SMALL + 0.4, color="#4F6B3A", weight="bold")
    if ctx["ovragi"] is not None:
        for nm, g in zip(ctx["ovragi"]["name"], ctx["ovragi"].geometry):
            dx, dy = {"Никольский овраг": (-26, -26), "Ямской овраг": (30, 16), "Успенский овраг": (28, -22)}[nm]
            p = anchor(g)
            label(ax, p.x, p.y, nm.replace(" ", "\n"), dx, dy, fs=FS_SMALL + 0.4, color="#4F6B3A", weight="bold",
                  leader=True)
    if bab is not None:
        p = bab.intersection(settlement.buffer(200))
        p = (p if p.geom_type == "LineString" else max(p.geoms, key=lambda g: g.length)).interpolate(0.5, normalized=True) \
            if not p.is_empty else bab.centroid
        label(ax, p.x, p.y, "пойма\nБабёнки", 38, 0, fs=FS_SMALL + 0.4, color="#4F6B3A", weight="bold")
    ok_ = unary_union(oka_parts).intersection(box(ext[0], ext[1], ext[2], ext[1] + (ext[3] - ext[1]) * 0.35))
    if not ok_.is_empty:
        p = ok_.representative_point()
        label(ax, p.x, p.y, "р. Ока", fs=FS_LABEL, color="#3F6E93", weight="bold")
    north(ax, loc=(0.93, 0.92))

    # 2 — ПЛАНИРОВКА
    ax = panel(1, "2  Планировка")
    sub(ax, "регулярная сетка XVIII–XIX вв.:\nулицы вдоль реки и спуски к ней")
    plot_line(ax, edges_named(d, PROTECTED_STREETS, settlement), color=C_PLAN, lw=0.9, zorder=5)
    plot_line(ax, edges_named(d, MAIN_STREETS, settlement), color=C_PLAN, lw=2.6, zorder=6)
    for t, p in ctx["squares"]:
        ax.plot(p.x, p.y, marker="s", ms=8, mfc="white", mec=C_PLAN, mew=1.8, zorder=8)
    for key, text, dx, dy in [("Советская", "ул. Советская", 40, -6), ("Набережная", "ул. Набережная", -10, -20)]:
        g = edges_named(d, [key], settlement)
        if not g.is_empty:
            g1 = linemerge(g) if g.geom_type == "MultiLineString" else g
            g1 = max(g1.geoms, key=lambda x: x.length) if hasattr(g1, "geoms") else g1
            p = g1.interpolate(0.5, normalized=True)
            label(ax, p.x, p.y, text, dx, dy, fs=FS_SMALL + 0.4, weight="bold", color=C_PLAN)
    sq = dict(ctx["squares"])
    if "Соборная пл." in sq:
        p = sq["Соборная пл."]
        label(ax, p.x, p.y, "Соборная пл.", -12, 12, fs=FS_SMALL, ha="right", color=C_PLAN)

    # 3 — ЗАСТРОЙКА
    ax = panel(2, "3  Застройка")
    sub(ax, "ансамбли и ценная историческая среда")
    for name, z in ctx["zones"].items():
        gpd.GeoSeries([z], crs=CRS_M).plot(ax=ax, facecolor=C_ENS, alpha=0.16, edgecolor="none", zorder=4)
        plot_line(ax, z.boundary, color=C_ENS, lw=1.1, zorder=5)
    if len(b_val):
        b_val.plot(ax=ax, color=C_BLD_VAL, lw=0, zorder=6)
    if len(b_okn):
        b_okn.plot(ax=ax, color=C_BLD_OKN, lw=0, zorder=7)
    zl = {"Ансамбль\nСоборной площади": ("Соборная\nплощадь", -60, -20),
          "Линейный ансамбль\nул. Советской": ("ул. Советская", 58, 8),
          "Татарская\nслобода": ("Татарская\nслобода", 52, -16),
          "Набережная\n(д. 21–36)": ("Набережная", -5, 22)}
    for name, z in ctx["zones"].items():
        t, dx, dy = zl.get(name, (name, 40, 10))
        p = anchor(z)
        label(ax, p.x, p.y, t, dx, dy, fs=FS_SMALL + 0.4, color=C_ENS, weight="bold", leader=True)
    ax.legend(handles=[Patch(fc=C_BLD_OKN, label="ОКН"), Patch(fc=C_BLD_VAL, label="ценные объекты"),
                       Patch(fc=C_ENS, alpha=0.3, ec=C_ENS, label="ансамбли")],
              loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=FS_SMALL, frameon=True, framealpha=0.95,
              edgecolor="none", fancybox=False, handlelength=1.2).set_zorder(40)
    scalebar(ax, 500, loc=(0.72, 0.05))

    # 4 — СИЛУЭТ И ВИДЫ
    ax = panel(3, "4  Силуэт и виды")
    sub(ax, "8 доминант читаются\nс реки единой панорамой")
    dom = dict(ctx["dominants"])
    views = view_links(ctx)
    from matplotlib.patches import FancyArrowPatch
    for a, b_ in views:
        ax.add_patch(FancyArrowPatch((a.x, a.y), (b_.x, b_.y), arrowstyle="-|>", mutation_scale=9, color=C_VIEW,
                                     lw=1.2, ls=(0, (3, 2)), shrinkA=3, shrinkB=8, zorder=6))
        ax.plot(a.x, a.y, "o", ms=6, mfc="white", mec=C_VIEW, mew=1.4, zorder=7)
    for t, p in ctx["dominants"]:
        ax.plot(p.x, p.y, marker="^", ms=10, mfc=C_TEXT, mec="white", mew=1.0, zorder=8)
    if "Вознесенский собор" in dom:
        p = dom["Вознесенский собор"]
        label(ax, p.x, p.y, "Вознесенский\nсобор", 14, 18, fs=FS_SMALL + 0.4, weight="bold", ha="left")
    if "Старая Татарская\nмечеть" in dom:
        p = dom["Старая Татарская\nмечеть"]
        label(ax, p.x, p.y, "минарет\nмечети", 16, -6, fs=FS_SMALL + 0.4, weight="bold", ha="left")
    if "Ильинская ц." in dom:
        p = dom["Ильинская ц."]
        label(ax, p.x, p.y, "Ильинская ц.", 12, 6, fs=FS_SMALL + 0.4, weight="bold", ha="left")
    if views:
        rp = views[0][0]
        label(ax, rp.x, rp.y, "панорама\nс Оки", -10, -22, fs=FS_SMALL + 0.4, color=C_VIEW, weight="bold")
    ax.legend(handles=[Line2D([0], [0], marker="^", color="w", mfc=C_TEXT, mec="white", ms=9, label="доминанты"),
                       Line2D([0], [0], color=C_VIEW, lw=1.2, ls=(0, (3, 2)), marker="o", mfc="white", mec=C_VIEW,
                              ms=5, label="видовые связи")],
              loc="lower left", bbox_to_anchor=(0.02, 0.1), fontsize=FS_SMALL, frameon=True, framealpha=0.95,
              edgecolor="none", fancybox=False, handlelength=1.6).set_zorder(40)
    for a in fig.axes:                                   # geopandas подписывает оси — убираем
        a.set_xlabel(""); a.set_ylabel("")
    # тонкие чёрные разделители панелей (1 px макета)
    lw = 72 / PPI
    fig.add_artist(Line2D([0.5, 0.5], [0, 1], transform=fig.transFigure, color="black", lw=lw, zorder=100))
    fig.add_artist(Line2D([0, 1], [0.5, 0.5], transform=fig.transFigure, color="black", lw=lw, zorder=100))
    save(fig, "slide34_values")


# слайд 35: что проверить на месте — (номер, подпись на карте)
VERIFY = {1: "Бровки\nНикольского оврага", 2: "Торговые ряды,\nкорпуса 2–3", 3: "Бровки Успенского\nи Ямского оврагов",
          4: "Склон и видовые\nплощадки Набережной", 5: "Татарская слобода:\nлавки и усадьбы",
          6: "Объекты прил. 2,\nне найденные по адресу", 7: "Многоэтажная\nзастройка",
          8: "Старый посад,\nИльинская ц."}


def view_links(ctx):
    """Охраняемые видовые связи: (точка обзора, доминанта)."""
    dom = dict(ctx["dominants"])
    views = []
    if not ctx["oka_line"].is_empty and "Вознесенский собор" in dom:
        rp = nearest_points(ctx["oka_line"], dom["Вознесенский собор"])[0]
        views += [(rp, dom[k]) for k in ("Вознесенский собор", "Никольская ц.") if k in dom]
    if ctx["mouth"] is not None and "Ильинская ц." in dom:
        views.append((ctx["mouth"], dom["Ильинская ц."]))
    return views


def verify_points(d, okn, r, ctx, tall):
    """Точки «проверить на месте» (номера — как в VERIFY): {номер: Point}."""
    tr = okn[okn["n"].astype(str) == "10"]                # Торговые ряды
    pos = {}
    pos[1] = anchor(r["nik"])
    if len(tr):
        pos[2] = tr.geometry.iloc[0]
    pos[3] = anchor(r["usp"])
    zast = okn[okn["n"].astype(str) == "2"]
    if len(zast):
        pos[4] = nearest_points(zast.geometry.iloc[0], ctx["shore"])[1] if not ctx["shore"].is_empty else zast.geometry.iloc[0]
    m = named_geom(d, ["Ханская мечеть"])
    if m is not None:
        pos[5] = anchor(m)
    m = named_geom(d, ["площадь Ленина"])
    if m is not None:
        pos[6] = anchor(m)
    if len(tall):
        cs = tall.geometry.centroid
        k = int(np.argmax([cs.distance(c).lt(250).sum() for c in cs]))
        pos[7] = cs.iloc[k]
    il = dict(ctx["dominants"]).get("Ильинская ц.")
    if il is not None:
        pos[8] = il
    return pos


def map_issues(d, settlement, core, okn, r, ctx, size=SLIDE_LEFT):
    """Проблемы, проекты и точки, требующие проверки на месте."""
    fig, ax = values_frame(settlement, size)
    bl = mkd_buildings(d)
    tall = bl[(bl["levels"] >= TALL_LEVELS) & bl.intersects(settlement)]
    print(f"  зданий {TALL_LEVELS}+ эт. в поселении: {len(tall)}")
    okn_mask = is_okn(okn["category"])
    basemap(ax, d, [(okn_buildings(d, okn[~okn_mask]), C_BLD_VAL), (okn_buildings(d, okn[okn_mask]), C_BLD_OKN),
                    (tall, C_PROB)])
    gpd.GeoSeries([settlement.exterior], crs=CRS_M).plot(ax=ax, color=C_L2, lw=1.3, ls="--", zorder=6, alpha=0.8)
    if not core.is_empty:
        plot_line(ax, core.boundary, color=C_L3, lw=2.2, zorder=7)

    # проекты и потенциал
    zones = ctx["zones"]
    sob_zone = zones.get("Ансамбль\nСоборной площади")
    if sob_zone is not None:
        gpd.GeoSeries([sob_zone], crs=CRS_M).plot(ax=ax, facecolor=C_POT, alpha=0.18, edgecolor=C_POT, lw=1.2, zorder=6.5)
    nab = edges_named(d, ["Набережная"], settlement.buffer(30))
    # одна слитная линия без прозрачности — иначе наложения рёбер дают «пятнистость»
    nab = linemerge(unary_union(nab)) if not nab.is_empty else nab
    plot_line(ax, nab, color=C_POT_LIGHT, lw=5, solid_capstyle="round", solid_joinstyle="round", zorder=6.4)
    if ctx["sob_c"] is not None and not nab.is_empty:
        q = nearest_points(ctx["sob_c"], nab)[1]
        ax.annotate("", (q.x, q.y), (ctx["sob_c"].x, ctx["sob_c"].y), zorder=8,
                    arrowprops=dict(arrowstyle="-|>", color=C_POT, lw=2.0, ls=(0, (3, 2)), shrinkA=4, shrinkB=2))

    pos = verify_points(d, okn, r, ctx, tall)
    voffs = {1: (-62, 28), 2: (-30, -44), 3: (55, 30), 4: (-35, -36), 5: (0, -46), 6: (-60, 26),
             7: (55, 22), 8: (-60, 20)}
    for n, p in pos.items():
        ax.plot(p.x, p.y, "o", ms=17, mfc=C_TEXT, mec="white", mew=1.4, zorder=18)
        ax.text(p.x, p.y, str(n), color="white", fontsize=FS_SMALL + 0.6, fontweight="bold", ha="center",
                va="center", zorder=19)
        label(ax, p.x, p.y, VERIFY[n], *voffs.get(n, (40, 20)), fs=FS_SMALL, leader=True, z=17)

    legend(ax, [legend_header("ПРОБЛЕМЫ И УГРОЗЫ"),
                Patch(fc=C_PROB, label=f"Многоэтажная застройка ({TALL_LEVELS}+ эт.)"),
                legend_header("ПРОЕКТЫ И ПОТЕНЦИАЛ"),
                Patch(fc=C_POT, alpha=0.25, ec=C_POT, label="«Слияние времён»: Соборная пл., 2022–2024"),
                Line2D([0], [0], color=C_POT_LIGHT, lw=5, label="Проект набережной"),
                Line2D([0], [0], color=C_POT, lw=2.0, ls=(0, (3, 2)), label="«Третий луч»: Соборная пл. → Набережная"),
                legend_header("КОНТЕКСТ"),
                Line2D([0], [0], color=C_L3, lw=2.2, label="Граница проекта (уровень 3)"),
                Line2D([0], [0], color=C_L2, lw=1.3, ls="--", label="Историческое поселение"),
                Patch(fc=C_BLD_OKN, label="Здания ОКН"),
                Patch(fc=C_BLD_VAL, label="Ценные градоформирующие объекты"),
                legend_header("ПРОВЕРИТЬ НА МЕСТЕ"),
                Line2D([0], [0], marker="o", color="w", mfc=C_TEXT, mec="white", ms=11, label="Точки 1–8 — см. список")],
           fs=FS_LEG, anchor=(0.02, 0.1))
    scalebar(ax, 500); north(ax)
    save(fig, "slide35_issues")


# ───────────────────────── ЭКСПОРТ ДЛЯ QGIS ─────────────────────────
def export_gpkg(d, poly, order_pts, okn, r):
    """Все контуры — в out/boundaries.gpkg (EPSG:32637), по слою на сущность."""
    OUT.mkdir(exist_ok=True)
    p = OUT / "boundaries.gpkg"
    if p.exists():
        p.unlink()

    def put(layer, rows):
        rows = [x for x in rows if x["geometry"] is not None and not x["geometry"].is_empty]
        if rows:
            gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_M).to_file(p, layer=layer, driver="GPKG")

    put("level1_okrug", [dict(level=1, name="Касимовский муниципальный округ",
                              area_ha=round(d["okrug"].geometry.iloc[0].area / 1e4), geometry=d["okrug"].geometry.iloc[0])])
    put("level2_settlement", [dict(level=2, name="Историческое поселение «Касимов»",
                                   source="Приказ Минкультуры России № 2969 от 07.12.2015",
                                   area_ha=round(poly.area / 1e4), geometry=poly)])
    put("level2_order_points", [dict(n=int(n), note="Характерная точка границы, Приказ № 2969, прил. 1", geometry=g)
                                for n, g in zip(order_pts["n"], order_pts.geometry)])
    put("level3_core", [dict(level=3, name="Граница проекта: между Никольским и Успенским/Ямским оврагами",
                             area_ha=round(r["core"].area / 1e4), geometry=r["core"])])
    put("level3_north_line", [dict(note="Север: по улицам между вершинами оврагов", geometry=r["north"])])
    put("ovragi", [dict(name=n, source=s, geometry=g) for n, s, g in
                   zip(r["ravines"]["name"], r["ravines"]["source"], r["ravines"].geometry)])
    o = okn.copy()
    keep = [c for c in ("n", "name", "address", "category", "source", "method") if c in o]
    for c in keep:
        o[c] = o[c].astype(str)
    if len(o):
        o[keep + ["geometry"]].to_file(p, layer="heritage_objects", driver="GPKG")
    print(f"  → {p}")


# ───────────────────────── MAIN ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--okn", help="CSV со своими ОКН: lat,lon[,name] (точнее, чем OSM)")
    args = ap.parse_args()

    d = load_all()
    print("Граница исторического поселения по координатам Приказа № 2969…")
    poly, order_pts = order_boundary(d)

    print("Карта 1…"); map1(d)
    print("Карта 2…"); map2(d, poly, order_pts)
    print("Объекты наследия, плотность и ядро…")
    okn = okn_points(d, args.okn)
    okn = okn[okn.geometry.within(poly.buffer(300))]
    print(f"  объектов в пределах поселения: {len(okn)} "
          f"(из них ОКН: {is_okn(okn['category']).sum()})")
    r = historic_core(d, poly, order_pts)
    print("Карта 3…"); map3(d, poly, okn, r)
    if not r["core"].is_empty:
        ins, inc = okn[okn.within(poly)], okn[okn.within(r["core"])]
        print(f"  граница проекта: {r['core'].area/1e4:.0f} га ({r['core'].area/poly.area*100:.0f}% поселения); "
              f"ценных объектов {(~is_okn(inc['category'])).sum()} из {(~is_okn(ins['category'])).sum()}, "
              f"ОКН {is_okn(inc['category']).sum()} из {is_okn(ins['category']).sum()}")
    ctx = heritage_context(d, okn, poly, r["ravines"])
    print("Схема ценностей…"); map_values(d, poly, r["core"], okn, ctx)
    print("Проблемы и проверка на месте…"); map_issues(d, poly, r["core"], okn, r, ctx)
    print("Экспорт для QGIS…"); export_gpkg(d, poly, order_pts, okn, r)


if __name__ == "__main__":
    main()
