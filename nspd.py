#!/usr/bin/env python3
"""
Сведения ЕГРН с публичной кадастровой карты (nspd.gov.ru) по зданиям исторического поселения.

Слои карты (WMS GetFeatureInfo, ответ — GeoJSON в EPSG:3857):
  36048 — земельные участки ЕГРН: адрес, кадастровый №, разрешённое использование;
  36049 — здания ЕГРН: адрес, назначение, этажность, год постройки, материал стен, кадастровый №.
Участок запрашивается один раз (по нему размечаются все дома на нём); слой зданий — только для главного
дома участка и крупных строений. Запросы медленные, с паузой; всё кэшируется в data/nspd_raw.jsonl,
так что прерванный запуск продолжается с места остановки.

Запуск:   python nspd.py            (цели — здания в поселении из docs/data/buildings.geojson)
          python nspd.py --area core   (только граница проекта)
Результат: data/nspd_parcels.gpkg, data/nspd_buildings.gpkg → подхватываются buildings_db.py.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import geopandas as gpd
import requests
import urllib3
from shapely.geometry import shape
from shapely.strtree import STRtree

urllib3.disable_warnings()   # у nspd.gov.ru сертификат российского УЦ, которого нет в стандартном наборе

DATA = Path("data")
RAW = DATA / "nspd_raw.jsonl"
PARCELS = DATA / "nspd_parcels.gpkg"
BUILDINGS = DATA / "nspd_buildings.gpkg"
SRC = Path("docs/data/buildings.geojson")
LAYER_PARCEL, LAYER_BUILDING = 36048, 36049
MIN_AREA = 80            # м²: кроме главного дома участка, слой зданий запрашивается для строений крупнее
PAUSE = (0.8, 1.4)       # с между запросами
H = {"User-Agent": "Mozilla/5.0 (HSE study project: Kasimov heritage map)", "Referer": "https://nspd.gov.ru/map"}


def merc(lon, lat):
    return lon * 20037508.34 / 180, math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180) * 20037508.34 / 180


def query(layer, lon, lat, session):
    x, y = merc(lon, lat)
    d = 2
    url = (f"https://nspd.gov.ru/api/aeggis/v3/{layer}/wms?REQUEST=GetFeatureInfo&SERVICE=WMS&VERSION=1.3.0"
           f"&FORMAT=image/png&STYLES=&TRANSPARENT=true&LAYERS={layer}&QUERY_LAYERS={layer}"
           f"&INFO_FORMAT=application/json&FEATURE_COUNT=5&I=128&J=128&WIDTH=256&HEIGHT=256&CRS=EPSG:3857"
           f"&BBOX={x - d},{y - d},{x + d},{y + d}")
    for attempt in range(4):
        try:
            r = session.get(url, headers=H, verify=False, timeout=40)
            if r.status_code == 200:
                return r.json().get("features", [])
            print(f"  HTTP {r.status_code}, пауза…", flush=True)
        except Exception as e:
            print(f"  ошибка сети: {str(e)[:80]}, пауза…", flush=True)
        time.sleep(30 * (attempt + 1))
    return None          # сервер не отвечает — останавливаемся, кэш сохранён


def load_raw():
    done, feats = {}, {LAYER_PARCEL: {}, LAYER_BUILDING: {}}
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["layer"], r["bid"])] = True
            for f in r["features"]:
                if f.get("geometry"):
                    feats[r["layer"]][f["properties"].get("label")] = f
    return done, feats


def opt(f, k):
    v = (f["properties"].get("options") or {}).get(k)
    return None if v in ("", None) else v


def save(feats):
    """Кэш → два слоя GPKG в CRS_M (EPSG:32637)."""
    pr = [dict(cad=opt(f, "cad_num"), address=opt(f, "readable_address"),
               use=opt(f, "permitted_use_established_by_document"), category=opt(f, "land_record_category_type"),
               area=opt(f, "specified_area") or opt(f, "declared_area"), geometry=shape(f["geometry"]))
          for f in feats[LAYER_PARCEL].values()]
    br = [dict(cad=opt(f, "cad_num"), address=opt(f, "readable_address"), name=opt(f, "building_name"),
               purpose=opt(f, "purpose"), floors=opt(f, "floors"), year_built=opt(f, "year_built"),
               year_commissioning=opt(f, "year_commisioning"), materials=opt(f, "materials"),
               ownership=opt(f, "ownership_type"), heritage=opt(f, "cultural_heritage_val"),
               area=opt(f, "build_record_area"), geometry=shape(f["geometry"]))
          for f in feats[LAYER_BUILDING].values()]
    for rows, p in ((pr, PARCELS), (br, BUILDINGS)):
        if rows:
            gpd.GeoDataFrame(rows, geometry="geometry", crs=3857).to_crs(32637).to_file(p, driver="GPKG")
    print(f"  → {PARCELS} ({len(pr)}), {BUILDINGS} ({len(br)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", choices=["settlement", "core"], default="settlement")
    args = ap.parse_args()
    b = gpd.read_file(SRC)
    b = b[b["in_core" if args.area == "core" else "in_settlement"].astype(bool)].copy()
    b["pt"] = b.geometry.representative_point()
    done, feats = load_raw()
    s = requests.Session()
    log = RAW.open("a", encoding="utf-8")
    n_req = 0

    def ask(layer, row):
        nonlocal n_req
        fs = query(layer, row.pt.x, row.pt.y, s)
        if fs is None:
            save(feats); sys.exit("Сервер не отвечает — запустите позже, сбор продолжится с места остановки.")
        log.write(json.dumps({"layer": layer, "bid": row.bid, "features": fs}, ensure_ascii=False) + "\n"); log.flush()
        done[(layer, row.bid)] = True
        for f in fs:
            if f.get("geometry"):
                feats[layer][f["properties"].get("label")] = f
        n_req += 1
        if n_req % 50 == 0:
            print(f"  запросов {n_req}; участков {len(feats[LAYER_PARCEL])}, зданий ЕГРН {len(feats[LAYER_BUILDING])}", flush=True)
        if n_req % 500 == 0:
            save(feats)
        time.sleep(random.uniform(*PAUSE))

    # 1. участки: по одному запросу на участок
    to84 = lambda g: gpd.GeoSeries([g], crs=4326).to_crs(3857).iloc[0]
    print(f"Участки: {len(b)} зданий…", flush=True)
    tree_geoms = []
    for row in b.itertuples():
        if (LAYER_PARCEL, row.bid) in done:
            continue
        p3857 = to84(row.pt)
        if len(feats[LAYER_PARCEL]) != len(tree_geoms):
            tree_geoms = [shape(f["geometry"]) for f in feats[LAYER_PARCEL].values()]
            tree = STRtree(tree_geoms)
        if tree_geoms and any(tree_geoms[i].contains(p3857) for i in tree.query(p3857)):
            continue
        ask(LAYER_PARCEL, row)

    # 2. здания: главный дом участка + крупные строения
    parcels = [shape(f["geometry"]) for f in feats[LAYER_PARCEL].values()]
    b3857 = b.set_geometry(b.geometry.to_crs(3857))
    b3857["pt"] = gpd.GeoSeries(b["pt"], crs=4326).to_crs(3857)
    main_ids = set()
    for pg in parcels:
        inside = b3857[b3857["pt"].within(pg)]
        if len(inside):
            main_ids.add(inside.loc[inside.geometry.area.idxmax(), "bid"])
    targets = b[b["bid"].isin(main_ids) | (b["area_m2"].fillna(0) >= MIN_AREA)]
    print(f"Здания ЕГРН: {len(targets)} запросов максимум…", flush=True)
    for row in targets.itertuples():
        if (LAYER_BUILDING, row.bid) in done:
            continue
        p3857 = to84(row.pt)
        if any(shape(f["geometry"]).contains(p3857) for f in feats[LAYER_BUILDING].values()):
            continue
        ask(LAYER_BUILDING, row)
    log.close()
    save(feats)
    print("Готово. Дальше: python export_web.py", flush=True)


if __name__ == "__main__":
    main()
