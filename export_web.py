#!/usr/bin/env python3
"""
Данные для интерактивной карты (docs/ → GitHub Pages).

Берёт те же источники и расчёты, что make_maps.py, и пишет слои в docs/data/*.geojson (WGS84)
+ docs/data/meta.json (дата сборки, число объектов).

Запуск:   python export_web.py
Потом:    git add -A ; git commit -m "Обновление данных" ; git push   — через минуту карта обновится.
Что править: data/objects_geocoded.csv (объекты, координаты), data/mkd_geocoded.csv (МКД),
             data/ovragi.gpkg (овраги, в QGIS), списки DOMINANTS / SQUARES / VERIFY в make_maps.py.
"""
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, mapping
from shapely.ops import linemerge, nearest_points, unary_union

import buildings_db as bdb
import make_maps as mm

WEB = Path("docs") / "data"
PREC = 6                                  # знаков после запятой: ~10 см


def cat_group(c):
    c = str(c)
    for key, g in (("Федерального", "federal"), ("Регионального", "regional"), ("Местного", "local")):
        if c.startswith(key):
            return g
    return "value"


def accuracy(method):
    """Грубая оценка точности привязки объекта — для фильтра на карте."""
    m = str(method)
    if m.startswith(("ЕГРОКН", "OSM")):
        return "точно (реестр / здание по адресу)"
    if m.startswith(("интерполяция", "соседний", "Nominatim")):
        return "приблизительно (соседние дома)"
    return "ориентировочно (по оси улицы)"


def clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if hasattr(v, "item"):                # numpy → python
        v = v.item()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def write(name, rows, simplify=0, drop_empty=False):
    """rows: dict(geometry=<shapely в CRS_M>, **свойства) → docs/data/<name>.geojson.
    drop_empty: не писать пустые свойства (экономит размер больших слоёв)."""
    rows = [r for r in rows if r["geometry"] is not None and not r["geometry"].is_empty]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=mm.CRS_M)
    if simplify:
        gdf["geometry"] = gdf.geometry.simplify(simplify)
    gdf = gdf.to_crs(4326)
    feats = []
    for _, r in gdf.iterrows():
        props = {k: clean(v) for k, v in r.items() if k != "geometry"}
        if drop_empty:
            props = {k: v for k, v in props.items() if v is not None}
        feats.append({"type": "Feature", "properties": props, "geometry": mapping(r.geometry)})
    js = json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False, separators=(",", ":"))
    # округление координат без повторного разбора: через shapely-представление уже float, режем строкой
    js = _round_coords(js)
    (WEB / f"{name}.geojson").write_text(js, encoding="utf-8")
    print(f"  {name}: {len(feats)}")
    return len(feats)


def _round_coords(js):
    import re
    return re.sub(r"(-?\d+\.\d{%d})\d+" % PREC, r"\1", js)


def lines(g):
    if g is None or g.is_empty:
        return None
    return linemerge(g) if g.geom_type == "MultiLineString" else g


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    d = mm.load_all()
    print("Границы…")
    poly, order_pts = mm.order_boundary(d)
    okn = mm.okn_points(d, None)
    okn = okn[okn.geometry.within(poly.buffer(300))]
    r = mm.historic_core(d, poly, order_pts)
    core = r["core"]
    ctx = mm.heritage_context(d, okn, poly, r["ravines"])
    counts = {}

    print("Слои → docs/data/")
    # 1. Границы
    counts["boundaries"] = write("boundaries", [
        dict(level=1, name="Касимовский муниципальный округ", note="Уровень 1 — исследовательский масштаб",
             area_ha=round(d["okrug"].geometry.iloc[0].area / 1e4), geometry=d["okrug"].geometry.iloc[0]),
        dict(level=2, name="Историческое поселение «Касимов»",
             note="Уровень 2 — Приказ Минкультуры России № 2969 от 07.12.2015 (прил. 1, 3)",
             area_ha=round(poly.area / 1e4), geometry=poly),
        dict(level=3, name="Граница проекта (между оврагами)",
             note="Уровень 3 — запад: Никольский овраг; восток: Успенский и Ямской; юго-запад: Ока; "
                  "север: ул. Академика Уткина → Карла Маркса → Карла Либкнехта → Советская",
             area_ha=round(core.area / 1e4), geometry=core),
    ], simplify=1)
    counts["order_points"] = write("order_points", [
        dict(n=int(n), name=f"Точка {int(n)}", note="Характерная точка границы, Приказ № 2969, прил. 1", geometry=g)
        for n, g in zip(order_pts["n"], order_pts.geometry)])
    counts["ovragi"] = write("ovragi", [
        dict(name=n, source=s, area_ha=round(g.area / 1e4, 1), geometry=g)
        for n, s, g in zip(r["ravines"]["name"], r["ravines"]["source"], r["ravines"].geometry)], simplify=1)

    # 2. ОКН и ценные объекты (точки) + их здания
    rows = []
    for _, o in okn.iterrows():
        g = o.geometry
        rows.append(dict(id=str(o["n"]), name=o["name"], address=o["address"], category=o["category"],
                         group=cat_group(o["category"]), source=o["source"], method=o["method"],
                         accuracy=accuracy(o["method"]), in_settlement=bool(poly.contains(g)),
                         in_core=bool(core.contains(g)), geometry=g))
    counts["objects"] = write("objects", rows)
    b = d["buildings"][["geometry"]]
    pts = okn[["geometry", "n", "name", "category"]].rename(columns={"name": "obj_name"})
    j = gpd.sjoin(b, pts, predicate="contains", how="inner")
    j = j[~j.index.duplicated()]
    counts["object_buildings"] = write("object_buildings", [
        dict(id=str(x["n"]), name=x["obj_name"], group=cat_group(x["category"]), geometry=x.geometry)
        for _, x in j.iterrows()], simplify=0.5)

    # 3. Все здания города (buildings_db.py)
    print("База зданий…")
    bd = bdb.buildings(d, poly, core, okn)
    keep = ["bid", "address", "addr_src", "type", "name", "levels", "levels_src", "year", "year_src", "period",
            "year_est", "heritage", "heritage_cat", "okn_id", "material", "condition", "use", "notes", "photo",
            "checked", "info_src", "cad_num", "egrn_purpose", "parcel_cad", "parcel_use",
            "src", "area_m2", "complete", "in_settlement", "in_core"]
    counts["buildings"] = write("buildings", [dict(geometry=g, **{k: x[k] for k in keep})
                                              for g, (_, x) in zip(bd.geometry, bd.iterrows())],
                                simplify=0.4, drop_empty=True)
    bl = mm.mkd_buildings(d)
    bl = bl[bl["levels"].notna()]

    # 4. Ценности
    counts["ensembles"] = write("ensembles", [
        dict(name=n.replace("\n", " "), geometry=z) for n, z in ctx["zones"].items()], simplify=1)
    counts["dominants"] = write("dominants", [
        dict(name=t.replace("\n", " "), kind="Высотная доминанта", geometry=p) for t, p in ctx["dominants"]])
    counts["squares"] = write("squares", [
        dict(name=t, kind="Историческая площадь", geometry=p) for t, p in ctx["squares"]])
    st = []
    for keys, kind in ((mm.MAIN_STREETS, "Главная улица"), (mm.PROTECTED_STREETS, "Охраняемая улица")):
        for k in keys:
            g = lines(mm.edges_named(d, [k], poly))
            if g is not None:
                st.append(dict(name=k, kind=kind, geometry=g))
    counts["streets"] = write("streets", st, simplify=1)
    counts["views"] = write("views", [
        dict(name=f"Вид: {'устье Бабёнки' if i == 2 else 'с Оки'} → {next(t for t, q in ctx['dominants'] if q.equals(b_))}".replace("\n", " "),
             geometry=LineString([(a.x, a.y), (b_.x, b_.y)])) for i, (a, b_) in enumerate(mm.view_links(ctx))])
    counts["landscape"] = write("landscape", [
        dict(name="Склон левого берега Оки", kind="Ландшафт", geometry=lines(ctx["shore"]))], simplify=1)

    # 5. Проблемы и потенциал
    tall = bl[bl["levels"] >= mm.TALL_LEVELS]
    tall = tall[tall.intersects(poly)]
    pos = mm.verify_points(d, okn, r, ctx, tall)
    counts["verify"] = write("verify", [
        dict(n=n, name=mm.VERIFY[n].replace("\n", " "), geometry=p) for n, p in pos.items()])
    pr = []
    sob = ctx["zones"].get("Ансамбль\nСоборной площади")
    if sob is not None:
        pr.append(dict(name="«Слияние времён»: благоустройство Соборной пл.",
                       note="Победитель VII Всероссийского конкурса (2022), бюджет 2023 г. — 89 млн ₽", geometry=sob))
    nab = mm.edges_named(d, ["Набережная"], poly.buffer(30))
    nab = lines(unary_union(nab)) if not nab.is_empty else None
    if nab is not None:
        pr.append(dict(name="Проект набережной", note="В общей программе развития (~2 млрд ₽)", geometry=nab))
        if ctx["sob_c"] is not None:
            q = nearest_points(ctx["sob_c"], nab)[1]
            pr.append(dict(name="«Третий луч»: Соборная пл. → Набережная", note="Связь центра с рекой",
                           geometry=LineString([(ctx["sob_c"].x, ctx["sob_c"].y), (q.x, q.y)])))
    counts["projects"] = write("projects", pr, simplify=1)

    meta = dict(updated=datetime.now().strftime("%Y-%m-%d %H:%M"), counts=counts,
                core_ha=round(core.area / 1e4), settlement_ha=round(poly.area / 1e4), tall_levels=mm.TALL_LEVELS)
    (WEB / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Готово:", WEB)


if __name__ == "__main__":
    main()
