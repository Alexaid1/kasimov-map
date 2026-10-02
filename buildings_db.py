"""
База зданий г. Касимова: контур каждого дома + всё, что о нём известно.

Контуры — Overture Maps (OSM + Microsoft ML Buildings по спутниковым снимкам), кэш data/overture_buildings.parquet.
ID здания (bid) — постоянный идентификатор Overture (GERS), по нему ведётся ручная таблица.

Что подтягивается автоматически:
  OSM            — адрес, тип, название, этажность (здания + адресные точки data/addr_points.gpkg);
  реестр МКД     — этажность и год (data/mkd_geocoded.csv, через make_maps.mkd_buildings);
  Приказ / ЕГРОКН — объект наследия в здании, период постройки из названия («Дом Барковых, 1828 год»).
Ручные дополнения — data/buildings_info.csv (разделитель «;», открывается в Excel):
  bid;address;type;levels;year;period;material;condition;use;notes;photo;source;checked
  Заполненное поле перекрывает автоматическое значение.
"""
import re

import geopandas as gpd
import numpy as np
import pandas as pd

import make_maps as mm

OVERTURE = mm.DATA / "overture_buildings.parquet"
ADDR_POINTS = mm.DATA / "addr_points.gpkg"
INFO = mm.DATA / "buildings_info.csv"
INFO_COLS = ["bid", "address", "type", "levels", "year", "period", "material", "condition", "use", "notes",
             "photo", "source", "checked"]
# поля, по которым считается заполненность карточки дома
COMPLETE = ["address", "type", "levels", "year_est", "material", "condition"]

TYPES = {"house": "Жилой дом (индивидуальный)", "detached": "Жилой дом (индивидуальный)",
         "apartments": "Многоквартирный дом", "residential": "Жилое здание", "dormitory": "Общежитие",
         "garage": "Гараж", "garages": "Гаражи", "shed": "Сарай", "hut": "Хозпостройка", "barn": "Хозпостройка",
         "industrial": "Производственное", "warehouse": "Склад", "service": "Служебное",
         "retail": "Торговое", "commercial": "Коммерческое", "kiosk": "Киоск", "supermarket": "Торговое",
         "office": "Офисное", "school": "Школа", "kindergarten": "Детский сад", "college": "Учебное",
         "hospital": "Больница", "public": "Общественное", "civic": "Общественное", "government": "Административное",
         "church": "Храм", "chapel": "Часовня", "mosque": "Мечеть", "religious": "Культовое",
         "ruins": "Руины", "construction": "Строится", "train_station": "Вокзал", "transportation": "Транспортное",
         "greenhouse": "Теплица", "roof": "Навес", "hotel": "Гостиница"}
ROMAN = {"XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20}


def overture(d):
    if not OVERTURE.exists():
        import overturemaps.core as oc
        print("  контуры зданий Overture (один раз → data/overture_buildings.parquet)…")
        b = d["city"].to_crs(4326).buffer(0.01).total_bounds
        oc.geodataframe("building", bbox=tuple(b)).to_parquet(OVERTURE)
    o = gpd.read_parquet(OVERTURE)
    if o.crs is None:
        o = o.set_crs(4326)
    o = o.to_crs(mm.CRS_M)
    src = o["sources"].map(lambda s: s[0] if len(s) else {})
    o["src"] = src.map(lambda s: "OSM" if s.get("dataset") == "OpenStreetMap" else "спутник (Microsoft ML)")
    o["osm"] = src.map(lambda s: (s.get("record_id") or "").split("@")[0] if s.get("dataset") == "OpenStreetMap" else None)
    return o[["id", "src", "osm", "geometry"]].rename(columns={"id": "bid"})


def addr_points(d):
    """Адресные точки OSM (узлы с addr:housenumber, не на контуре здания)."""
    def fetch():
        import osmnx as ox
        ll = gpd.GeoSeries([d["city"].geometry.iloc[0].buffer(500)], crs=mm.CRS_M).to_crs(4326).iloc[0]
        g = ox.features_from_polygon(ll, {"addr:housenumber": True}).to_crs(mm.CRS_M)
        g = g[g.geometry.geom_type == "Point"]
        return g[[c for c in ("addr:street", "addr:housenumber") if c in g] + ["geometry"]]
    return mm.cached("addr_points", fetch)


def period_of(name):
    """«Дом Барковых, 1828 год» → (1828, '1828 год'); «…, XVIII век» → (None, 'XVIII век')."""
    tail = str(name).split(",")[-1].strip()
    m = re.search(r"\b(1[5-9]\d\d)\b", tail)
    if m:
        return int(m.group(1)), tail
    if re.search(r"\b[XVI]+\b.*(век|в\.)|(век|в\.)", tail):
        return None, tail
    return None, None


def year_est(year, period):
    """Год для раскраски и фильтра: точный, иначе середина века из периода."""
    if pd.notna(year):
        return int(year)
    m = re.findall(r"\b(XV|XVI|XVII|XVIII|XIX|XX)\b", str(period or ""))
    if m:
        c = ROMAN[m[-1]]
        p = str(period).lower()
        off = 15 if re.search(r"нача|перв|1-й", p) else 85 if re.search(r"коне|втор|2-й", p) else 50
        return (c - 1) * 100 + off
    return None


ADDR_TYPES = [(r"улица|ул\.?", "ул."), (r"переулок|пер\.?", "пер."), (r"площадь|пл\.?", "пл."),
              (r"проезд|пр-д\.?", "пр."), (r"тупик|туп\.?", "туп.")]   # «Набережная» в Касимове — название улицы


def norm_addr(a):
    """Единый вид адреса: «улица Набережная, дом 12.» / «ул Набережная, д 12» → «ул. Набережная, 12»."""
    if a is None or (isinstance(a, float) and pd.isna(a)) or not str(a).strip():
        return None
    s = re.sub(r"\s+", " ", str(a)).strip(" .,")
    for pat, rep in ADDR_TYPES:
        s = re.sub(rf"(?<![А-Яа-яЁё]){pat}(?![А-Яа-яЁё])", rep, s, flags=re.I)
    s = re.sub(r",?\s*(?:дом|д\.?)\s*(?=\d)", ", ", s, flags=re.I)
    s = re.sub(r"\s*,\s*", ", ", s).replace("..", ".")
    s = re.sub(r"\.\s+(?=\d)", ", ", s) if "," not in s else s
    return s.strip(" ,")


def short_addr(a):
    """«Российская Федерация, Рязанская область, м.о. Касимовский, г. Касимов, ул. Начальная, д. 10» → «ул. Начальная, д. 10»."""
    if not a:
        return None
    a = re.split(r"(?:г\.?|город)\s*Касимов,?", str(a))[-1].strip(" ,")
    return a or None


def egrn(b, rp):
    """ЕГРН (nspd.py): здание — назначение, этажность, год, материал, адрес; участок — адрес и использование."""
    if mm.DATA.joinpath("nspd_buildings.gpkg").exists():
        e = gpd.read_file(mm.DATA / "nspd_buildings.gpkg")
        j = gpd.sjoin(rp, e, predicate="within", how="inner")
        j = j[~j.index.duplicated()]
        b.loc[j.index, "cad_num"] = j["cad"]
        b.loc[j.index, "egrn_purpose"] = j["purpose"].fillna(j["name"])
        lv = pd.to_numeric(j["floors"], errors="coerce")
        b.loc[lv.dropna().index, "levels"] = lv.dropna()
        b.loc[lv.dropna().index, "levels_src"] = "ЕГРН"
        yr = pd.to_numeric(j["year_built"].fillna(j["year_commissioning"]), errors="coerce")
        yr = yr[(yr > 1500) & (yr < 2100)]
        b.loc[yr.index, "year"] = yr
        b.loc[yr.index, "year_src"] = "ЕГРН"
        b.loc[j.index, "material"] = j["materials"]
        ad = j["address"].map(short_addr).dropna()
        b.loc[ad.index, "address"] = ad
        b.loc[ad.index, "addr_src"] = "ЕГРН: здание"
        tp = b.loc[j.index, "type"].isna()
        b.loc[tp[tp].index, "type"] = j.loc[tp[tp].index, "purpose"]
        print(f"  ЕГРН, здания: {len(j)}")
    if mm.DATA.joinpath("nspd_parcels.gpkg").exists():
        p = gpd.read_file(mm.DATA / "nspd_parcels.gpkg")
        j = gpd.sjoin(rp, p, predicate="within", how="inner")
        j = j[~j.index.duplicated()]
        b.loc[j.index, "parcel_cad"] = j["cad"]
        b.loc[j.index, "parcel_use"] = j["use"]
        ad = j["address"].map(short_addr).dropna()
        ad = ad[b.loc[ad.index, "address"].isna()]
        b.loc[ad.index, "address"] = ad
        b.loc[ad.index, "addr_src"] = "ЕГРН: земельный участок"
        print(f"  ЕГРН, участки: зданий на участках {len(j)}, адрес по участку {len(ad)}")


def load_info():
    if not INFO.exists():
        pd.DataFrame(columns=INFO_COLS).to_csv(INFO, sep=";", index=False, encoding="utf-8-sig")
    t = pd.read_csv(INFO, sep=";", dtype=str, encoding="utf-8-sig").fillna("")
    t = t[t["bid"].str.strip() != ""]
    return t.drop_duplicates("bid", keep="last").set_index("bid")


def buildings(d, settlement, core, okn):
    """Все здания города с атрибутами (CRS_M)."""
    city = d["city"].geometry.iloc[0]
    b = overture(d)
    b = b[b.intersects(city)].reset_index(drop=True)
    rp = gpd.GeoDataFrame(geometry=b.representative_point(), crs=mm.CRS_M)

    # OSM: адрес, тип, название, этажность + реестр МКД (этажность, год) — по точке внутри здания OSM
    ob = mm.mkd_buildings(d)
    ob = ob.assign(geometry=ob.representative_point())
    j = gpd.sjoin(ob, b[["geometry"]], predicate="within", how="inner")
    j = j[~j["index_right"].duplicated()].set_index("index_right")
    hn, st = j["addr:housenumber"], j["addr:street"]
    b["address"] = (st + ", " + hn).where(st.notna() & hn.notna())
    b["addr_src"] = pd.Series("OSM", index=j.index).where(b["address"].notna())
    b["type"] = j["building"].map(lambda t: TYPES.get(t) if t != "yes" else None)
    b["name"] = j["name"]
    b["levels"] = j["levels"]
    b["levels_src"] = j["src"].where(j["levels"].notna())
    b["year"] = j["year"]
    b["year_src"] = pd.Series("реестр МКД", index=j.index).where(j["year"].notna())

    # адресные точки OSM для зданий без адреса
    ap = addr_points(d)
    if len(ap):
        ja = gpd.sjoin(ap, b[["geometry"]], predicate="within", how="inner")
        ja = ja[~ja["index_right"].duplicated()].set_index("index_right")
        a = ja["addr:street"] + ", " + ja["addr:housenumber"]
        need = b["address"].isna() & b.index.isin(a.index)
        b.loc[need, "address"] = a[b.index[need]]
        b.loc[need, "addr_src"] = "OSM: адресная точка"

    # объекты наследия: точка объекта внутри здания
    oj = gpd.sjoin(okn[["geometry", "n", "name", "category", "address"]].rename(
        columns={"name": "o_name", "address": "o_addr"}), b[["geometry"]], predicate="within", how="inner")
    oj = oj.sort_values("category").drop_duplicates("index_right").set_index("index_right")
    b["okn_id"] = oj["n"].astype(str)
    b["heritage"] = oj["o_name"]
    b["heritage_cat"] = oj["category"]
    per = oj["o_name"].map(period_of)
    b["period"] = per.map(lambda p: p[1])
    py = per.map(lambda p: p[0])
    fill = b["year"].isna() & py.reindex(b.index).notna()
    b.loc[fill, "year"] = py[b.index[fill]]
    b.loc[fill, "year_src"] = "Приказ № 2969 / ЕГРОКН"
    need = b["address"].isna() & b.index.isin(oj.index)
    b.loc[need, "address"] = oj["o_addr"].reindex(b.index[need]).values
    b.loc[need, "addr_src"] = "Приказ № 2969 / ЕГРОКН"

    for c in ("material", "condition", "use", "notes", "photo", "checked", "info_src", "cad_num", "egrn_purpose",
              "parcel_cad", "parcel_use"):
        b[c] = None
    egrn(b, rp)

    # ручная таблица перекрывает всё
    info = load_info()
    if len(info):
        m = b["bid"].isin(info.index)
        print(f"  ручная таблица: {len(info)} строк, найдено зданий: {m.sum()}")
        for c in INFO_COLS[1:]:
            col = "info_src" if c == "source" else c
            v = b.loc[m, "bid"].map(info[c]).replace("", np.nan)
            if c in ("levels", "year"):
                v = pd.to_numeric(v, errors="coerce")
            b.loc[m & v.notna().reindex(b.index, fill_value=False), col] = v.dropna()
            if c in ("address", "levels", "year"):
                b.loc[m & v.notna().reindex(b.index, fill_value=False), f"{'addr' if c == 'address' else c}_src"] = "ручной ввод"

    b["address"] = b["address"].map(norm_addr)
    b["year_est"] = [year_est(y, p) for y, p in zip(b["year"], b["period"])]
    b["complete"] = b[COMPLETE].notna().sum(axis=1)
    b["area_m2"] = b.area.round()
    b["in_settlement"] = rp.within(settlement).values
    b["in_core"] = rp.within(core).values
    print(f"  зданий: {len(b)} (OSM {(b['src'] == 'OSM').sum()}, спутник {(b['src'] != 'OSM').sum()}); "
          f"с адресом {b['address'].notna().sum()}, этажность {b['levels'].notna().sum()}, "
          f"год {b['year_est'].notna().sum()}, тип {b['type'].notna().sum()}")
    return b
