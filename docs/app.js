// Интерактивная карта Касимова. Данные — docs/data/*.geojson, собираются export_web.py.
"use strict";

const C = {
  l1: "#1F3A5F", l2: "#B5532E", l3: "#7A1F2B",
  federal: "#2B0F14", regional: "#6E1F2B", local: "#A8434F", value: "#6B6157",
  ravine: "#6B7F4E", ens: "#8A5A44", view: "#2F5D8A", plan: "#4A4A52", prob: "#C8402F", pot: "#2E7D6B",
  bldOkn: "#5A2A30", bldVal: "#A7676F",
};
const GROUPS = {
  federal: { label: "Федерального значения", color: C.federal, r: 8 },
  regional: { label: "Регионального значения", color: C.regional, r: 7 },
  local: { label: "Местного значения", color: C.local, r: 6 },
  value: { label: "Ценный градоформирующий объект", color: C.value, r: 4.5 },
};
const FILES = ["boundaries", "order_points", "ovragi", "objects", "object_buildings", "buildings",
  "ensembles", "dominants", "squares", "streets", "views", "landscape", "verify", "projects"];
// подписи полей во всплывающих окнах и в выгрузке
const FIELD = {
  id: "№", name: "Название", category: "Категория", address: "Адрес", source: "Источник", method: "Способ привязки",
  accuracy: "Точность", in_settlement: "В поселении", in_core: "В границе проекта", levels: "Этажей", year: "Год постройки",
  src: "Источник этажности", tall: "Многоэтажное", area_ha: "Площадь, га", note: "Примечание", kind: "Тип", n: "№",
  level: "Уровень", layer: "Слой", lat: "Широта", lon: "Долгота",
  bid: "ID здания", addr_src: "Источник адреса", type: "Тип", levels_src: "Источник этажности", year_src: "Источник года",
  period: "Период", year_est: "Год (оценка)", heritage: "Объект наследия", heritage_cat: "Категория наследия",
  okn_id: "№ объекта", material: "Материал стен", condition: "Состояние", use: "Использование", notes: "Заметки",
  photo: "Фото", checked: "Проверено", info_src: "Источник ручных данных", area_m2: "Площадь застройки, м²",
  complete: "Заполненность (из 6)",
};
// карточка здания: поля по порядку и источники к ним
const BCARD = [["type"], ["levels", "levels_src"], ["year", "year_src"], ["period"], ["material"], ["condition"],
  ["use"], ["heritage"], ["heritage_cat"], ["name"], ["area_m2"], ["notes"], ["checked"], ["info_src"]];
const HIDDEN = new Set(["group"]);

const D = {};          // данные по файлам
let META = {};
const layers = {};     // id → { def, layer }

// ───────── карта ─────────
const map = L.map("map", { zoomControl: true, preferCanvas: false }).setView([54.937, 41.392], 15);
for (const [name, z] of [["areas", 405], ["lines", 420], ["bld", 430], ["pts", 450], ["marks", 460]]) {
  map.createPane(name).style.zIndex = z;
}
const base = {
  // та же OSM, приглушённая CSS-фильтром (.muted), чтобы слои читались поверх
  "Светлая": L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, className: "muted", attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>' }),
  "OpenStreetMap": L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>' }),
  "Спутник": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 19, attribution: "Esri, Maxar, Earthstar Geographics" }),
};
base["Светлая"].addTo(map);
L.control.layers(base, null, { position: "topright" }).addTo(map);
L.control.scale({ imperial: false }).addTo(map);

// ───────── утилиты ─────────
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v) => v === true ? "да" : v === false ? "нет" : v;

function popup(p, title) {
  const rows = Object.entries(p)
    .filter(([k, v]) => v !== null && v !== "" && !HIDDEN.has(k) && k !== "name")
    .map(([k, v]) => `<tr><td>${esc(FIELD[k] || k)}</td><td>${esc(fmt(v))}</td></tr>`).join("");
  return `<div class="pop"><h3>${esc(p.name || title || "")}</h3><table>${rows}</table></div>`;
}

function centroid(f) {
  const b = L.geoJSON(f).getBounds().getCenter();
  return [b.lat, b.lng];
}

function inView(f) {
  if (!$("f-view").checked) return true;
  const g = f.geometry;
  if (g.type === "Point") return map.getBounds().contains([g.coordinates[1], g.coordinates[0]]);
  return map.getBounds().intersects(L.geoJSON(f).getBounds());
}

// ───────── фильтры ─────────
const F = { groups: new Set(Object.keys(GROUPS)), acc: new Set(), area: "all", q: "", lv: 0, y0: null, y1: null,
  gap: "", type: "", color: "complete" };

function areaOk(p) {
  if (F.area === "settlement") return !!p.in_settlement;
  if (F.area === "core") return !!p.in_core;
  if (F.area === "outside") return !!p.in_settlement && !p.in_core;
  return true;
}

function objOk(f) {
  const p = f.properties;
  if (!F.groups.has(p.group) || !F.acc.has(p.accuracy)) return false;
  if (!areaOk(p)) return false;
  if (F.q) {
    const s = `${p.name} ${p.address} ${p.id}`.toLowerCase().replace(/ё/g, "е");
    if (!F.q.split(/\s+/).every((w) => s.includes(w))) return false;
  }
  return true;
}

const GAPS = {
  address: (p) => !p.address, levels: (p) => p.levels == null, year: (p) => p.year_est == null,
  sparse: (p) => (p.complete || 0) <= 1, known: (p) => !!p.address, heritage: (p) => !!p.heritage,
  manual: (p) => !!(p.info_src || p.checked || p.material || p.condition || p.notes),
};

function bldOk(f) {
  const p = f.properties;
  if (!areaOk(p)) return false;
  if (F.gap && !GAPS[F.gap](p)) return false;
  if (F.type && (p.type || "Не указан") !== F.type) return false;
  if (F.lv && !(p.levels >= F.lv)) return false;
  if ((F.y0 || F.y1) && p.year_est == null) return false;
  if (F.y0 && p.year_est < F.y0) return false;
  if (F.y1 && p.year_est > F.y1) return false;
  return true;
}

const objects = () => D.objects.features.filter(objOk);
const objIds = () => new Set(objects().map((f) => f.properties.id));
const buildings = () => D.buildings.features.filter(bldOk);

// раскраска зданий: режим → [[подпись, цвет, условие]], первый подходящий
const NODATA = "#E2DED6";
const TYPE_COLORS = { "Жилой дом (индивидуальный)": "#D9A88C", "Многоквартирный дом": "#B5532E", "Жилое здание": "#C98F6E",
  "Гараж": "#A9A9B0", "Гаражи": "#A9A9B0", "Производственное": "#6E6A80", "Склад": "#8C8899", "Торговое": "#2F5D8A",
  "Коммерческое": "#2F5D8A", "Храм": "#5A2A30", "Мечеть": "#5A2A30", "Культовое": "#5A2A30" };
const COLOR = {
  complete: [["0", NODATA, (p) => !p.complete], ["1", "#C9E2D9", (p) => p.complete === 1], ["2", "#93C7B4", (p) => p.complete === 2],
    ["3", "#5FA48E", (p) => p.complete === 3], ["4–6", "#2E7D6B", (p) => p.complete >= 4]],
  levels: [["нет данных", NODATA, (p) => p.levels == null], ["1", "#E9D9B8", (p) => p.levels <= 1], ["2", "#D9A88C", (p) => p.levels === 2],
    ["3", "#B9707A", (p) => p.levels === 3], ["4+", C.prob, (p) => p.levels >= 4]],
  year: [["нет данных", NODATA, (p) => p.year_est == null], ["до 1800", "#3B1E14", (p) => p.year_est < 1800],
    ["1800–1917", "#8A5A44", (p) => p.year_est <= 1917], ["1918–1960", "#C79A6B", (p) => p.year_est <= 1960],
    ["1961–1991", "#7FA2C1", (p) => p.year_est <= 1991], ["после 1991", "#2F5D8A", () => true]],
  type: [["не указан", NODATA, (p) => !p.type], ...Object.entries(TYPE_COLORS)
    .filter(([k], i, a) => a.findIndex(([, c]) => c === TYPE_COLORS[k]) === i)
    .map(([k, c]) => [k.replace(/ \(.*/, ""), c, (p) => TYPE_COLORS[p.type] === c]), ["прочее", "#C8C2B6", () => true]],
  src: [["OSM", "#B9707A", (p) => p.src === "OSM"], ["спутник (Microsoft ML)", "#9DB4C8", () => true]],
};
const bColor = (p) => (COLOR[F.color].find(([, , ok]) => ok(p)) || [0, NODATA])[1];

function renderLegend() {
  $("b-legend").innerHTML = COLOR[F.color].map(([t, c]) => `<span><i style="background:${c}"></i>${esc(t)}</span>`).join("");
}

function buildingPopup(p) {
  const v = (k) => p[k] != null && p[k] !== "";
  const rows = BCARD.filter(([k]) => v(k)).map(([k, s]) =>
    `<tr><td>${esc(FIELD[k] || k)}</td><td>${esc(fmt(p[k]))}${s && v(s) ? ` <span class="src">· ${esc(p[s])}</span>` : ""}</td></tr>`).join("");
  const pct = Math.round(((p.complete || 0) / 6) * 100);
  const photo = v("photo") ? `<p><a href="${esc(p.photo)}" target="_blank" rel="noopener">Фото</a></p>` : "";
  return `<div class="pop"><h3>${esc(p.address || "Адрес не известен")}</h3>
    ${v("addr_src") ? `<div class="src">адрес: ${esc(p.addr_src)}</div>` : ""}
    <div class="src">Заполнено ${p.complete || 0} из 6</div><div class="bar"><b style="width:${pct}%"></b></div>
    <table>${rows || '<tr><td colspan="2">Данных пока нет</td></tr>'}</table>${photo}
    <div class="src" style="margin-top:6px">Контур: ${esc(p.src)}</div>
    <div style="display:flex;align-items:center;margin-top:4px"><span class="bid">${esc(p.bid)}</span>
    <button class="copy" onclick="navigator.clipboard.writeText('${esc(p.bid)}');this.textContent='скопировано'">ID</button></div></div>`;
}
const bldRenderer = L.canvas({ pane: "bld", tolerance: 2 });

// ───────── слои ─────────
const fc = (src, where) => ({ type: "FeatureCollection", features: D[src].features.filter(where || (() => true)) });

function outline(src, where, style, label) {
  const data = fc(src, where);
  return L.layerGroup([
    L.geoJSON(data, { pane: "areas", interactive: false, style: { stroke: false, fillColor: style.color, fillOpacity: style.fillOpacity || 0 } }),
    L.geoJSON(data, { pane: "lines", style: { ...style, fill: false },
      onEachFeature: (f, l) => l.bindPopup(popup(f.properties, label)) }),
  ]);
}

function labelled(f, l, cls = "lbl") {
  l.bindTooltip(esc(f.properties.name), { permanent: true, direction: "center", className: cls });
}

const DEFS = [
  { group: "Границы", id: "l1", label: "Уровень 1 — округ", sw: ["dash", C.l1], on: false,
    build: () => outline("boundaries", (f) => f.properties.level === 1, { color: C.l1, weight: 2, dashArray: "6 5" }) },
  { id: "l2", label: "Уровень 2 — историческое поселение", sw: ["dash", C.l2], on: true,
    build: () => outline("boundaries", (f) => f.properties.level === 2, { color: C.l2, weight: 2.2, dashArray: "8 5" }) },
  { id: "pts", label: "Характерные точки Приказа № 2969", sw: ["dot", C.l2], on: false, src: "order_points",
    build: () => L.geoJSON(D.order_points, { pane: "pts",
      pointToLayer: (f, ll) => L.circleMarker(ll, { pane: "pts", radius: 3.5, color: "#fff", weight: 1, fillColor: C.l2, fillOpacity: 1 })
        .bindTooltip(String(f.properties.n), { direction: "top" }),
      onEachFeature: (f, l) => l.bindPopup(popup(f.properties)) }) },
  { id: "l3", label: "Уровень 3 — граница проекта", sw: ["line", C.l3], on: true,
    build: () => outline("boundaries", (f) => f.properties.level === 3, { color: C.l3, weight: 3.5, fillOpacity: 0.07 }) },
  { id: "ovragi", label: "Овраги", sw: ["fill", C.ravine], on: true, src: "ovragi",
    build: () => L.geoJSON(D.ovragi, { pane: "areas", style: { color: C.ravine, weight: 1, fillColor: C.ravine, fillOpacity: 0.4 },
      onEachFeature: (f, l) => { l.bindPopup(popup(f.properties)); labelled(f, l); } }) },

  { group: "Объекты наследия", id: "objects", label: "ОКН и ценные объекты (точки)", sw: ["dot", C.regional], on: true, dynamic: true,
    features: objects,
    build: () => L.geoJSON({ type: "FeatureCollection", features: objects() }, { pane: "pts",
      pointToLayer: (f, ll) => { const g = GROUPS[f.properties.group];
        return L.circleMarker(ll, { pane: "pts", radius: g.r, color: "#fff", weight: 1.2, fillColor: g.color, fillOpacity: 1 }); },
      onEachFeature: (f, l) => { l.bindPopup(popup(f.properties)); l.bindTooltip(esc(f.properties.name), { direction: "top" });
        f._layer = l; } }) },
  { id: "obld", label: "Здания ОКН и ценных объектов (обводка)", sw: ["line", C.bldOkn], on: true, dynamic: true,
    features: () => { const ids = objIds(); return D.object_buildings.features.filter((f) => ids.has(f.properties.id)); },
    // только обводка и без кликов — клик попадает в карточку здания под ней
    build() { return L.geoJSON({ type: "FeatureCollection", features: this.features() }, { pane: "lines", interactive: false,
      style: (f) => ({ color: f.properties.group === "value" ? C.bldVal : C.bldOkn, weight: 2.2, fill: false }) }); } },

  { group: "Здания города", id: "buildings", label: "Все здания (контуры)", sw: ["fill", "#93C7B4"], on: true, dynamic: true,
    features: buildings,
    build() { return L.geoJSON({ type: "FeatureCollection", features: this.features() }, { renderer: bldRenderer,
      style: (f) => ({ color: "#7D776C", weight: 0.6, opacity: 0.8, fillOpacity: 0.9, fillColor: bColor(f.properties) }),
      onEachFeature: (f, l) => l.bindPopup(() => buildingPopup(f.properties), { maxWidth: 320 }) }); } },

  { group: "Ценности", id: "ensembles", label: "Ансамбли и ценная среда", sw: ["fill", C.ens], on: false, src: "ensembles",
    build: () => L.geoJSON(D.ensembles, { pane: "areas", style: { color: C.ens, weight: 1.5, fillColor: C.ens, fillOpacity: 0.15 },
      onEachFeature: (f, l) => l.bindPopup(popup(f.properties)) }) },
  { id: "dominants", label: "Высотные доминанты", sw: ["tri", C.federal], on: false, src: "dominants",
    build: () => L.geoJSON(D.dominants, { pane: "marks",
      pointToLayer: (f, ll) => L.marker(ll, { pane: "marks", icon: L.divIcon({ className: "dom", html: "▲", iconSize: [18, 18] }) }),
      onEachFeature: (f, l) => { l.bindPopup(popup(f.properties)); l.bindTooltip(esc(f.properties.name), { direction: "top", offset: [0, -8] }); } }) },
  { id: "squares", label: "Исторические площади", sw: ["sq", C.plan], on: false, src: "squares",
    build: () => L.geoJSON(D.squares, { pane: "marks",
      pointToLayer: (f, ll) => L.marker(ll, { pane: "marks", icon: L.divIcon({ className: "",
        html: `<div style="width:12px;height:12px;background:#fff;border:2.5px solid ${C.plan}"></div>`, iconSize: [12, 12] }) }),
      onEachFeature: (f, l) => { l.bindPopup(popup(f.properties)); l.bindTooltip(esc(f.properties.name), { direction: "right", offset: [8, 0] }); } }) },
  { id: "streets", label: "Охраняемая планировка (улицы)", sw: ["line", C.plan], on: false, src: "streets",
    build: () => L.geoJSON(D.streets, { pane: "lines",
      style: (f) => ({ color: C.plan, weight: f.properties.kind === "Главная улица" ? 4 : 1.8, opacity: 0.85 }),
      onEachFeature: (f, l) => l.bindPopup(popup(f.properties)) }) },
  { id: "views", label: "Видовые связи", sw: ["dash", C.view], on: false, src: "views",
    build: () => L.geoJSON(D.views, { pane: "lines", style: { color: C.view, weight: 2.2, dashArray: "6 4" },
      onEachFeature: (f, l) => l.bindPopup(popup(f.properties)) }) },
  { id: "landscape", label: "Склон берега Оки", sw: ["line", C.ravine], on: false, src: "landscape",
    build: () => L.geoJSON(D.landscape, { pane: "lines", style: { color: C.ravine, weight: 6, opacity: 0.7 },
      onEachFeature: (f, l) => l.bindPopup(popup(f.properties)) }) },

  { group: "Проблемы и потенциал", id: "projects", label: "Проекты: Соборная пл., набережная, «третий луч»", sw: ["fill", C.pot], on: false, src: "projects",
    build: () => L.geoJSON(D.projects, { pane: "lines",
      style: (f) => f.geometry.type.endsWith("Polygon") ? { color: C.pot, weight: 1.5, fillColor: C.pot, fillOpacity: 0.18 }
        : f.properties.name.includes("луч") ? { color: C.pot, weight: 3, dashArray: "6 4" } : { color: "#9CCBBE", weight: 7, opacity: 0.9 },
      onEachFeature: (f, l) => l.bindPopup(popup(f.properties)) }) },
  { id: "verify", label: "Проверить на месте (1–8)", sw: ["dot", "#26262B"], on: false, src: "verify",
    build: () => L.geoJSON(D.verify, { pane: "marks",
      pointToLayer: (f, ll) => L.marker(ll, { pane: "marks", icon: L.divIcon({ className: "", html: `<div class="vnum">${f.properties.n}</div>`, iconSize: [24, 24] }) }),
      onEachFeature: (f, l) => { l.bindPopup(popup(f.properties)); l.bindTooltip(esc(f.properties.name), { direction: "top", offset: [0, -10] }); } }) },
];
for (const d of DEFS) if (!d.features) d.features = () => D[d.src || "boundaries"].features;
DEFS.find((d) => d.id === "l1").features = () => fc("boundaries", (f) => f.properties.level === 1).features;
DEFS.find((d) => d.id === "l2").features = () => fc("boundaries", (f) => f.properties.level === 2).features;
DEFS.find((d) => d.id === "l3").features = () => fc("boundaries", (f) => f.properties.level === 3).features;

function swatch([kind, color]) {
  const cls = { dash: "sw dash", line: "sw line", dot: "sw dot", fill: "sw" }[kind];
  if (kind === "tri") return `<span class="sw" style="color:${color};text-align:center;line-height:12px">▲</span>`;
  if (kind === "sq") return `<span class="sw" style="width:12px;margin:0 3px;border:2.5px solid ${color};background:#fff"></span>`;
  const st = kind === "fill" || kind === "dot" ? `background:${color}` : `border-color:${color}`;
  return `<span class="${cls}" style="${st}"></span>`;
}

function buildLayerList() {
  const box = $("layers");
  let html = "";
  for (const d of DEFS) {
    if (d.group) html += `${html ? "</div>" : ""}<div class="lgroup"><div class="lgroup-title">${d.group}</div>`;
    html += `<label class="check"><input type="checkbox" data-id="${d.id}" ${d.on ? "checked" : ""}>${swatch(d.sw)}${d.label}<span class="count" id="c-${d.id}"></span></label>`;
  }
  box.innerHTML = html + "</div>";
  box.querySelectorAll("input").forEach((el) => el.addEventListener("change", () => {
    const d = DEFS.find((x) => x.id === el.dataset.id);
    d.on = el.checked;
    render(d);
    saveState();
  }));
}

function render(d) {
  if (layers[d.id]) map.removeLayer(layers[d.id]);
  layers[d.id] = null;
  if (d.on) layers[d.id] = d.build().addTo(map);
  const c = $("c-" + d.id);
  if (c) c.textContent = d.dynamic || d.src ? d.features().length : "";
}

function refresh() {
  for (const d of DEFS) if (d.dynamic) render(d);
  const bs = buildings();
  const known = (k) => bs.filter((f) => f.properties[k] != null).length;
  $("b-count").textContent = `Зданий: ${bs.length} · с адресом ${known("address")} · этажность ${known("levels")} · год ${known("year_est")}`;
  renderList();
  exportCount();
  saveState();
}

// ───────── список ─────────
function renderList() {
  const order = Object.keys(GROUPS);
  const fs = objects().filter(inView)
    .sort((a, b) => order.indexOf(a.properties.group) - order.indexOf(b.properties.group) || a.properties.name.localeCompare(b.properties.name, "ru"));
  $("n-found").textContent = fs.length;
  $("list").innerHTML = fs.slice(0, 400).map((f, i) =>
    `<li data-i="${i}"><span class="nm">${esc(f.properties.name)}</span><span class="ad">${esc(f.properties.address)} · ${esc(GROUPS[f.properties.group].label)}</span></li>`).join("");
  $("list").querySelectorAll("li").forEach((li) => li.addEventListener("click", () => {
    const f = fs[+li.dataset.i];
    const [lng, lat] = f.geometry.coordinates;
    map.setView([lat, lng], Math.max(map.getZoom(), 17));
    if (f._layer && map.hasLayer(f._layer)) f._layer.openPopup();
    else L.popup().setLatLng([lat, lng]).setContent(popup(f.properties)).openOn(map);
    document.getElementById("panel").classList.remove("open");
  }));
}

// ───────── выгрузка ─────────
function selection() {
  const what = $("x-what").value;
  const tag = (d) => (f) => ({ ...f, properties: { layer: d.label, ...f.properties } });
  if (what === "objects") return objects().filter(inView);
  if (what === "buildings") return buildings().filter(inView);
  return DEFS.filter((d) => d.on).flatMap((d) => d.features().filter(inView).map(tag(d)));
}

function exportCount() {
  $("x-count").textContent = `Будет выгружено: ${selection().length}`;
}

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const stamp = () => new Date().toISOString().slice(0, 10);

function toCSV(fs) {
  const keys = [...new Set(fs.flatMap((f) => Object.keys(f.properties)))].filter((k) => !HIDDEN.has(k));
  const q = (v) => `"${String(fmt(v ?? "")).replace(/"/g, '""')}"`;
  const head = [...keys, "lat", "lon"].map((k) => q(FIELD[k] || k)).join(";");
  const rows = fs.map((f) => {
    const [lat, lon] = f.geometry.type === "Point" ? [f.geometry.coordinates[1], f.geometry.coordinates[0]] : centroid(f);
    return [...keys.map((k) => q(f.properties[k])), q(lat.toFixed(6)), q(lon.toFixed(6))].join(";");
  });
  return "﻿" + [head, ...rows].join("\r\n");
}

$("x-csv").addEventListener("click", () =>
  download(`kasimov_${$("x-what").value}_${stamp()}.csv`, toCSV(selection()), "text/csv;charset=utf-8"));
$("x-geojson").addEventListener("click", () =>
  download(`kasimov_${$("x-what").value}_${stamp()}.geojson`,
    JSON.stringify({ type: "FeatureCollection", features: selection().map(({ _layer, ...f }) => f) }), "application/geo+json"));

// ───────── состояние в адресе (ссылку можно переслать) ─────────
function saveState() {
  const c = map.getCenter();
  const on = DEFS.filter((d) => d.on).map((d) => d.id).join(",");
  const h = `${map.getZoom()}/${c.lat.toFixed(5)}/${c.lng.toFixed(5)}/${on}`;
  history.replaceState(null, "", "#" + h);
}

function loadState() {
  const m = location.hash.slice(1).split("/");
  if (m.length >= 3 && !isNaN(+m[0])) {
    map.setView([+m[1], +m[2]], +m[0]);
    if (m[3] !== undefined) {
      const on = new Set(m[3].split(","));
      DEFS.forEach((d) => (d.on = on.has(d.id)));
    }
    return true;
  }
  return false;
}

// ───────── элементы фильтров ─────────
function buildFilters() {
  const counts = {};
  D.objects.features.forEach((f) => (counts[f.properties.group] = (counts[f.properties.group] || 0) + 1));
  $("f-group").innerHTML = Object.entries(GROUPS).map(([k, g]) =>
    `<span class="chip on" data-g="${k}"><span class="sw dot" style="background:${g.color}"></span>${g.label.replace(" значения", "")} <span class="count">${counts[k] || 0}</span></span>`).join("");
  $("f-group").querySelectorAll(".chip").forEach((el) => el.addEventListener("click", () => {
    el.classList.toggle("on");
    F.groups[el.classList.contains("on") ? "add" : "delete"](el.dataset.g);
    refresh();
  }));

  const accs = [...new Set(D.objects.features.map((f) => f.properties.accuracy))].sort();
  accs.forEach((a) => F.acc.add(a));
  $("f-acc").innerHTML = accs.map((a) =>
    `<label class="check"><input type="checkbox" data-a="${esc(a)}" checked>${esc(a)}</label>`).join("");
  $("f-acc").querySelectorAll("input").forEach((el) => el.addEventListener("change", () => {
    F.acc[el.checked ? "add" : "delete"](el.dataset.a);
    refresh();
  }));

  $("f-area").addEventListener("change", (e) => { F.area = e.target.value; refresh(); });
  let t;
  $("q").addEventListener("input", (e) => {
    clearTimeout(t);
    t = setTimeout(() => { F.q = e.target.value.trim().toLowerCase().replace(/ё/g, "е"); refresh(); }, 150);
  });
  const lvText = () => ($("lv-v").textContent = F.lv ? F.lv : "любой");
  $("f-lv").value = F.lv; lvText();
  let tl;
  $("f-lv").addEventListener("input", (e) => {
    F.lv = +e.target.value; lvText();
    clearTimeout(tl); tl = setTimeout(refresh, 120);
  });
  const types = {};
  D.buildings.features.forEach((f) => { const t = f.properties.type || "Не указан"; types[t] = (types[t] || 0) + 1; });
  $("f-type").innerHTML += Object.entries(types).sort((a, b) => b[1] - a[1])
    .map(([t, n]) => `<option value="${esc(t)}">${esc(t)} (${n})</option>`).join("");
  $("f-type").addEventListener("change", (e) => { F.type = e.target.value; refresh(); });
  $("f-gap").addEventListener("change", (e) => { F.gap = e.target.value; refresh(); });
  $("f-color").addEventListener("change", (e) => { F.color = e.target.value; renderLegend(); render(DEFS.find((d) => d.id === "buildings")); });
  renderLegend();
  $("f-y0").addEventListener("change", (e) => { F.y0 = +e.target.value || null; refresh(); });
  $("f-y1").addEventListener("change", (e) => { F.y1 = +e.target.value || null; refresh(); });
  $("f-view").addEventListener("change", () => { renderList(); exportCount(); });
  $("x-what").addEventListener("change", exportCount);
  map.on("moveend", () => { if ($("f-view").checked) { renderList(); exportCount(); } saveState(); });
  $("toggle").addEventListener("click", () => $("panel").classList.toggle("open"));
}

// ───────── старт ─────────
(async function init() {
  const get = (n) => fetch(`data/${n}`, { cache: "no-cache" }).then((r) => r.json());
  [META] = await Promise.all([get("meta.json"), ...FILES.map(async (n) => (D[n] = await get(n + ".geojson")))]);
  $("meta").textContent = `Данные обновлены: ${META.updated} · поселение ${META.settlement_ha} га, граница проекта ${META.core_ha} га`;
  const fromHash = loadState();
  buildLayerList();
  buildFilters();
  for (const d of DEFS) if (!d.dynamic) render(d);
  if (!fromHash) map.fitBounds(L.geoJSON(fc("boundaries", (f) => f.properties.level === 2)).getBounds(), { padding: [20, 20] });
  refresh();
})();
