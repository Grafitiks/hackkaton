// GEO-VEGA // Панель мониторинга вегетационной динамики и детекции аномалий (Космохакатон)

let map, drawnItems, drawControl, activeDrawHandler = null;
let esriSatelliteLayer, osmLayer;
let osmFieldLayers = {};
let osmFieldsData = {}; // Словарь найденных на карте контуров OSM: polyId -> { id, name, geojson, areaHa, centerLat, centerLon, layer, isOsm: true }
let userSavedFields = {}; // Словарь полей текущей группы: id -> { id, name, color, geojson, areaHa, centerLat, centerLon, layer, data }
let customFieldGroups = {}; // Словарь групп полей: groupId -> { id, name, desc, createdAt, fields: {} }
let activeGroupId = null; // Идентификатор текущей выбранной группы полей или null
let selectedStartDate = "01.01.2026";
let selectedEndDate = "05.09.2026";
let fieldCounter = 1;

// Получение активного объекта поля (из полей группы или из найденных контуров OSM)
function getActiveFieldObject() {
  if (!selectedFieldId) return null;
  return userSavedFields[selectedFieldId] || osmFieldsData[selectedFieldId] || null;
}

// ============================================================================
// УТИЛИТЫ ФОРМАТИРОВАНИЯ ДАТ (ДД.ММ.ГГГГ <-> ГГГГ-ММ-ДД ISO)
// Обеспечивают единый российский агрономический формат интерфейса и совместимость с API
// ============================================================================

/** Преобразование даты из формата ДД.ММ.ГГГГ в канонический ISO ГГГГ-ММ-ДД */
function formatDateToIso(dateStr) {
  if (!dateStr || typeof dateStr !== 'string') return dateStr;
  const str = dateStr.trim();
  const ruMatch = str.match(/^(\d{2})\.(\d{2})\.(\d{4})$/);
  if (ruMatch) {
    return `${ruMatch[3]}-${ruMatch[2]}-${ruMatch[1]}`;
  }
  return str;
}

/** Преобразование даты из канонического ISO ГГГГ-ММ-ДД в формат ДД.ММ.ГГГГ */
function formatDateToRu(dateStr) {
  if (!dateStr || typeof dateStr !== 'string') return dateStr;
  const str = dateStr.trim();
  const isoMatch = str.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (isoMatch) {
    return `${isoMatch[3]}.${isoMatch[2]}.${isoMatch[1]}`;
  }
  return str;
}

let ndviChartInstance = null;
let weatherChartInstance = null;
let currentTimeseriesData = null;
let currentActiveAnalysis = null; // Хранит последнее выполненное исследование поля (field, data)
let isSyncingScales = false;
let isDrawingActive = false;

// Регистрация плагина масштабирования Chart.js при наличии
if (typeof Chart !== 'undefined') {
  try {
    if (typeof ChartZoom !== 'undefined') {
      Chart.register(ChartZoom);
    } else if (typeof window['chartjs-plugin-zoom'] !== 'undefined') {
      Chart.register(window['chartjs-plugin-zoom']);
    }
  } catch (e) {
    console.debug("ChartZoom registration:", e);
  }
}

// Состояние модального окна настройки имени и цвета поля
let currentModalContext = null;
let currentModalColor = "#00f0ff";

// Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED) и ERA5 обеспечивают глобальное покрытие
function isInsideAgroZone(lat, lon) {
  return lat >= -60.0 && lat <= 85.0 && lon >= -180.0 && lon <= 180.0;
}

document.addEventListener("DOMContentLoaded", () => {
  try { localStorage.removeItem("geovega_saved_user_fields"); } catch (e) {}
  initMap();
  initEventHandlers();
  initDropdowns();
  initRegionsModule();
  initFieldModal();
  initGroupModal();
  initOsmFieldModal();
  loadGroupsFromStorage();
  initPassportModal();
});

// 1. Инициализация интерактивной карты Leaflet
function initMap() {
  // Центрирование карты на ключевом аграрном регионе Поволжья (Самарская область)
  map = L.map("map", {
    center: [53.25, 50.25],
    zoom: 10,
    zoomControl: true
  });

  // Базовые картографические слои (спутниковый ESRI и картосхема OSM)
  esriSatelliteLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles &copy; Esri, Maxar, Earthstar Geographics",
    maxZoom: 18
  }).addTo(map);

  osmLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19
  });

  // Слой для отрисованных пользователем контуров полей
  drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  // Элементы управления рисованием контуров Leaflet Draw
  drawControl = new L.Control.Draw({
    draw: {
      polygon: {
        allowIntersection: false,
        showArea: true,
        shapeOptions: {
          color: "#00f0ff",
          weight: 3,
          fillColor: "#00f0ff",
          fillOpacity: 0.35
        }
      },
      polyline: false,
      circle: false,
      rectangle: {
        shapeOptions: {
          color: "#00f0ff",
          weight: 3,
          fillColor: "#00f0ff",
          fillOpacity: 0.35
        }
      },
      circlemarker: false,
      marker: false
    },
    edit: {
      featureGroup: drawnItems
    }
  });
  map.addControl(drawControl);

  // Обработчики событий рисования контуров
  map.on(L.Draw.Event.DRAWSTART, () => {
    setDrawingMode(true);
  });

  map.on(L.Draw.Event.DRAWSTOP, () => {
    setDrawingMode(false);
  });

  // Обработка завершения рисования пользовательского полигона
  map.on(L.Draw.Event.CREATED, (event) => {
    const layer = event.layer;
    const geojson = layer.toGeoJSON();
    
    const coords = geojson.geometry.coordinates[0];
    let sumLat = 0, sumLon = 0;
    coords.forEach(pt => { sumLon += pt[0]; sumLat += pt[1]; });
    const centerLat = sumLat / coords.length;
    const centerLon = sumLon / coords.length;
    const areaHa = calculatePolygonAreaHa(coords);

    // Ограничение: максимум 10 000 га для контура одного поля
    const MAX_FIELD_AREA_HA = 10000;
    if (areaHa > MAX_FIELD_AREA_HA) {
      drawnItems.removeLayer(layer);
      showToast(`Выделена слишком масштабная область (${Math.round(areaHa).toLocaleString()} га)! Для агрономического анализа выделите контур поля или массива до ${MAX_FIELD_AREA_HA.toLocaleString()} га.`, true);
      setDrawingMode(false);
      return;
    }

    drawnItems.addLayer(layer);
    setDrawingMode(false);

    // Открытие модального окна для ввода названия и выбора цвета поля
    openFieldModal({
      isNew: true,
      name: `Поле #${fieldCounter} (${areaHa.toFixed(1)} га)`,
      defaultName: `Поле #${fieldCounter} (${areaHa.toFixed(1)} га)`,
      color: "#00f0ff",
      areaHa: areaHa,
      centerLat: centerLat,
      centerLon: centerLon,
      layer: layer,
      geojson: geojson
    });
  });

  // Отслеживание добавления вершин для обновления плавающей панели
  map.on(L.Draw.Event.DRAWVERTEX, () => {
    updateDrawToolbarState();
  });
  map.on("click", () => {
    if (isDrawingActive) {
      setTimeout(updateDrawToolbarState, 40);
    }
  });

  // Отслеживание курсора в режиме интерактивного рисования
  const cursorTooltip = document.getElementById("cursorGeoTooltip");
  map.on("mousemove", (e) => {
    if (!isDrawingActive) {
      cursorTooltip.classList.add("hidden");
      return;
    }

    cursorTooltip.style.left = `${e.originalEvent.clientX}px`;
    cursorTooltip.style.top = `${e.originalEvent.clientY}px`;
    cursorTooltip.classList.remove("hidden");

    const mapContainer = document.getElementById("map");
    mapContainer.classList.remove("cursor-forbidden");
    cursorTooltip.className = "cursor-geo-tooltip allowed";
    cursorTooltip.innerHTML = `<i class="fa-solid fa-satellite"></i> <span>Спутниковое покрытие GEE активно (${e.latlng.lat.toFixed(3)}°N, ${e.latlng.lng.toFixed(3)}°E)</span>`;
  });

  map.on("mouseout", () => {
    cursorTooltip.classList.add("hidden");
  });
}

function setDrawingMode(active) {
  isDrawingActive = active;
  const guideBanner = document.getElementById("drawingGuideBanner");
  const drawToolbar = document.getElementById("drawToolbar");
  const cursorTooltip = document.getElementById("cursorGeoTooltip");
  const mapEl = document.getElementById("map");

  if (active) {
    document.body.classList.add("drawing-active-mode");
    if (guideBanner) guideBanner.classList.remove("hidden");
    if (drawToolbar) drawToolbar.classList.remove("hidden");
    mapEl.style.cursor = "crosshair";
    map.doubleClickZoom.disable();
    updateDrawToolbarState();
  } else {
    document.body.classList.remove("drawing-active-mode");
    if (guideBanner) guideBanner.classList.add("hidden");
    if (drawToolbar) drawToolbar.classList.add("hidden");
    if (cursorTooltip) cursorTooltip.classList.add("hidden");
    mapEl.classList.remove("cursor-forbidden");
    mapEl.style.cursor = "";
    map.doubleClickZoom.enable();
    if (activeDrawHandler) {
      try { activeDrawHandler.disable(); } catch (e) {}
      activeDrawHandler = null;
    }
  }
}

function startDrawingField() {
  // Проверка: выбрана ли папка (группа) полей перед началом рисования
  if (!activeGroupId || !customFieldGroups[activeGroupId]) {
    const groupKeys = Object.keys(customFieldGroups);
    if (groupKeys.length === 0) {
      showToast("Предупреждение: создайте папку перед рисованием контура поля!", true);
      const createGroupBtn = document.getElementById("createGroupBtn");
      if (createGroupBtn) createGroupBtn.click();
    } else {
      showToast("Предупреждение: создайте папку или выберите существующую из списка!", true);
      const regionSelect = document.getElementById("regionSelect");
      if (regionSelect) {
        regionSelect.focus();
      }
    }
    return;
  }

  // Автоматическое приближение карты до уровня полей (зум 12), если масштаб слишком общий
  if (map.getZoom() < 11) {
    showToast("🔍 Карта автоматически приближена (зум 12) для четкой видимости границ поля и защиты от захвата лишних земель.");
    map.setZoom(12, { animate: true });
  }

  if (activeDrawHandler) {
    try { activeDrawHandler.disable(); } catch (e) {}
  }

  map.doubleClickZoom.disable();
  activeDrawHandler = new L.Draw.Polygon(map, drawControl.options.draw.polygon);
  activeDrawHandler.enable();
  setDrawingMode(true);
}

function updateDrawToolbarState() {
  if (!activeDrawHandler) return;
  const markers = activeDrawHandler._markers || [];
  const count = markers.length;

  const pointsEl = document.getElementById("drawPointsCount");
  const areaEl = document.getElementById("drawAreaCount");
  const areaBadge = document.getElementById("drawAreaBadge");
  const finishBtn = document.getElementById("finishDrawBtn");
  const undoBtn = document.getElementById("undoDrawBtn");

  if (pointsEl) pointsEl.textContent = count;
  if (undoBtn) undoBtn.disabled = (count <= 1);

  if (count >= 3) {
    const coords = markers.map(m => {
      const ll = m.getLatLng();
      return [ll.lng, ll.lat];
    });
    coords.push(coords[0]); // Замыкание контура полигона для геометрического расчета площади
    const areaHa = calculatePolygonAreaHa(coords);

    if (areaEl) areaEl.textContent = `${areaHa.toFixed(1)} га`;

    if (areaHa > 10000) {
      if (areaBadge) areaBadge.classList.add("overlimit");
      if (finishBtn) {
        finishBtn.disabled = true;
        finishBtn.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Превышен лимит (>10 000 га)`;
        finishBtn.title = "Поле слишком велико (>10 000 га). Уменьшите контур или приблизьте карту к конкретному участку.";
      }
    } else {
      if (areaBadge) areaBadge.classList.remove("overlimit");
      if (finishBtn) {
        finishBtn.disabled = false;
        finishBtn.innerHTML = `<i class="fa-solid fa-check"></i> Завершить <span class="hotkey-hint">Enter</span>`;
        finishBtn.title = "Завершить контур поля и запустить спутниковый анализ GEE";
      }
    }
  } else {
    if (areaEl) areaEl.textContent = "-- га";
    if (areaBadge) areaBadge.classList.remove("overlimit");
    if (finishBtn) {
      finishBtn.disabled = true;
      finishBtn.innerHTML = `<i class="fa-solid fa-check"></i> Завершить (${count}/3 точек)`;
      finishBtn.title = "Поставьте минимум 3 точки на карте, чтобы замкнуть контур";
    }
  }
}

// ============================================================================
// ВЫБОР И АНАЛИЗ НАЙДЕННОГО ПОЛЯ OSM БЕЗ ДОБАВЛЕНИЯ В ГРУППУ
// Поля группы содержат только поля, сохраненные пользователем в созданные папки
// ============================================================================

function selectAndAnalyzeOsmField(polyId) {
  const osmField = osmFieldsData[polyId];
  if (!osmField) return;

  selectedFieldId = polyId;

  // Селектор «Поля группы» не пополняется чужими полями OSM
  const select = document.getElementById("polygonSelect");
  if (select) {
    select.value = "";
  }

  // Кнопки редактирования и удаления поля группы отключаются для несохраненного поля OSM
  const editFieldBtn = document.getElementById("editFieldBtn");
  if (editFieldBtn) editFieldBtn.disabled = true;
  const deleteFieldBtn = document.getElementById("deleteFieldBtn");
  if (deleteFieldBtn) deleteFieldBtn.disabled = true;

  // Снимаем подсветку со всех полей группы
  Object.keys(userSavedFields).forEach(id => {
    const f = userSavedFields[id];
    if (f && f.layer && typeof f.layer.setStyle === 'function') {
      f.layer.setStyle({
        color: f.color || "#00f0ff",
        fillColor: f.color || "#00f0ff",
        weight: 2.5,
        fillOpacity: 0.25,
        className: "verified-field-path"
      });
      if (f.layer._path) {
        f.layer._path.classList.remove("cinematic-field-selected");
      }
    }
  });

  // Обновляем подсветку найденных слоев OSM (активный слой подсвечивается)
  Object.keys(osmFieldLayers).forEach(id => {
    const l = osmFieldLayers[id];
    if (!l) return;
    const isThis = (id === polyId);
    if (typeof l.setStyle === 'function') {
      l.setStyle({
        color: isThis ? "#34d399" : "#10b981",
        fillColor: isThis ? "#34d399" : "#10b981",
        weight: isThis ? 4.5 : 2,
        fillOpacity: isThis ? 0.45 : 0.28,
        className: isThis ? "verified-field-path cinematic-field-selected" : "verified-field-path"
      });
    }
    if (l._path) {
      if (isThis) {
        l._path.classList.add("cinematic-field-selected");
      } else {
        l._path.classList.remove("cinematic-field-selected");
      }
    }
    if (isThis && typeof l.bringToFront === 'function') {
      l.bringToFront();
    }
  });

  // Кинематографический перелет камеры к выбранному контуру OSM
  if (osmField.layer && typeof osmField.layer.getBounds === 'function' && osmField.layer.getBounds().isValid()) {
    const bounds = osmField.layer.getBounds();
    let optimalZoom = 14;
    try {
      optimalZoom = Math.min(Math.max(map.getBoundsZoom(bounds, false, [75, 75]), 13), 16);
    } catch (e) {
      optimalZoom = 14;
    }
    if (typeof map.flyTo === 'function') {
      map.flyTo(bounds.getCenter(), optimalZoom, { animate: true, duration: 1.2 });
    } else {
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
    }
  } else if (osmField.centerLat && osmField.centerLon) {
    map.flyTo([osmField.centerLat, osmField.centerLon], 14, { animate: true, duration: 1.2 });
  }

  updateCoordinatesDisplay(osmField.centerLat, osmField.centerLon, osmField.areaHa);
  showToast(`Поле выбрано: «${osmField.name}» (нажмите ПКМ для сохранения в группу)`);

  // Запуск комплексного спутникового анализа контура
  analyzeCustomPolygon(osmField);
}

// 2. Обработчики событий интерфейса
function initEventHandlers() {
  // Переключение базовых слоев (Спутник / Схема)
  document.getElementById("layerSatBtn").addEventListener("click", () => {
    map.removeLayer(osmLayer);
    map.addLayer(esriSatelliteLayer);
    document.getElementById("layerSatBtn").classList.add("active");
    document.getElementById("layerOsmBtn").classList.remove("active");
  });

  document.getElementById("layerOsmBtn").addEventListener("click", () => {
    map.removeLayer(esriSatelliteLayer);
    map.addLayer(osmLayer);
    document.getElementById("layerOsmBtn").classList.add("active");
    document.getElementById("layerSatBtn").classList.remove("active");
  });

  // Кнопка очистки найденных полей OSM
  const clearOsmBtn = document.getElementById("clearOsmBtn");
  if (clearOsmBtn) {
    clearOsmBtn.addEventListener("click", () => {
      const count = Object.keys(osmFieldLayers).length;
      Object.keys(osmFieldLayers).forEach(polyId => {
        if (!userSavedFields[polyId]) {
          map.removeLayer(osmFieldLayers[polyId]);
        }
      });
      osmFieldLayers = {};
      osmFieldsData = {};
      if (selectedFieldId && !userSavedFields[selectedFieldId]) {
        selectedFieldId = null;
      }
      clearOsmBtn.classList.add("hidden");
      showToast(`Слой найденных полей OSM убран с карты (${count} объектов)`);
    });
  }

  // Кнопка динамического поиска полей через OpenStreetMap Overpass API
  const fetchOsmBtn = document.getElementById("fetchOsmBtn");
  if (fetchOsmBtn) {
    fetchOsmBtn.addEventListener("click", async () => {
      if (map.getZoom() < 10) {
        showToast("Пожалуйста, приблизьте карту ближе к аграрному району (зум 10+), чтобы загрузить контуры из OSM.", true);
        return;
      }

      const bounds = map.getBounds();
      const minLon = bounds.getWest();
      const minLat = bounds.getSouth();
      const maxLon = bounds.getEast();
      const maxLat = bounds.getNorth();
      
      fetchOsmBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Поиск полей...`;
      showToast("Поиск реальных контуров полей через OpenStreetMap Overpass API...");
      
      try {
        const resp = await fetch(`/api/osm-farmlands?min_lon=${minLon}&min_lat=${minLat}&max_lon=${maxLon}&max_lat=${maxLat}&limit=20`);
        const data = await resp.json();
        const features = data.features || [];
        
        if (features.length === 0) {
          showToast("В текущем масштабе не найдено с/х полей. Приблизьте карту к аграрным территориям.", true);
        } else {
          features.forEach(f => {
            const polyId = f.properties.anon_polygon_id;
            if (osmFieldLayers[polyId]) return; // Поле уже добавлено на карту
            
            const coords = f.geometry.coordinates[0];
            let sumLat = 0, sumLon = 0;
            coords.forEach(pt => { sumLon += pt[0]; sumLat += pt[1]; });
            const cLat = sumLat / coords.length;
            const cLon = sumLon / coords.length;
            const areaHa = f.properties.area_ha || calculatePolygonAreaHa(coords);

            const layer = L.geoJSON(f, {
              style: () => ({
                color: "#10b981",
                weight: 2,
                fillColor: "#10b981",
                fillOpacity: 0.30,
                className: "verified-field-path"
              })
            }).addTo(map);
            
            osmFieldLayers[polyId] = layer;
            
            const shortTitle = (f.properties.name && !f.properties.name.startsWith("Поле OSM"))
              ? `${f.properties.name}`
              : `Поле #${Object.keys(osmFieldLayers).length}`;

            layer.bindPopup(`
              <div style="font-family: sans-serif; font-size: 13px; color: #111;">
                <strong>${f.properties.name}</strong><br>
                Культура: <b>${f.properties.crop_type}</b><br>
                Площадь: <b>${areaHa} га</b><br>
                Источник: <b>OpenStreetMap</b><br>
                <div style="margin-top: 6px; font-size: 11.5px; color: #475569; line-height: 1.4;">
                  <span><b>ЛКМ</b> — выбрать и анализировать поле</span><br>
                  <span><b>ПКМ</b> — добавить в папку / группу</span>
                </div>
                <div style="margin-top: 8px;">
                  <button type="button" class="btn-osm-popup-action" style="background: #0284c7; color: #fff; border: none; padding: 5px 10px; border-radius: 4px; font-size: 12px; cursor: pointer; display: inline-flex; align-items: center; gap: 5px; font-weight: 600;">
                    <i class="fa-solid fa-folder-plus"></i> Добавить в группу
                  </button>
                </div>
              </div>
            `);

            osmFieldsData[polyId] = {
              id: polyId,
              name: shortTitle,
              color: "#10b981",
              geojson: f,
              areaHa: areaHa,
              centerLat: cLat,
              centerLon: cLon,
              layer: layer,
              isOsm: true
            };

            // Обработка клика левой кнопкой мыши (ЛКМ) — автономный спутниковый анализ без добавления в группу
            layer.on("click", () => {
              selectAndAnalyzeOsmField(polyId);
            });

            // Обработка клика правой кнопкой мыши (ПКМ) — открытие диалога добавления в группу / создания новой папки
            layer.on("contextmenu", (e) => {
              if (e && e.originalEvent) {
                e.originalEvent.preventDefault();
                e.originalEvent.stopPropagation();
              }
              if (map) map.closePopup();
              try { layer.closePopup(); } catch (err) {}

              openOsmAddToGroupModal({
                polyId,
                name: shortTitle,
                cropType: f.properties.crop_type || "зерновые",
                areaHa,
                centerLat: cLat,
                centerLon: cLon,
                geojson: f,
                layer
              });
            });

            // Обработка клика по кнопке добавления внутри всплывающего окна
            layer.on("popupopen", () => {
              const popupEl = layer.getPopup() ? layer.getPopup().getElement() : null;
              if (popupEl) {
                const btn = popupEl.querySelector(".btn-osm-popup-action");
                if (btn) {
                  btn.onclick = (e) => {
                    if (e) e.stopPropagation();
                    if (map) map.closePopup();
                    try { layer.closePopup(); } catch (err) {}
                    openOsmAddToGroupModal({
                      polyId,
                      name: shortTitle,
                      cropType: f.properties.crop_type || "зерновые",
                      areaHa,
                      centerLat: cLat,
                      centerLon: cLon,
                      geojson: f,
                      layer
                    });
                  };
                }
              }
            });
          });
          
          if (clearOsmBtn) clearOsmBtn.classList.remove("hidden");
          showToast(`Найдено ${features.length} полей из OpenStreetMap! ЛКМ — выбор и анализ поля, ПКМ — добавить в папку.`);
        }
      } catch (err) {
        console.error("Ошибка запроса OSM:", err);
        showToast("Ошибка при поиске полей в OpenStreetMap", true);
      } finally {
        fetchOsmBtn.innerHTML = `<i class="fa-solid fa-satellite"></i> Найти поля (OSM)`;
      }
    });
  }

  // Выбор поля из выпадающего списка
  document.getElementById("polygonSelect").addEventListener("change", (e) => {
    const polyId = e.target.value;
    if (polyId && userSavedFields[polyId]) {
      activateAndAnalyzeField(polyId);
    }
  });

  // Кнопка редактирования параметров поля
  const editFieldBtn = document.getElementById("editFieldBtn");
  if (editFieldBtn) {
    editFieldBtn.addEventListener("click", () => {
      if (selectedFieldId && userSavedFields[selectedFieldId]) {
        const f = userSavedFields[selectedFieldId];
        openFieldModal({
          isNew: false,
          id: f.id,
          name: f.name,
          defaultName: f.name,
          color: f.color || "#00f0ff",
          areaHa: f.areaHa,
          centerLat: f.centerLat,
          centerLon: f.centerLon,
          layer: f.layer,
          geojson: f.geojson
        });
      }
    });
  }

  // Кнопка удаления выбранного поля
  const deleteFieldBtn = document.getElementById("deleteFieldBtn");
  if (deleteFieldBtn) {
    deleteFieldBtn.addEventListener("click", () => {
      if (selectedFieldId) {
        deleteSelectedField(selectedFieldId);
      }
    });
  }

  // Кнопка удаления выбранной группы полей
  const deleteGroupBtn = document.getElementById("deleteGroupBtn");
  if (deleteGroupBtn) {
    deleteGroupBtn.addEventListener("click", () => {
      deleteCurrentGroup();
    });
  }

  // Изменение сельскохозяйственной культуры
  const cropSelectEl = document.getElementById("cropSelect");
  if (cropSelectEl) {
    cropSelectEl.addEventListener("change", () => {
      const activeField = getActiveFieldObject();
      if (activeField) {
        analyzeCustomPolygon(activeField);
      }
    });
  }

  // Кнопка включения режима рисования в шапке
  const drawModeBtn = document.getElementById("drawModeBtn");
  if (drawModeBtn) {
    drawModeBtn.addEventListener("click", () => {
      startDrawingField();
    });
  }

  // Кнопки управления в плавающей панели рисования
  const finishBtn = document.getElementById("finishDrawBtn");
  if (finishBtn) {
    finishBtn.addEventListener("click", () => {
      if (activeDrawHandler && activeDrawHandler._markers && activeDrawHandler._markers.length >= 3) {
        activeDrawHandler.completeShape();
      }
    });
  }

  const undoBtn = document.getElementById("undoDrawBtn");
  if (undoBtn) {
    undoBtn.addEventListener("click", () => {
      if (activeDrawHandler && activeDrawHandler._markers && activeDrawHandler._markers.length > 0) {
        activeDrawHandler.deleteLastVertex();
        updateDrawToolbarState();
      }
    });
  }

  const cancelBtn = document.getElementById("cancelDrawBtn");
  if (cancelBtn) {
    cancelBtn.addEventListener("click", () => {
      setDrawingMode(false);
    });
  }

  // Глобальные горячие клавиши в режиме рисования
  document.addEventListener("keydown", (e) => {
    if (!isDrawingActive) return;

    if (e.key === "Enter") {
      e.preventDefault();
      const fBtn = document.getElementById("finishDrawBtn");
      if (fBtn && !fBtn.disabled && activeDrawHandler && activeDrawHandler._markers && activeDrawHandler._markers.length >= 3) {
        activeDrawHandler.completeShape();
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      setDrawingMode(false);
    } else if (e.key === "Backspace" || (e.ctrlKey && e.key === "z")) {
      e.preventDefault();
      if (activeDrawHandler && activeDrawHandler._markers && activeDrawHandler._markers.length > 0) {
        activeDrawHandler.deleteLastVertex();
        updateDrawToolbarState();
      }
    }
  });

  // Управление масштабированием графиков Chart.js
  const zoomInBtn = document.getElementById("chartZoomInBtn");
  if (zoomInBtn) {
    zoomInBtn.addEventListener("click", () => {
      if (ndviChartInstance && typeof ndviChartInstance.zoom === 'function') {
        ndviChartInstance.zoom(1.25);
        syncChartScales(ndviChartInstance);
      }
    });
  }

  const zoomOutBtn = document.getElementById("chartZoomOutBtn");
  if (zoomOutBtn) {
    zoomOutBtn.addEventListener("click", () => {
      if (ndviChartInstance && typeof ndviChartInstance.zoom === 'function') {
        ndviChartInstance.zoom(0.8);
        syncChartScales(ndviChartInstance);
      }
    });
  }

  const resetZoomBtn = document.getElementById("chartResetZoomBtn");
  if (resetZoomBtn) {
    resetZoomBtn.addEventListener("click", () => {
      resetBothChartsZoom();
    });
  }

  // Двойной клик по графику для сброса масштаба к исходному диапазону
  const ndviCanvas = document.getElementById("ndviChart");
  if (ndviCanvas) {
    ndviCanvas.addEventListener("dblclick", () => {
      resetBothChartsZoom();
    });
  }

  const weatherCanvas = document.getElementById("weatherChart");
  if (weatherCanvas) {
    weatherCanvas.addEventListener("dblclick", () => {
      resetBothChartsZoom();
    });
  }
}

// 3. Инициализация диапазона дат и элементов управления
async function initDropdowns() {
  const startInput = document.getElementById("startDateInput");
  const endInput = document.getElementById("endDateInput");
  const applyBtn = document.getElementById("applyDateBtn");
  const cropSelect = document.getElementById("cropSelect");
  const polySelect = document.getElementById("polygonSelect");
  const group = document.querySelector(".date-range-group");

  const todayIso = "2026-09-05";
  const minArchiveIso = "2014-01-01";
  const todayRu = "05.09.2026";
  const minArchiveRu = "01.01.2014";

  // Проверка физического существования даты в реальном календаре (ДД.ММ.ГГГГ и ГГГГ-ММ-ДД)
  function isValidCalendarDate(str) {
    if (!str || typeof str !== 'string') return false;
    const trimmed = str.trim();

    // Формат ДД.ММ.ГГГГ (основной российский стандарт)
    const ruMatch = trimmed.match(/^(\d{2})\.(\d{2})\.(\d{4})$/);
    if (ruMatch) {
      const d = parseInt(ruMatch[1], 10);
      const mon = parseInt(ruMatch[2], 10);
      const y = parseInt(ruMatch[3], 10);
      if (mon < 1 || mon > 12) return false;
      if (d < 1 || d > 31) return false;
      const dt = new Date(y, mon - 1, d);
      return dt.getFullYear() === y && (dt.getMonth() + 1) === mon && dt.getDate() === d;
    }

    // Формат ГГГГ-ММ-ДД (ISO)
    const isoMatch = trimmed.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (isoMatch) {
      const y = parseInt(isoMatch[1], 10);
      const mon = parseInt(isoMatch[2], 10);
      const d = parseInt(isoMatch[3], 10);
      if (mon < 1 || mon > 12) return false;
      if (d < 1 || d > 31) return false;
      const dt = new Date(y, mon - 1, d);
      return dt.getFullYear() === y && (dt.getMonth() + 1) === mon && dt.getDate() === d;
    }

    return false;
  }

  // Строгая валидация и ограничение интервалов дат с защитой от несуществующих периодов
  function validateAndSyncDateInputs(triggerAlert = false) {
    if (!startInput || !endInput) return false;

    const sVal = startInput.value.trim();
    const eVal = endInput.value.trim();

    const sValid = isValidCalendarDate(sVal);
    const eValid = isValidCalendarDate(eVal);

    startInput.classList.toggle("invalid-date", !sValid);
    endInput.classList.toggle("invalid-date", !eValid);

    if (!sValid || !eValid) {
      if (group) group.classList.add("invalid");
      if (applyBtn) applyBtn.disabled = true;
      if (triggerAlert) {
        showToast("Указана несуществующая календарная дата. Проверьте число и месяц (формат: ДД.ММ.ГГГГ).", true);
      }
      return false;
    }

    let curStartIso = formatDateToIso(sVal);
    let curEndIso = formatDateToIso(eVal);

    // 1. Ограничение снизу: запуск космических архивов ДЗЗ (01.01.2014)
    if (curStartIso < minArchiveIso) {
      curStartIso = minArchiveIso;
      if (triggerAlert) showToast(`Спутниковые архивы доступны с 01.01.2014.`, true);
    }
    if (curEndIso < minArchiveIso) {
      curEndIso = minArchiveIso;
    }

    // 2. Ограничение сверху: дата не может быть из будущего (сегодня: 05.09.2026)
    if (curStartIso > todayIso) {
      curStartIso = todayIso;
      if (triggerAlert) showToast(`Начальная дата не может быть в будущем (сегодня: 05.09.2026).`, true);
    }
    if (curEndIso > todayIso) {
      curEndIso = todayIso;
      if (triggerAlert) showToast(`Конечная дата не может быть в будущем (сегодня: 05.09.2026).`, true);
    }

    // 3. Защита от хронологически инвертированных периодов (начало > конец)
    if (curStartIso > curEndIso) {
      if (triggerAlert) {
        showToast("Несуществующий период: начальная дата не может быть позже конечной.", true);
      }
      curEndIso = curStartIso;
    }

    const curStartRu = formatDateToRu(curStartIso);
    const curEndRu = formatDateToRu(curEndIso);

    startInput.value = curStartRu;
    endInput.value = curEndRu;

    selectedStartDate = curStartRu;
    selectedEndDate = curEndRu;

    if (startPicker) {
      startPicker.setDate(curStartRu, false);
      startPicker.set("maxDate", curEndRu);
    }
    if (endPicker) {
      endPicker.setDate(curEndRu, false);
      endPicker.set("minDate", curStartRu);
    }

    if (group) group.classList.remove("invalid");
    if (applyBtn) applyBtn.disabled = false;
    startInput.classList.remove("invalid-date");
    endInput.classList.remove("invalid-date");

    return true;
  }

  function handleValidDateApplied() {
    const activeField = getActiveFieldObject();
    if (activeField) {
      showToast(`Обновление спутникового анализа за период: ${selectedStartDate} .. ${selectedEndDate}`);
      analyzeCustomPolygon(activeField);
    }
  }

  let startPicker = null;
  let endPicker = null;

  // Инициализация единого кибер-агрономического календаря Flatpickr в формате ДД.ММ.ГГГГ
  if (typeof flatpickr !== 'undefined') {
    startPicker = flatpickr("#startDateInput", {
      locale: "ru",
      dateFormat: "d.m.Y",
      defaultDate: selectedStartDate,
      minDate: minArchiveRu,
      maxDate: selectedEndDate,
      allowInput: true, // Разрешает прямой ввод даты с клавиатуры в формате ДД.ММ.ГГГГ
      clickOpens: true,
      disableMobile: true,
      onChange: function(selectedDates, dateStr) {
        if (!dateStr) return;
        selectedStartDate = dateStr;
        if (endPicker) {
          endPicker.set("minDate", dateStr);
        }
        validateAndSyncDateInputs(false);
        handleValidDateApplied();
      },
      onClose: function() {
        validateAndSyncDateInputs(false);
      }
    });

    endPicker = flatpickr("#endDateInput", {
      locale: "ru",
      dateFormat: "d.m.Y",
      defaultDate: selectedEndDate,
      minDate: selectedStartDate,
      maxDate: todayRu,
      allowInput: true, // Разрешает прямой ввод даты с клавиатуры в формате ДД.ММ.ГГГГ
      clickOpens: true,
      disableMobile: true,
      onChange: function(selectedDates, dateStr) {
        if (!dateStr) return;
        selectedEndDate = dateStr;
        if (startPicker) {
          startPicker.set("maxDate", dateStr);
        }
        validateAndSyncDateInputs(false);
        handleValidDateApplied();
      },
      onClose: function() {
        validateAndSyncDateInputs(false);
      }
    });
  }

  // Обработка ручного ввода дат с клавиатуры (ввод текста, Enter и смена фокуса)
  function handleManualDateInput(inputEl, pickerInstance, isStart) {
    if (!inputEl) return;

    inputEl.addEventListener("input", () => {
      const val = inputEl.value.trim();
      // Если введен полный формат даты (10 символов: ДД.ММ.ГГГГ или ГГГГ-ММ-ДД)
      if (val.length === 10) {
        if (isValidCalendarDate(val)) {
          const ruVal = formatDateToRu(val);
          inputEl.value = ruVal;
          if (pickerInstance) {
            pickerInstance.setDate(ruVal, false);
          }
          validateAndSyncDateInputs(false);
        } else {
          inputEl.classList.add("invalid-date");
        }
      }
    });

    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        const val = inputEl.value.trim();
        if (isValidCalendarDate(val)) {
          inputEl.value = formatDateToRu(val);
        }
        if (validateAndSyncDateInputs(true)) {
          if (pickerInstance) {
            pickerInstance.setDate(inputEl.value.trim(), false);
          }
          handleValidDateApplied();
          inputEl.blur();
        }
      }
    });

    inputEl.addEventListener("change", () => {
      const val = inputEl.value.trim();
      if (isValidCalendarDate(val)) {
        inputEl.value = formatDateToRu(val);
      }
      if (validateAndSyncDateInputs(true)) {
        if (pickerInstance) {
          pickerInstance.setDate(inputEl.value.trim(), false);
        }
        handleValidDateApplied();
      }
    });
  }

  handleManualDateInput(startInput, startPicker, true);
  handleManualDateInput(endInput, endPicker, false);

  if (applyBtn) {
    applyBtn.addEventListener("click", () => {
      if (!validateAndSyncDateInputs(true)) return;
      handleValidDateApplied();
      if (!selectedFieldId) {
        showToast(`Период установлен: ${selectedStartDate} .. ${selectedEndDate}. Выберите или нарисуйте поле на карте.`);
      }
    });
  }

  if (cropSelect) {
    cropSelect.addEventListener("change", () => {
      const activeField = getActiveFieldObject();
      if (activeField) {
        analyzeCustomPolygon(activeField);
      }
    });
  }

  if (polySelect) {
    polySelect.addEventListener("change", (e) => {
      const fieldId = e.target.value;
      if (fieldId && userSavedFields[fieldId]) {
        activateAndAnalyzeField(fieldId);
      }
    });
    polySelect.innerHTML = `<option value="" disabled selected>— Нет активной группы (выберите папку или добавьте поле через ПКМ) —</option>`;
  }
}

// ============================================================================
// 3.1 МОДУЛЬ РЕГИОНАЛЬНОГО АГРОМОНИТОРИНГА
// Критерий: Адаптивность под множественные регионы и автопоиск контуров полей
// ============================================================================

const DEFAULT_REGIONS = {
  samara: { id: "samara", name: "Самарская область (Поволжье)", macro_region: "Среднее Поволжье", climate_zone: "Лесостепная / Степная зона", dominant_crops: ["яровая пшеница", "подсолнечник", "ячмень"], center_lat: 53.25, center_lon: 50.35, zoom: 11, bbox: [50.1, 53.1, 50.6, 53.4] },
  krasnodar: { id: "krasnodar", name: "Краснодарский край (Кубань)", macro_region: "Южный ФО / Прикубанская равнина", climate_zone: "Умеренно-теплый", dominant_crops: ["озимая пшеница", "кукуруза", "подсолнечник", "соя"], center_lat: 45.20, center_lon: 39.10, zoom: 11, bbox: [38.85, 45.05, 39.35, 45.35] },
  rostov: { id: "rostov", name: "Ростовская область (Дон)", macro_region: "Южный ФО / Нижний Дон", climate_zone: "Умеренно-засушливая степь", dominant_crops: ["озимая пшеница", "подсолнечник", "зернобобовые"], center_lat: 47.35, center_lon: 39.90, zoom: 11, bbox: [39.65, 47.2, 40.15, 47.5] },
  voronezh: { id: "voronezh", name: "Воронежская область (Черноземье)", macro_region: "Центрально-Черноземный район", climate_zone: "Типичная лесостепь (черноземы)", dominant_crops: ["сахарная свекла", "озимая пшеница", "подсолнечник"], center_lat: 51.50, center_lon: 39.40, zoom: 11, bbox: [39.15, 51.35, 39.65, 51.65] },
  stavropol: { id: "stavropol", name: "Ставропольский край (Кавказ)", macro_region: "Северо-Кавказский ФО", climate_zone: "Засушливая и умеренная степь", dominant_crops: ["озимая пшеница", "горох", "рапс"], center_lat: 45.10, center_lon: 42.10, zoom: 11, bbox: [41.85, 44.95, 42.35, 45.25] },
  altay: { id: "altay", name: "Алтайский край (Сибирь)", macro_region: "Западная Сибирь", climate_zone: "Резко континентальный", dominant_crops: ["яровая пшеница", "гречиха", "овес"], center_lat: 52.80, center_lon: 83.20, zoom: 11, bbox: [82.95, 52.65, 83.45, 52.95] },
  tatarstan: { id: "tatarstan", name: "Республика Татарстан", macro_region: "Среднее Поволжье", climate_zone: "Умеренно-континентальный лесостепной", dominant_crops: ["яровая пшеница", "рожь", "рапс"], center_lat: 55.65, center_lon: 49.30, zoom: 11, bbox: [49.05, 55.5, 49.55, 55.8] },
  belgorod: { id: "belgorod", name: "Белгородская область", macro_region: "Центрально-Черноземный район", climate_zone: "Лесостепная зона высокой продуктивности", dominant_crops: ["соя", "кукуруза на зерно", "озимая пшеница"], center_lat: 50.60, center_lon: 36.80, zoom: 11, bbox: [36.55, 50.45, 37.05, 50.75] },
  saratov: { id: "saratov", name: "Саратовская область", macro_region: "Нижнее Поволжье", climate_zone: "Засушливая степь", dominant_crops: ["твердая пшеница", "подсолнечник", "просо"], center_lat: 51.60, center_lon: 46.40, zoom: 11, bbox: [46.15, 51.45, 46.65, 51.75] },
  orenburg: { id: "orenburg", name: "Оренбургская область", macro_region: "Южный Урал / Степь", climate_zone: "Сухостепная", dominant_crops: ["яровая твердая пшеница", "подсолнечник"], center_lat: 51.85, center_lon: 55.30, zoom: 11, bbox: [55.05, 51.7, 55.55, 52.0] }
};

let currentRegionId = null;
let regionsCatalog = { ...DEFAULT_REGIONS };

// ============================================================================
// УПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЬСКИМИ ГРУППАМИ ПОЛЕЙ (ХОЗЯЙСТВА / КЛАСТЕРЫ В «РЕГИОНЕ»)
// ============================================================================

function getRussianFieldCountWord(count) {
  const abs = Math.abs(count) % 100;
  const last = abs % 10;
  if (abs > 10 && abs < 20) return "полей";
  if (last > 1 && last < 5) return "поля";
  if (last === 1) return "поле";
  return "полей";
}

function saveGroupsToStorage() {
  try {
    const toSave = {};
    Object.keys(customFieldGroups).forEach(gId => {
      const g = customFieldGroups[gId];
      const cleanFields = {};
      Object.keys(g.fields || {}).forEach(fId => {
        const f = g.fields[fId];
        cleanFields[fId] = {
          id: f.id,
          name: f.name,
          color: f.color || "#00f0ff",
          geojson: f.geojson,
          areaHa: f.areaHa,
          centerLat: f.centerLat,
          centerLon: f.centerLon
        };
      });
      toSave[gId] = {
        id: g.id,
        name: g.name,
        desc: g.desc || "",
        createdAt: g.createdAt,
        fields: cleanFields
      };
    });
    localStorage.setItem("geovega_custom_field_groups", JSON.stringify(toSave));
  } catch (e) {
    console.error("Ошибка сохранения групп полей в localStorage:", e);
  }
}

function loadGroupsFromStorage() {
  try {
    const stored = localStorage.getItem("geovega_custom_field_groups");
    if (stored) {
      customFieldGroups = JSON.parse(stored) || {};
    }
  } catch (e) {
    console.error("Ошибка загрузки групп полей из localStorage:", e);
    customFieldGroups = {};
  }
  renderGroupsInRegionSelect();
}

function renderGroupsInRegionSelect() {
  const optgroup = document.getElementById("customGroupsOptgroup");
  if (!optgroup) return;

  optgroup.innerHTML = "";
  const groupKeys = Object.keys(customFieldGroups);

  if (groupKeys.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.disabled = true;
    opt.textContent = "— Нет созданных групп —";
    optgroup.appendChild(opt);
    return;
  }

  groupKeys.forEach(gId => {
    const grp = customFieldGroups[gId];
    const count = Object.keys(grp.fields || {}).length;
    const opt = document.createElement("option");
    opt.value = grp.id;
    opt.textContent = `📁 ${grp.name} (${count} ${getRussianFieldCountWord(count)})`;
    optgroup.appendChild(opt);
  });

  const regionSelect = document.getElementById("regionSelect");
  if (regionSelect && activeGroupId && customFieldGroups[activeGroupId]) {
    regionSelect.value = activeGroupId;
  }
}

// Удаление активной группы полей
function deleteCurrentGroup() {
  if (!activeGroupId || !customFieldGroups[activeGroupId]) {
    showToast("Пожалуйста, сначала выберите группу полей из списка!", true);
    return;
  }

  const group = customFieldGroups[activeGroupId];
  const groupName = group.name;
  const count = Object.keys(group.fields || {}).length;

  if (!confirm(`Вы уверены, что хотите удалить группу «${groupName}» и все входящие в неё поля (${count} ${getRussianFieldCountWord(count)})?`)) {
    return;
  }

  // 1. Удаляем все слои полей этой группы с карты
  clearPreviousRegionFields(groupName);

  // 2. Удаляем группу из коллекции и обновляем хранилище
  delete customFieldGroups[activeGroupId];
  activeGroupId = null;
  currentRegionId = null;
  saveGroupsToStorage();
  renderGroupsInRegionSelect();

  // 3. Сбрасываем селекторы
  const regionSelect = document.getElementById("regionSelect");
  if (regionSelect) regionSelect.value = "";

  const polygonSelect = document.getElementById("polygonSelect");
  if (polygonSelect) {
    polygonSelect.innerHTML = '<option value="" disabled selected>— Нет полей (выберите группу или нарисуйте на карте) —</option>';
  }

  const labelEl = document.getElementById("polygonSelectLabel");
  if (labelEl) labelEl.innerHTML = '<i class="fa-solid fa-map-pin"></i> Поля группы:';

  const deleteGroupBtn = document.getElementById("deleteGroupBtn");
  if (deleteGroupBtn) deleteGroupBtn.disabled = true;
  const editGroupBtn = document.getElementById("editGroupBtn");
  if (editGroupBtn) editGroupBtn.disabled = true;
  const editFieldBtn = document.getElementById("editFieldBtn");
  if (editFieldBtn) editFieldBtn.disabled = true;
  const deleteFieldBtn = document.getElementById("deleteFieldBtn");
  if (deleteFieldBtn) deleteFieldBtn.disabled = true;

  showToast(`Группа полей «${groupName}» успешно удалена.`);
}

// Удаление выбранного поля
function deleteSelectedField(fieldId) {
  const targetId = fieldId || selectedFieldId;
  if (!targetId) {
    showToast("Поле для удаления не выбрано!", true);
    return;
  }

  const field = userSavedFields[targetId];
  const fieldName = field ? field.name : "выбранное поле";

  if (!confirm(`Вы действительно хотите удалить поле «${fieldName}»?`)) {
    return;
  }

  // 1. Удаляем слой с карты и из drawnItems
  if (field && field.layer) {
    try {
      if (drawnItems && drawnItems.hasLayer(field.layer)) {
        drawnItems.removeLayer(field.layer);
      }
      map.removeLayer(field.layer);
    } catch (e) {}
  }

  // 2. Удаляем из рабочей коллекции userSavedFields
  delete userSavedFields[targetId];

  // 3. Если привязано к группе, удаляем из группы
  if (activeGroupId && customFieldGroups[activeGroupId] && customFieldGroups[activeGroupId].fields) {
    delete customFieldGroups[activeGroupId].fields[targetId];
    saveGroupsToStorage();
    renderGroupsInRegionSelect();
  }

  // 4. Удаляем пункт из выпадающего списка
  const select = document.getElementById("polygonSelect");
  if (select) {
    const opt = select.querySelector(`option[value="${targetId}"]`);
    if (opt) opt.remove();
  }

  closeFieldModal();

  // 5. Проверяем оставшиеся поля
  const remainingKeys = Object.keys(userSavedFields);
  if (remainingKeys.length > 0) {
    activateAndAnalyzeField(remainingKeys[0]);
    showToast(`Поле «${fieldName}» удалено. Активировано поле «${userSavedFields[remainingKeys[0]].name}».`);
  } else {
    selectedFieldId = null;
    if (select) {
      select.innerHTML = (activeGroupId && customFieldGroups[activeGroupId])
        ? `<option value="" disabled selected>— В группе «${customFieldGroups[activeGroupId].name}» нет полей (нарисуйте на карте) —</option>`
        : '<option value="" disabled selected>— Нет полей (выберите группу или нарисуйте на карте) —</option>';
    }
    const editFieldBtn = document.getElementById("editFieldBtn");
    if (editFieldBtn) editFieldBtn.disabled = true;
    const deleteFieldBtn = document.getElementById("deleteFieldBtn");
    if (deleteFieldBtn) deleteFieldBtn.disabled = true;

    // Сброс индикаторов KPI
    const statusEl = document.getElementById("statusText");
    const beaconEl = document.getElementById("statusIndicator");
    const ndviEl = document.getElementById("currentNdvi");
    const cropSub = document.getElementById("cropTypeSub");
    const zscoreEl = document.getElementById("currentZscore");
    const gapsEl = document.getElementById("gapsCount");
    const areaSub = document.getElementById("areaSub");

    if (statusEl) statusEl.textContent = "Ожидание выбора поля";
    if (beaconEl) beaconEl.className = "status-beacon normal";
    if (ndviEl) ndviEl.textContent = "--";
    if (cropSub) cropSub.textContent = "Культура: --";
    if (zscoreEl) zscoreEl.textContent = "-- σ";
    if (gapsEl) gapsEl.textContent = "--";
    if (areaSub) areaSub.textContent = "Площадь: -- га";

    showToast(`Поле «${fieldName}» удалено.`);
  }
}

let currentEditingGroupId = null;

function initGroupModal() {
  const createGroupBtn = document.getElementById("createGroupBtn");
  const editGroupBtn = document.getElementById("editGroupBtn");
  const groupModal = document.getElementById("groupModal");
  const closeGroupModalBtn = document.getElementById("closeGroupModalBtn");
  const cancelGroupModalBtn = document.getElementById("cancelGroupModalBtn");
  const saveGroupModalBtn = document.getElementById("saveGroupModalBtn");
  const groupNameInput = document.getElementById("groupNameInput");
  const groupDescInput = document.getElementById("groupDescInput");
  const modalTitle = document.getElementById("groupModalTitle");
  const modalDescText = document.getElementById("groupModalDescText");
  const saveBtnText = document.getElementById("saveGroupModalBtnText");

  if (!groupModal) return;

  function openModal(groupId = null) {
    currentEditingGroupId = groupId;
    if (groupId && customFieldGroups[groupId]) {
      const grp = customFieldGroups[groupId];
      if (modalTitle) modalTitle.innerHTML = `<i class="fa-solid fa-pen-to-square text-cyan"></i> <span>Переименовать группу полей</span>`;
      if (modalDescText) modalDescText.textContent = `Измените название или примечание группы «${grp.name}». Это имя отображается в выпадающих списках и паспортах полей:`;
      if (saveBtnText) saveBtnText.textContent = "Сохранить изменения";
      if (groupNameInput) groupNameInput.value = grp.name || "";
      if (groupDescInput) groupDescInput.value = grp.desc || "";
    } else {
      currentEditingGroupId = null;
      if (modalTitle) modalTitle.innerHTML = `<i class="fa-solid fa-folder-plus text-cyan"></i> <span>Создать группу полей</span>`;
      if (modalDescText) modalDescText.textContent = `Созданная группа полей появится в категории «Группы». Вы сможете объединить свои поля (по хозяйству, севообороту или кластеру) и переключаться между ними:`;
      if (saveBtnText) saveBtnText.textContent = "Создать группу";
      if (groupNameInput) groupNameInput.value = "";
      if (groupDescInput) groupDescInput.value = "";
    }

    groupModal.classList.remove("hidden");
    if (groupNameInput) {
      setTimeout(() => {
        groupNameInput.focus();
        groupNameInput.select();
      }, 60);
    }
  }

  function closeModal() {
    groupModal.classList.add("hidden");
    currentEditingGroupId = null;
  }

  if (createGroupBtn) {
    createGroupBtn.addEventListener("click", () => openModal(null));
  }
  if (editGroupBtn) {
    editGroupBtn.addEventListener("click", () => {
      if (!activeGroupId || !customFieldGroups[activeGroupId]) {
        showToast("Пожалуйста, сначала выберите группу полей из списка!", true);
        return;
      }
      openModal(activeGroupId);
    });
  }
  if (closeGroupModalBtn) {
    closeGroupModalBtn.addEventListener("click", closeModal);
  }
  if (cancelGroupModalBtn) {
    cancelGroupModalBtn.addEventListener("click", closeModal);
  }

  groupModal.addEventListener("click", (e) => {
    if (e.target === groupModal) closeModal();
  });

  function saveGroup() {
    const name = groupNameInput ? groupNameInput.value.trim() : "";
    const desc = groupDescInput ? groupDescInput.value.trim() : "";

    if (!name) {
      showToast("Пожалуйста, введите название группы полей!", true);
      if (groupNameInput) groupNameInput.focus();
      return;
    }

    if (currentEditingGroupId && customFieldGroups[currentEditingGroupId]) {
      // Режим переименования и редактирования существующей группы
      const grp = customFieldGroups[currentEditingGroupId];
      const oldName = grp.name;
      grp.name = name;
      grp.desc = desc;

      saveGroupsToStorage();
      renderGroupsInRegionSelect();

      // Обновление заголовка полей группы в шапке
      const labelEl = document.getElementById("polygonSelectLabel");
      if (labelEl && activeGroupId === currentEditingGroupId) {
        labelEl.innerHTML = `<i class="fa-solid fa-folder-open"></i> Поля группы (${name}):`;
      }

      closeModal();
      showToast(`Группа полей «${oldName}» успешно переименована в «${name}»!`);
      return;
    }

    // Режим создания новой группы
    const groupId = `group_${Date.now()}`;
    customFieldGroups[groupId] = {
      id: groupId,
      name: name,
      desc: desc,
      createdAt: new Date().toISOString(),
      fields: {}
    };

    saveGroupsToStorage();
    renderGroupsInRegionSelect();
    closeModal();

    // Автоматически переключаем категорию «Регион» на созданную группу
    const regionSelect = document.getElementById("regionSelect");
    if (regionSelect) {
      regionSelect.value = groupId;
    }
    switchRegion(groupId);
    showToast(`Группа полей «${name}» создана! Нарисуйте контур поля на карте для добавления в группу.`);
  }

  if (saveGroupModalBtn) {
    saveGroupModalBtn.addEventListener("click", saveGroup);
  }

  [groupNameInput, groupDescInput].forEach(inp => {
    if (inp) {
      inp.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          saveGroup();
        } else if (e.key === "Escape") {
          closeModal();
        }
      });
    }
  });
}

// Временное хранение метаданных поля OSM при открытии диалога добавления в группу
let currentPendingOsmField = null;

// Открытие модального окна добавления найденного поля OSM в группу
function openOsmAddToGroupModal(fieldData) {
  if (!fieldData) return;
  currentPendingOsmField = fieldData;

  const modal = document.getElementById("osmFieldModal");
  if (!modal) return;

  const customFieldNameInput = document.getElementById("osmCustomFieldNameInput");
  const nameEl = document.getElementById("osmModalFieldName");
  const areaEl = document.getElementById("osmModalFieldArea");
  const cropEl = document.getElementById("osmModalFieldCrop");
  const groupSelect = document.getElementById("osmExistingGroupSelect");
  const actionExistingRadio = document.getElementById("osmActionExisting");
  const actionNewRadio = document.getElementById("osmActionNew");
  const existingWrap = document.getElementById("osmExistingGroupSelectWrap");
  const newWrap = document.getElementById("osmNewGroupInputWrap");
  const existingOptionWrap = document.getElementById("osmGroupExistingOptionWrap");
  const newNameInput = document.getElementById("osmNewGroupNameInput");

  // Обязательное поле ввода названия поля (в начале формы)
  if (customFieldNameInput) {
    customFieldNameInput.value = "";
    customFieldNameInput.classList.remove("input-error");
    const placeholderHint = (fieldData.name && !fieldData.name.startsWith("Поле OSM") && !fieldData.name.startsWith("Поле #"))
      ? `Например: ${fieldData.name}`
      : "Введите название поля (обязательно)";
    customFieldNameInput.placeholder = placeholderHint;
  }

  if (nameEl) nameEl.textContent = fieldData.name || "Поле OSM";
  if (areaEl) areaEl.textContent = `${Number(fieldData.areaHa || 0).toFixed(1)} га`;
  if (cropEl) cropEl.textContent = fieldData.cropType || "зерновые";

  // Заполнение выпадающего списка существующих групп
  const groupKeys = Object.keys(customFieldGroups);
  if (groupSelect) {
    groupSelect.innerHTML = "";
    groupSelect.classList.remove("input-error");
  }

  if (groupKeys.length > 0) {
    if (existingOptionWrap) {
      existingOptionWrap.style.opacity = "1";
      existingOptionWrap.style.pointerEvents = "auto";
    }
    if (actionExistingRadio) {
      actionExistingRadio.disabled = false;
      actionExistingRadio.checked = true;
    }
    if (actionNewRadio) {
      actionNewRadio.checked = false;
    }
    if (existingWrap) existingWrap.style.display = "block";
    if (newWrap) newWrap.style.display = "none";

    groupKeys.forEach(gId => {
      const grp = customFieldGroups[gId];
      const count = Object.keys(grp.fields || {}).length;
      const opt = document.createElement("option");
      opt.value = grp.id;
      opt.textContent = `📁 ${grp.name} (${count} ${getRussianFieldCountWord(count)})`;
      if (groupSelect) groupSelect.appendChild(opt);
    });

    // Если активна какая-то группа, выбираем её по умолчанию
    if (groupSelect && activeGroupId && customFieldGroups[activeGroupId]) {
      groupSelect.value = activeGroupId;
    }
  } else {
    // Если еще нет ни одной группы, переключаем на создание новой
    if (existingOptionWrap) {
      existingOptionWrap.style.opacity = "0.45";
      existingOptionWrap.style.pointerEvents = "none";
    }
    if (actionExistingRadio) {
      actionExistingRadio.disabled = true;
      actionExistingRadio.checked = false;
    }
    if (actionNewRadio) {
      actionNewRadio.checked = true;
    }
    if (existingWrap) existingWrap.style.display = "none";
    if (newWrap) newWrap.style.display = "block";
  }

  // Генерация названия по умолчанию для новой группы (например: "Мои поля" или "Мои поля 2")
  let defaultNewGroupName = "Мои поля";
  const existingNames = Object.values(customFieldGroups).map(g => (g.name || "").trim().toLowerCase());
  if (existingNames.includes(defaultNewGroupName.toLowerCase())) {
    let counter = 2;
    while (existingNames.includes(`мои поля ${counter}`)) {
      counter++;
    }
    defaultNewGroupName = `Мои поля ${counter}`;
  }

  if (newNameInput) {
    newNameInput.value = defaultNewGroupName;
    newNameInput.classList.remove("input-error");
  }

  modal.classList.remove("hidden");

  // Автоматическая установка фокуса на обязательное поле названия поля
  if (customFieldNameInput) {
    setTimeout(() => {
      customFieldNameInput.focus();
    }, 60);
  }
}

// Закрытие модального окна добавления поля OSM
function closeOsmFieldModal() {
  const modal = document.getElementById("osmFieldModal");
  if (modal) modal.classList.add("hidden");
  currentPendingOsmField = null;
}

// Подтверждение добавления поля OSM в выбранную или новую группу
async function confirmOsmAddToGroup() {
  if (!currentPendingOsmField) {
    showToast("Данные поля не найдены!", true);
    closeOsmFieldModal();
    return;
  }

  // 1. Проверка обязательного ввода названия поля (для обоих вариантов действий)
  const customFieldNameInput = document.getElementById("osmCustomFieldNameInput");
  const fieldName = customFieldNameInput ? customFieldNameInput.value.trim() : "";
  if (!fieldName) {
    showToast("Пожалуйста, обязательно укажите название поля!", true);
    if (customFieldNameInput) {
      customFieldNameInput.classList.add("input-error");
      customFieldNameInput.focus();
    }
    return;
  }

  const actionNewRadio = document.getElementById("osmActionNew");
  const isNewGroup = actionNewRadio && actionNewRadio.checked;
  const groupSelect = document.getElementById("osmExistingGroupSelect");
  const newNameInput = document.getElementById("osmNewGroupNameInput");

  let targetGroupId = null;
  let targetGroupName = "";

  if (isNewGroup) {
    let chosenName = newNameInput ? newNameInput.value.trim() : "";
    if (!chosenName) {
      showToast("Пожалуйста, обязательно укажите название новой папки!", true);
      if (newNameInput) {
        newNameInput.classList.add("input-error");
        newNameInput.focus();
      }
      return;
    }
    targetGroupId = `group_${Date.now()}`;
    targetGroupName = chosenName;

    customFieldGroups[targetGroupId] = {
      id: targetGroupId,
      name: targetGroupName,
      desc: "Создано из контуров OpenStreetMap",
      createdAt: new Date().toISOString(),
      fields: {}
    };
  } else {
    targetGroupId = groupSelect ? groupSelect.value : null;
    if (!targetGroupId || !customFieldGroups[targetGroupId]) {
      showToast("Пожалуйста, выберите существующую группу полей!", true);
      if (groupSelect) {
        groupSelect.classList.add("input-error");
        groupSelect.focus();
      }
      return;
    }
    targetGroupName = customFieldGroups[targetGroupId].name;
  }

  const polyId = currentPendingOsmField.polyId;
  const areaHa = currentPendingOsmField.areaHa;
  const cLat = currentPendingOsmField.centerLat;
  const cLon = currentPendingOsmField.centerLon;
  const geojson = currentPendingOsmField.geojson;
  const layer = currentPendingOsmField.layer;

  // Добавляем поле с указанным пользователем названием в выбранную группу
  if (!customFieldGroups[targetGroupId].fields) {
    customFieldGroups[targetGroupId].fields = {};
  }

  customFieldGroups[targetGroupId].fields[polyId] = {
    id: polyId,
    name: fieldName,
    color: "#00f0ff",
    geojson: geojson,
    areaHa: areaHa,
    centerLat: cLat,
    centerLon: cLon
  };

  // Удаляем контур из найденных данных OSM, так как он стал полноценным полем группы
  if (osmFieldsData[polyId]) {
    delete osmFieldsData[polyId];
  }

  // Сохраняем обновленные группы в localStorage и обновляем селекторы
  saveGroupsToStorage();
  renderGroupsInRegionSelect();

  // Переключаемся на целевую группу с сохранением видимости слоев OSM на карте
  await switchRegion(targetGroupId, false, true);

  // Обновляем стиль и всплывающее окно для сохраненного контура
  if (layer && layer.setStyle) {
    layer.setStyle({
      color: "#00f0ff",
      fillColor: "#00f0ff",
      weight: 3.5,
      fillOpacity: 0.35,
      className: "verified-field-path"
    });
    layer.bindPopup(`
      <div style="font-family: sans-serif; font-size: 13px; color: #111;">
        <strong>📁 ${targetGroupName}: ${fieldName}</strong><br>
        Площадь: <b>${Number(areaHa).toFixed(1)} га</b><br>
        <span style="color: #059669; font-weight: 600; font-size: 11.5px;">✓ Сохранено в группе «${targetGroupName}»</span><br>
        <span style="color: #0284c7; font-size: 11px;">Кликните на контур для запуска анализа</span>
      </div>
    `);
  }

  // Активируем выбранное поле и запускаем расчет аналитики
  activateAndAnalyzeField(polyId);

  closeOsmFieldModal();

  if (isNewGroup) {
    showToast(`Создана папка «${targetGroupName}» с полем «${fieldName}»!`);
  } else {
    showToast(`Поле «${fieldName}» добавлено в группу «${targetGroupName}»!`);
  }
}

// Инициализация событий модального окна добавления поля OSM
function initOsmFieldModal() {
  const modal = document.getElementById("osmFieldModal");
  const closeBtn = document.getElementById("closeOsmFieldModalBtn");
  const cancelBtn = document.getElementById("cancelOsmFieldModalBtn");
  const confirmBtn = document.getElementById("confirmOsmFieldModalBtn");
  const actionExistingRadio = document.getElementById("osmActionExisting");
  const actionNewRadio = document.getElementById("osmActionNew");
  const existingWrap = document.getElementById("osmExistingGroupSelectWrap");
  const newWrap = document.getElementById("osmNewGroupInputWrap");
  const newNameInput = document.getElementById("osmNewGroupNameInput");
  const groupSelect = document.getElementById("osmExistingGroupSelect");
  const customFieldNameInput = document.getElementById("osmCustomFieldNameInput");

  if (!modal) return;

  if (closeBtn) closeBtn.addEventListener("click", closeOsmFieldModal);
  if (cancelBtn) cancelBtn.addEventListener("click", closeOsmFieldModal);
  if (confirmBtn) confirmBtn.addEventListener("click", confirmOsmAddToGroup);

  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeOsmFieldModal();
  });

  if (customFieldNameInput) {
    customFieldNameInput.addEventListener("input", () => {
      customFieldNameInput.classList.remove("input-error");
    });
    customFieldNameInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (actionNewRadio && actionNewRadio.checked && newNameInput && !newNameInput.value.trim()) {
          newNameInput.focus();
        } else {
          confirmOsmAddToGroup();
        }
      } else if (e.key === "Escape") {
        closeOsmFieldModal();
      }
    });
  }

  if (groupSelect) {
    groupSelect.addEventListener("change", () => {
      groupSelect.classList.remove("input-error");
    });
  }

  const toggleAction = () => {
    if (actionNewRadio && actionNewRadio.checked) {
      if (existingWrap) existingWrap.style.display = "none";
      if (newWrap) newWrap.style.display = "block";
      if (newNameInput) {
        if (!customFieldNameInput || customFieldNameInput.value.trim()) {
          newNameInput.focus();
          newNameInput.select();
        }
      }
    } else {
      if (existingWrap) existingWrap.style.display = "block";
      if (newWrap) newWrap.style.display = "none";
    }
  };

  if (actionExistingRadio) actionExistingRadio.addEventListener("change", toggleAction);
  if (actionNewRadio) actionNewRadio.addEventListener("change", toggleAction);

  if (newNameInput) {
    newNameInput.addEventListener("input", () => {
      newNameInput.classList.remove("input-error");
    });
    newNameInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        confirmOsmAddToGroup();
      } else if (e.key === "Escape") {
        closeOsmFieldModal();
      }
    });
  }
}

async function initRegionsModule() {
  const regionSelect = document.getElementById("regionSelect");
  const regionSummaryBtn = document.getElementById("regionSummaryBtn");
  const searchRegionBtn = document.getElementById("searchRegionBtn");
  const summaryModal = document.getElementById("regionSummaryModal");
  const closeSummaryBtn = document.getElementById("closeRegionModalBtn");
  const regCloseBtn = document.getElementById("regCloseBtn");
  const exploreFieldsBtn = document.getElementById("regExploreFieldsBtn");

  const customModal = document.getElementById("customRegionModal");
  const closeCustomBtn = document.getElementById("closeCustomRegBtn");
  const geocodeInput = document.getElementById("geocodeInput");
  const doGeocodeBtn = document.getElementById("doGeocodeBtn");
  const geocodeResultsList = document.getElementById("geocodeResultsList");

  // 1. Загрузка каталога регионов с сервера (обогащение метаданными)
  try {
    const resp = await fetch("/api/regions");
    if (resp.ok) {
      const data = await resp.json();
      (data.regions || []).forEach(r => {
        regionsCatalog[r.id] = { ...regionsCatalog[r.id], ...r };
      });
    }
  } catch (e) {
    console.warn("Не удалось загрузить каталог регионов с сервера (используется встроенный):", e);
  }

  // 2. Обработчик смены региона в селекторе
  if (regionSelect) {
    regionSelect.addEventListener("change", async (e) => {
      await switchRegion(e.target.value);
    });
  }

  // 3. Открытие сводки региона
  if (regionSummaryBtn) {
    regionSummaryBtn.addEventListener("click", () => {
      if (!currentRegionId) {
        showToast("Пожалуйста, сначала выберите группу или регион из списка!", true);
        return;
      }
      openRegionSummaryModal(currentRegionId);
    });
  }

  // 4. Кнопка глобального поиска региона на карте
  if (searchRegionBtn) {
    searchRegionBtn.addEventListener("click", () => {
      if (customModal) {
        customModal.classList.remove("hidden");
        if (geocodeInput) {
          geocodeInput.focus();
          geocodeInput.select();
        }
      }
    });
  }

  if (closeSummaryBtn) {
    closeSummaryBtn.addEventListener("click", () => {
      if (summaryModal) summaryModal.classList.add("hidden");
    });
  }
  if (regCloseBtn) {
    regCloseBtn.addEventListener("click", () => {
      if (summaryModal) summaryModal.classList.add("hidden");
    });
  }
  if (exploreFieldsBtn) {
    exploreFieldsBtn.addEventListener("click", () => {
      if (summaryModal) summaryModal.classList.add("hidden");
      if (currentRegionId && currentRegionId.startsWith("group_")) {
        const grp = customFieldGroups[currentRegionId];
        if (grp) {
          const bounds = L.latLngBounds([]);
          Object.values(grp.fields || {}).forEach(f => {
            if (f.layer) {
              try { bounds.extend(f.layer.getBounds()); } catch (e) {}
            } else if (f.centerLat && f.centerLon) {
              bounds.extend([f.centerLat, f.centerLon]);
            }
          });
          if (bounds.isValid()) {
            map.fitBounds(bounds, { padding: [50, 50], maxZoom: 14 });
          }
        }
        return;
      }
      const region = regionsCatalog[currentRegionId];
      if (region) {
        map.flyTo([region.center_lat, region.center_lon], region.zoom || 11, { animate: true, duration: 1.2 });
      }
    });
  }

  // Закрытие модалок по клику на затемненный фон
  [summaryModal, customModal].forEach(modal => {
    if (modal) {
      modal.addEventListener("click", (e) => {
        if (e.target === modal) {
          modal.classList.add("hidden");
          if (regionSelect) regionSelect.value = currentRegionId;
        }
      });
    }
  });

  // 5. Поиск произвольного региона через Nominatim
  if (closeCustomBtn) {
    closeCustomBtn.addEventListener("click", () => {
      if (customModal) customModal.classList.add("hidden");
      if (regionSelect) regionSelect.value = currentRegionId;
    });
  }

  async function executeGeocoding() {
    const q = geocodeInput ? geocodeInput.value.trim() : "";
    if (!q || q.length < 2) {
      showToast("Введите название региона для поиска (от 2 символов)", true);
      return;
    }
    if (doGeocodeBtn) doGeocodeBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Поиск...`;
    try {
      const resp = await fetch(`/api/regions/geocode?query=${encodeURIComponent(q)}`);
      const data = await resp.json();
      const items = data.results || [];
      if (!geocodeResultsList) return;
      geocodeResultsList.innerHTML = "";

      if (items.length === 0) {
        geocodeResultsList.innerHTML = `<div style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 12px;">Ничего не найдено. Уточните запрос (например, «Тамбовская область»).</div>`;
        return;
      }

      items.forEach(item => {
        const div = document.createElement("div");
        div.className = "geocode-result-item";
        div.innerHTML = `
          <i class="fa-solid fa-map-pin"></i>
          <div>
            <strong>${item.name}</strong><br>
            <span style="color: var(--text-muted); font-size: 11px;">Широта: ${item.lat.toFixed(3)}°, Долгота: ${item.lon.toFixed(3)}°</span>
          </div>
        `;
        div.addEventListener("click", async () => {
          if (customModal) customModal.classList.add("hidden");
          await switchToCustomLocation(item);
        });
        geocodeResultsList.appendChild(div);
      });
    } catch (err) {
      console.error("Ошибка геокодинга:", err);
      showToast("Ошибка при поиске региона", true);
    } finally {
      if (doGeocodeBtn) doGeocodeBtn.innerHTML = `<i class="fa-solid fa-search"></i> Найти`;
    }
  }

  if (doGeocodeBtn) {
    doGeocodeBtn.addEventListener("click", executeGeocoding);
  }
  if (geocodeInput) {
    geocodeInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        executeGeocoding();
      }
    });
  }
}

// Переключение на регион из каталога с автоматическим поиском полей в нем
async function switchRegion(regionId, showFlyToast = true, preserveOsmFields = false) {
  // Проверяем, выбрана ли пользовательская группа полей
  if (regionId && regionId.startsWith("group_")) {
    activeGroupId = regionId;
    currentRegionId = regionId;
    const group = customFieldGroups[regionId];
    if (!group) return;

    const regionSelect = document.getElementById("regionSelect");
    if (regionSelect) regionSelect.value = regionId;

    const deleteGroupBtn = document.getElementById("deleteGroupBtn");
    if (deleteGroupBtn) deleteGroupBtn.disabled = false;
    const editGroupBtn = document.getElementById("editGroupBtn");
    if (editGroupBtn) editGroupBtn.disabled = false;

    // 1. Очистка полей предыдущего региона/группы с учетом флага сохранения контуров OSM
    clearPreviousRegionFields(group.name, !preserveOsmFields);

    // 2. Обновление заголовка полей
    const labelEl = document.getElementById("polygonSelectLabel");
    if (labelEl) {
      labelEl.innerHTML = `<i class="fa-solid fa-folder-open"></i> Поля группы (${group.name}):`;
    }

    const select = document.getElementById("polygonSelect");
    const fields = group.fields || {};
    const fieldKeys = Object.keys(fields);

    if (fieldKeys.length === 0) {
      if (select) {
        select.innerHTML = `<option value="" disabled selected>— В группе «${group.name}» пока нет полей (нарисуйте на карте) —</option>`;
      }
      if (showFlyToast) {
        showToast(`Выбрана группа «${group.name}». Нажмите «Нарисовать контур», чтобы добавить поле.`);
      }
      return;
    }

    if (select) {
      select.innerHTML = `<option value="" disabled selected>— Выберите поле группы «${group.name}» —</option>`;
    }
    let firstFieldId = null;
    const bounds = L.latLngBounds([]);

    fieldKeys.forEach((fId, idx) => {
      const f = fields[fId];
      let layer = f.layer;
      // Если слой уже отображается на карте в osmFieldLayers, повторно используем его
      if (!layer && osmFieldLayers[fId]) {
        layer = osmFieldLayers[fId];
        f.layer = layer;
      }
      if (!layer && f.geojson) {
        layer = L.geoJSON(f.geojson, {
          style: () => ({
            color: f.color || "#00f0ff",
            weight: 3.5,
            fillColor: f.color || "#00f0ff",
            fillOpacity: 0.35,
            className: "verified-field-path"
          })
        });
        f.layer = layer;
      }

      if (layer) {
        if (!map.hasLayer(layer)) {
          drawnItems.addLayer(layer);
        }
        if (layer.setStyle) {
          layer.setStyle({
            color: f.color || "#00f0ff",
            fillColor: f.color || "#00f0ff",
            weight: 3.5,
            fillOpacity: 0.35,
            className: "verified-field-path"
          });
        }
        try { bounds.extend(layer.getBounds()); } catch (e) {}

        layer.bindPopup(`
          <div style="font-family: sans-serif; font-size: 13px; color: #111;">
            <strong>📁 ${group.name}: ${f.name}</strong><br>
            Площадь: <b>${Number(f.areaHa).toFixed(1)} га</b><br>
            <span style="color: #059669; font-weight: 600; font-size: 11.5px;">✓ Сохранено в группе «${group.name}»</span><br>
            <span style="color: #0284c7; font-size: 11px;">Кликните на контур для запуска анализа</span>
          </div>
        `);

        layer.off("click");
        layer.on("click", (e) => {
          if (e && e.originalEvent) e.originalEvent.stopPropagation();
          layer.closePopup();
          userSavedFields[fId] = f;
          activateAndAnalyzeField(fId);
          showToast(`Поле выбрано: ${f.name}`);
        });
      }

      userSavedFields[fId] = f;

      const opt = document.createElement("option");
      opt.value = fId;
      opt.textContent = `● ${f.name} (${Number(f.areaHa).toFixed(1)} га)`;
      select.appendChild(opt);

      if (idx === 0) firstFieldId = fId;
    });

    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 14 });
    }

    if (firstFieldId) {
      activateAndAnalyzeField(firstFieldId);
    }

    if (showFlyToast) {
      showToast(`Выбрана группа «${group.name}» (${fieldKeys.length} ${getRussianFieldCountWord(fieldKeys.length)}).`);
    }
    return;
  }

  // Если выбран официальный регион РФ или OSM-поиск
  activeGroupId = null;
  const deleteGroupBtn = document.getElementById("deleteGroupBtn");
  if (deleteGroupBtn) deleteGroupBtn.disabled = true;
  const editGroupBtn = document.getElementById("editGroupBtn");
  if (editGroupBtn) editGroupBtn.disabled = true;
  const region = regionsCatalog[regionId];
  if (!region) return;

  currentRegionId = regionId;
  const regionSelect = document.getElementById("regionSelect");
  if (regionSelect) regionSelect.value = regionId;

  // 1. Плавный перелет карты к новому региону
  map.flyTo([region.center_lat, region.center_lon], region.zoom || 11, {
    animate: true,
    duration: showFlyToast ? 1.5 : 0.5
  });

  if (showFlyToast) {
    showToast(`🌾 Переход в регион: «${region.name}». Автопоиск полей...`);
  }

  // 2. Автоматический поиск и загрузка доступных с/х полей региона через OSM
  await loadFarmlandsForBbox(region.bbox, region.name);
}

// Переключение на произвольную геокодированную локацию
async function switchToCustomLocation(item) {
  const shortName = item.name.split(',')[0].trim();
  showToast(`🌾 Переход в регион: «${shortName}»...`);

  // Добавляем найденный регион в каталог и выбираем его в селекторе
  const customId = `custom_${shortName.toLowerCase().replace(/[^a-zA-Zа-яА-Я0-9]/g, '_')}`;
  regionsCatalog[customId] = {
    id: customId,
    name: shortName,
    macro_region: "Географический поиск",
    climate_zone: "Определяется широтой местности",
    dominant_crops: ["зерновые", "масличные"],
    center_lat: item.lat,
    center_lon: item.lon,
    zoom: 11,
    bbox: item.bbox
  };
  currentRegionId = customId;

  const regionSelect = document.getElementById("regionSelect");
  if (regionSelect) {
    let opt = regionSelect.querySelector(`option[value="${customId}"]`);
    if (!opt) {
      opt = document.createElement("option");
      opt.value = customId;
      opt.textContent = `📍 ${shortName}`;
      regionSelect.appendChild(opt);
    }
    regionSelect.value = customId;
  }

  map.flyTo([item.lat, item.lon], 11, { animate: true, duration: 1.5 });
  await loadFarmlandsForBbox(item.bbox, shortName);
}

// Функция полной очистки полей предыдущего региона
function clearPreviousRegionFields(regionLabel = "", clearOsm = true) {
  // 1. Удаляем с карты слои полей OSM предыдущего региона при необходимости
  if (clearOsm) {
    Object.keys(osmFieldLayers).forEach(polyId => {
      try {
        if (osmFieldLayers[polyId]) {
          map.removeLayer(osmFieldLayers[polyId]);
        }
      } catch (e) {}
    });
    osmFieldLayers = {};
    const clearOsmBtn = document.getElementById("clearOsmBtn");
    if (clearOsmBtn) clearOsmBtn.classList.add("hidden");
  }

  // 2. Удаляем слои сохраненных полей из карты
  Object.keys(userSavedFields).forEach(polyId => {
    const f = userSavedFields[polyId];
    if (f && f.layer) {
      try {
        if (drawnItems && drawnItems.hasLayer(f.layer)) {
          drawnItems.removeLayer(f.layer);
        }
        // Если слой не из активных слоев OSM или если полностью очищаем OSM
        if (clearOsm || !osmFieldLayers[polyId]) {
          map.removeLayer(f.layer);
        }
      } catch (e) {}
    }
  });

  // Полностью очищаем коллекцию полей, чтобы в списке не оставалось полей прошлых регионов
  userSavedFields = {};
  osmFieldsData = {};
  selectedFieldId = null;

  // 3. Очищаем выпадающий список (#polygonSelect)
  const select = document.getElementById("polygonSelect");
  if (select) {
    select.innerHTML = '<option value="" disabled selected>— Нет активной группы (выберите папку или добавьте поле через ПКМ) —</option>';
  }

  // 4. Обновляем метку в шапке
  const labelEl = document.getElementById("polygonSelectLabel");
  if (labelEl) {
    labelEl.innerHTML = '<i class="fa-solid fa-map-pin"></i> Поля группы:';
  }

  // 5. Блокируем кнопку настроек и удаления поля до выбора нового контура
  const editFieldBtn = document.getElementById("editFieldBtn");
  if (editFieldBtn) editFieldBtn.disabled = true;
  const deleteFieldBtn = document.getElementById("deleteFieldBtn");
  if (deleteFieldBtn) deleteFieldBtn.disabled = true;
}

// Загрузка контуров полей для Bounding Box с регистрацией и запуском анализа
async function loadFarmlandsForBbox(bbox, regionLabel) {
  if (!bbox || bbox.length < 4) return;
  const [minLon, minLat, maxLon, maxLat] = bbox;

  // 1. Полная очистка полей прошлого региона (слои на карте, список в селекторе, объект userSavedFields)
  clearPreviousRegionFields(regionLabel);

  const select = document.getElementById("polygonSelect");

  try {
    const resp = await fetch(`/api/osm-farmlands?min_lon=${minLon}&min_lat=${minLat}&max_lon=${maxLon}&max_lat=${maxLat}&limit=15`);
    const data = await resp.json();
    const features = data.features || [];

    if (features.length === 0) {
      showToast(`В границах региона «${regionLabel}» контуры OSM не найдены. Нарисуйте поле вручную кнопкой «Нарисовать контур».`, true);
      return;
    }

    let firstFieldId = null;

    features.forEach((f, idx) => {
      const polyId = f.properties.anon_polygon_id || `OSM-${idx + 1}`;
      const coords = f.geometry.coordinates[0];
      let sumLat = 0, sumLon = 0;
      coords.forEach(pt => { sumLon += pt[0]; sumLat += pt[1]; });
      const cLat = sumLat / coords.length;
      const cLon = sumLon / coords.length;
      const areaHa = f.properties.area_ha || calculatePolygonAreaHa(coords);

      const layer = L.geoJSON(f, {
        style: () => ({
          color: "#10b981",
          weight: 2,
          fillColor: "#10b981",
          fillOpacity: 0.28,
          className: "verified-field-path"
        })
      }).addTo(map);

      osmFieldLayers[polyId] = layer;

      const shortTitle = (f.properties.name && !f.properties.name.startsWith("Поле OSM")) ? f.properties.name : `Поле #${idx + 1}`;
      const fieldFullName = `${regionLabel}: ${shortTitle}`;

      osmFieldsData[polyId] = {
        id: polyId,
        name: fieldFullName,
        color: "#10b981",
        geojson: f,
        areaHa: areaHa,
        centerLat: cLat,
        centerLon: cLon,
        layer: layer,
        isOsm: true
      };

      layer.bindPopup(`
        <div style="font-family: sans-serif; font-size: 13px; color: #111;">
          <strong>${fieldFullName}</strong><br>
          Регион: <b>${regionLabel}</b><br>
          Культура: <b>${f.properties.crop_type || 'зерновые'}</b><br>
          Площадь: <b>${areaHa} га</b><br>
          <div style="margin-top: 6px;">
            <button type="button" class="btn-osm-popup-action" style="background: #0284c7; color: #fff; border: none; padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; display: inline-flex; align-items: center; gap: 4px;">
              <i class="fa-solid fa-folder-plus"></i> В группу полей
            </button>
          </div>
        </div>
      `);

      layer.on("popupopen", () => {
        const popupEl = layer.getPopup() ? layer.getPopup().getElement() : null;
        if (popupEl) {
          const btn = popupEl.querySelector(".btn-osm-popup-action");
          if (btn) {
            btn.onclick = (e) => {
              if (e) e.stopPropagation();
              layer.closePopup();
              openOsmAddToGroupModal({
                polyId,
                name: shortTitle,
                cropType: f.properties.crop_type || 'зерновые',
                areaHa,
                centerLat: cLat,
                centerLon: cLon,
                geojson: f,
                layer
              });
            };
          }
        }
      });

      // Обработка клика правой кнопкой мыши (ПКМ) для добавления в группу
      layer.on("contextmenu", (e) => {
        if (e && e.originalEvent) {
          e.originalEvent.preventDefault();
          e.originalEvent.stopPropagation();
        }
        if (map) map.closePopup();
        try { layer.closePopup(); } catch (err) {}

        openOsmAddToGroupModal({
          polyId,
          name: shortTitle,
          cropType: f.properties.crop_type || 'зерновые',
          areaHa,
          centerLat: cLat,
          centerLon: cLon,
          geojson: f,
          layer
        });
      });

      // Клик левой кнопкой мыши по полю выбирает его и запускает спутниковый анализ без добавления в группу
      layer.on("click", () => {
        selectAndAnalyzeOsmField(polyId);
      });

      if (idx === 0) {
        firstFieldId = polyId;
      }
    });

    showToast(`Найдено ${features.length} полей региона «${regionLabel}».`);

    // Автоматический запуск анализа первого поля региона для мгновенного отображения аналитики
    if (firstFieldId && osmFieldsData[firstFieldId]) {
      selectAndAnalyzeOsmField(firstFieldId);
    }
  } catch (err) {
    console.error("Ошибка автопоиска полей региона:", err);
    if (select) {
      select.innerHTML = `<option value="" disabled selected>— Ошибка загрузки полей региона —</option>`;
    }
  }
}

// Открытие модального окна сводной аналитики региона
async function openRegionSummaryModal(regionId) {
  const modal = document.getElementById("regionSummaryModal");
  if (!modal) return;

  const titleEl = document.getElementById("regModalTitle");
  const macroEl = document.getElementById("regMacroRegion");
  const climEl = document.getElementById("regClimateZone");
  const cropsEl = document.getElementById("regDominantCrops");
  const meanTempEl = document.getElementById("regMeanTemp");
  const maxTempEl = document.getElementById("regMaxTemp");
  const precipEl = document.getElementById("regPrecip");
  const gtkEl = document.getElementById("regGtk");
  const moistEl = document.getElementById("regMoistureStatus");
  const riskEl = document.getElementById("regRiskLevel");
  const discFieldsEl = document.getElementById("regDiscoveredFields");

  const notesBanner = document.getElementById("regGroupNotesBanner");
  const notesText = document.getElementById("regGroupNotesText");

  // Обработка пользовательской группы полей
  if (regionId && regionId.startsWith("group_")) {
    const grp = customFieldGroups[regionId];
    if (!grp) return;
    const fieldCount = Object.keys(grp.fields || {}).length;
    let totalArea = 0;
    Object.values(grp.fields || {}).forEach(f => { totalArea += (Number(f.areaHa) || 0); });

    if (titleEl) titleEl.textContent = `Группа полей: ${grp.name}`;
    if (macroEl) macroEl.textContent = grp.desc || "Пользовательская группа полей";
    if (climEl) climEl.textContent = "Локальный агрономический кластер";
    if (cropsEl) cropsEl.textContent = "Определяется культурами полей группы";
    if (meanTempEl) meanTempEl.textContent = "Мониторинг группы";
    if (maxTempEl) maxTempEl.textContent = `Полей: ${fieldCount}`;
    if (precipEl) precipEl.textContent = `Суммарная площадь: ${totalArea.toFixed(1)} га`;
    if (gtkEl) gtkEl.textContent = "Кластерный расчет";
    if (moistEl) moistEl.textContent = "Штатный режим группы";
    if (riskEl) {
      riskEl.textContent = "Штатный";
      riskEl.className = "region-kpi-value normal";
    }
    if (discFieldsEl) discFieldsEl.textContent = `${fieldCount} ${getRussianFieldCountWord(fieldCount)} в группе`;

    // Отображаем примечание к группе в сводном анализе
    if (notesBanner && notesText) {
      notesBanner.classList.remove("hidden");
      notesText.textContent = grp.desc ? grp.desc : "Примечание не указано";
    }

    modal.classList.remove("hidden");
    return;
  }

  // Для обычных регионов РФ скрываем блок примечания
  if (notesBanner) {
    notesBanner.classList.add("hidden");
  }

  const regionInfo = regionsCatalog[regionId] || { name: regionId };
  if (titleEl) titleEl.textContent = `Агроклиматическая сводка: ${regionInfo.name}`;
  if (macroEl) macroEl.textContent = regionInfo.macro_region || "Определение...";
  if (climEl) climEl.textContent = regionInfo.climate_zone || "Определение...";
  if (cropsEl) cropsEl.textContent = (regionInfo.dominant_crops || []).join(", ") || "зерновые";
  if (meanTempEl) meanTempEl.textContent = "Загрузка...";
  if (maxTempEl) maxTempEl.textContent = "Макс: -- °C";
  if (precipEl) precipEl.textContent = "Загрузка...";
  if (gtkEl) gtkEl.textContent = "Расчет...";
  if (moistEl) moistEl.textContent = "Запрос метеорологии ERA5...";
  if (riskEl) {
    riskEl.textContent = "Оценка...";
    riskEl.className = "region-kpi-value";
  }
  if (discFieldsEl) discFieldsEl.textContent = "Поиск полей...";

  // Показываем окно мгновенно по клику
  modal.classList.remove("hidden");

  const startInput = document.getElementById("startDateInput");
  const endInput = document.getElementById("endDateInput");
  const sIso = formatDateToIso(startInput ? startInput.value : selectedStartDate);
  const eIso = formatDateToIso(endInput ? endInput.value : selectedEndDate);

  try {
    const resp = await fetch(`/api/regions/${regionId}/summary?start_date=${sIso}&end_date=${eIso}`);
    if (!resp.ok) throw new Error("Ошибка получения сводки");
    const data = await resp.json();

    if (titleEl) titleEl.textContent = `Агроклиматическая сводка: ${data.region.name}`;
    if (macroEl) macroEl.textContent = data.region.macro_region;
    if (climEl) climEl.textContent = data.region.climate_zone;
    if (cropsEl) cropsEl.textContent = (data.region.dominant_crops || []).join(", ");
    if (meanTempEl) meanTempEl.textContent = `${data.weather.mean_temp_c} °C`;
    if (maxTempEl) maxTempEl.textContent = `Макс: ${data.weather.max_temp_c} °C`;
    if (precipEl) precipEl.textContent = `${data.weather.total_precip_mm} мм`;
    if (gtkEl) gtkEl.textContent = `ГТК = ${data.weather.gtk_index}`;
    if (moistEl) moistEl.textContent = data.weather.moisture_status;
    if (riskEl) {
      riskEl.textContent = data.weather.risk_level;
      riskEl.className = `region-kpi-value ${data.weather.risk_level === 'Критический' ? 'text-rose' : (data.weather.risk_level.includes('Повышенный') ? 'text-amber' : 'text-emerald')}`;
    }
    if (discFieldsEl) discFieldsEl.textContent = `Найдено с/х угодий: ${data.discovered_fields_count}`;
  } catch (e) {
    console.error("Ошибка загрузки сводки региона:", e);
    showToast("Не удалось загрузить подробную метеосводку региона.", true);
    if (moistEl) moistEl.textContent = "Метеосервис временно недоступен";
  }
}

// 4. Регистрация и управление пользовательскими полями
function registerCustomField(id, name, color, geojson, areaHa, centerLat, centerLon, layer) {
  const select = document.getElementById("polygonSelect");
  color = color || "#00f0ff";

  if (!userSavedFields[id]) {
    // Очистка заглушки списка при добавлении первого поля
    if (select && (Object.keys(userSavedFields).length === 0 || select.querySelector('option[disabled]'))) {
      select.innerHTML = "";
    }

    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = `● ${name} (${Number(areaHa).toFixed(1)} га)`;
    select.appendChild(opt);
  } else {
    // Обновление текста пункта списка, если поле уже было добавлено
    const opt = select.querySelector(`option[value="${id}"]`);
    if (opt) {
      opt.textContent = `● ${name} (${Number(areaHa).toFixed(1)} га)`;
    }
  }

  userSavedFields[id] = {
    id: id,
    name: name,
    color: color,
    geojson: geojson,
    areaHa: areaHa,
    centerLat: centerLat,
    centerLon: centerLon,
    layer: layer,
    data: null
  };

  if (layer && layer.setStyle) {
    layer.setStyle({
      color: color,
      fillColor: color,
      weight: 3.5,
      fillOpacity: 0.35,
      className: "verified-field-path"
    });
  }

  // Привязка интерактивного клика по контуру поля для прямого выбора и кинематографического зума
  const bindFieldLayerInteraction = (targetLayer) => {
    if (!targetLayer) return;
    if (typeof targetLayer.off === 'function') targetLayer.off('click');
    if (typeof targetLayer.on === 'function') {
      targetLayer.on('click', (e) => {
        if (typeof L !== 'undefined' && L.DomEvent && e) {
          L.DomEvent.stopPropagation(e);
        }
        activateAndAnalyzeField(id);
        showToast(`Выбрано поле: «${name}»`);
      });
    }
  };

  bindFieldLayerInteraction(layer);
  if (layer && typeof layer.eachLayer === 'function') {
    layer.eachLayer(childLayer => bindFieldLayerInteraction(childLayer));
  }
}

// 5. Активация и спутниковый анализ выбранного поля
async function activateAndAnalyzeField(fieldId) {
  // Если это найденное поле OSM, перенаправляем на автономный анализ без добавления в группу
  if (osmFieldsData[fieldId] && !userSavedFields[fieldId]) {
    selectAndAnalyzeOsmField(fieldId);
    return;
  }

  selectedFieldId = fieldId;
  const select = document.getElementById("polygonSelect");
  if (select) select.value = fieldId;

  const field = userSavedFields[fieldId];
  if (!field) return;

  // Активация кнопок редактирования и удаления
  const editFieldBtn = document.getElementById("editFieldBtn");
  if (editFieldBtn) editFieldBtn.disabled = false;
  const deleteFieldBtn = document.getElementById("deleteFieldBtn");
  if (deleteFieldBtn) deleteFieldBtn.disabled = false;

  // Автоматический кинематографический зум камеры на выбранное поле (профессиональный drone-flight)
  if (field.layer && typeof field.layer.getBounds === 'function' && field.layer.getBounds().isValid()) {
    const bounds = field.layer.getBounds();
    const center = bounds.getCenter();
    // Вычисляем оптимальный масштаб под размеры контура с комфортным охватом окружения
    let optimalZoom = 14;
    try {
      optimalZoom = Math.min(Math.max(map.getBoundsZoom(bounds, false, [75, 75]), 13), 16);
    } catch (e) {
      optimalZoom = 14;
    }

    if (typeof map.flyTo === 'function') {
      map.flyTo(center, optimalZoom, {
        animate: true,
        duration: 1.6,
        easeLinearity: 0.22
      });
    } else if (typeof map.flyToBounds === 'function') {
      map.flyToBounds(bounds, {
        padding: [60, 60],
        maxZoom: 15,
        duration: 1.6,
        easeLinearity: 0.22
      });
    } else {
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
    }
  } else if (field.centerLat && field.centerLon) {
    if (typeof map.flyTo === 'function') {
      map.flyTo([field.centerLat, field.centerLon], 14, {
        animate: true,
        duration: 1.6,
        easeLinearity: 0.22
      });
    } else {
      map.setView([field.centerLat, field.centerLon], 14);
    }
  }

  // Вспомогательная функция применения стилей и анимации свечения к SVG-путям
  const updateFieldLayerHighlight = (targetLayer, isSelected, strokeColor) => {
    if (!targetLayer) return;
    if (typeof targetLayer.setStyle === 'function') {
      targetLayer.setStyle({
        color: strokeColor,
        fillColor: strokeColor,
        weight: isSelected ? 4.5 : 2.2,
        fillOpacity: isSelected ? 0.48 : 0.22,
        className: isSelected ? "verified-field-path cinematic-field-selected" : "verified-field-path"
      });
    }
    // Прямое управление классами SVG-элемента для гарантии запуска CSS-анимации свечения
    if (targetLayer._path) {
      targetLayer._path.classList.add("verified-field-path");
      if (isSelected) {
        targetLayer._path.classList.remove("cinematic-field-selected");
        void targetLayer._path.offsetWidth; // Принудительный перезапуск анимации
        targetLayer._path.classList.add("cinematic-field-selected");
      } else {
        targetLayer._path.classList.remove("cinematic-field-selected");
      }
    }
    if (typeof targetLayer.eachLayer === 'function') {
      targetLayer.eachLayer(child => {
        if (child._path) {
          child._path.classList.add("verified-field-path");
          if (isSelected) {
            child._path.classList.remove("cinematic-field-selected");
            void child._path.offsetWidth;
            child._path.classList.add("cinematic-field-selected");
          } else {
            child._path.classList.remove("cinematic-field-selected");
          }
        }
      });
    }
    if (isSelected && typeof targetLayer.bringToFront === 'function') {
      targetLayer.bringToFront();
    }
  };

  // Обновление подсветки всех полей: выбранное получает кинематографический импульс
  Object.keys(userSavedFields).forEach(id => {
    const f = userSavedFields[id];
    if (f.layer) {
      const col = f.color || "#00f0ff";
      updateFieldLayerHighlight(f.layer, id === fieldId, col);
    }
  });

  // Снятие подсветки со всех найденных контуров OSM при переключении на поле группы
  Object.keys(osmFieldLayers).forEach(id => {
    const l = osmFieldLayers[id];
    if (l && typeof l.setStyle === 'function') {
      l.setStyle({
        color: "#10b981",
        fillColor: "#10b981",
        weight: 2,
        fillOpacity: 0.28,
        className: "verified-field-path"
      });
      if (l._path) l._path.classList.remove("cinematic-field-selected");
    }
  });

  updateCoordinatesDisplay(field.centerLat, field.centerLon, field.areaHa);
  await analyzeCustomPolygon(field);
}

function updateCoordinatesDisplay(lat, lon, areaHa) {
  const coordsEl = document.getElementById("geoCoordsText");
  const areaEl = document.getElementById("areaSub");
  if (coordsEl) {
    coordsEl.textContent = `${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E`;
  }
  if (areaEl && areaHa) {
    areaEl.textContent = `Площадь: ${Number(areaHa).toFixed(1)} га`;
  }
}

function calculatePolygonAreaHa(coords) {
  let area = 0;
  for (let i = 0; i < coords.length - 1; i++) {
    const x1 = coords[i][0] * 111320 * Math.cos(coords[i][1] * Math.PI / 180);
    const y1 = coords[i][1] * 110574;
    const x2 = coords[i+1][0] * 111320 * Math.cos(coords[i+1][1] * Math.PI / 180);
    const y2 = coords[i+1][1] * 110574;
    area += (x1 * y2 - x2 * y1);
  }
  return Math.abs(area / 2.0) / 10000.0;
}

let loaderInterval = null;

function showAnalyticsLoading(title = "Анализ данных поля", isGEE = true) {
  const loader = document.getElementById("analyticsLoader");
  const titleEl = document.getElementById("loaderTitle");
  const stepEl = document.getElementById("loaderStepText");
  if (!loader) return;

  if (titleEl) titleEl.textContent = title;
  loader.classList.remove("hidden");

  const steps = isGEE ? [
    "🛰️ Запрос к Google Earth Engine (Sentinel-2 L2A Harmonized, 10м)...",
    "☁️ Фильтрация облачности (SCL) и расчет зонального среднего NDVI...",
    "🌡️ Запрос суточного метеоархива ECMWF ERA5 (температура и осадки)...",
    "⚡ ML-реконструкция временного ряда и расчет Z-Score отклонений..."
  ] : [
    "📊 Загрузка мультисенсорного ряда ДЗЗ (Sentinel-2, Landsat, MODIS)...",
    "🌡️ Синхронизация метеоданных ERA5...",
    "⚡ Расчет ансамбля LightGBM + CatBoost и Z-Score нормы..."
  ];

  let stepIdx = 0;
  if (stepEl) stepEl.textContent = steps[0];

  if (loaderInterval) clearInterval(loaderInterval);
  loaderInterval = setInterval(() => {
    stepIdx = (stepIdx + 1) % steps.length;
    if (stepEl) stepEl.textContent = steps[stepIdx];
  }, 1000);
}

function hideAnalyticsLoading() {
  if (loaderInterval) {
    clearInterval(loaderInterval);
    loaderInterval = null;
  }
  const loader = document.getElementById("analyticsLoader");
  if (loader) {
    loader.classList.add("hidden");
  }
}

// 6. Комплексный анализ полигона через GEE, ERA5 и ML-пайплайн
async function analyzeCustomPolygon(field) {
  showAnalyticsLoading("Анализ контура через Google Earth Engine & ERA5", true);
  try {
    const crop = document.getElementById("cropSelect").value || "озимая пшеница";
    const startInput = document.getElementById("startDateInput");
    const endInput = document.getElementById("endDateInput");
    const startDateRaw = startInput ? startInput.value : selectedStartDate;
    const endDateRaw = endInput ? endInput.value : selectedEndDate;
    // Преобразование даты к каноническому формату ISO (ГГГГ-ММ-ДД) для API бэкенда
    const startDateIso = formatDateToIso(startDateRaw);
    const endDateIso = formatDateToIso(endDateRaw);
    const yr = startDateIso ? parseInt(startDateIso.slice(0, 4)) : 2026;
    
    document.getElementById("statusText").textContent = "Анализ ДЗЗ и погоды...";
    
    const resp = await fetch("/api/analyze-custom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        geometry: field.geojson.geometry,
        crop_type: crop,
        year: yr,
        start_date: startDateIso,
        end_date: endDateIso
      })
    });
    
    const data = await resp.json();
    if (!resp.ok) {
      showToast(data.detail || "Несуществующий или недопустимый период времени.", true);
      return;
    }
    
    // Сохранение результатов в глобальное состояние для генерации агропаспорта
    currentActiveAnalysis = { field, data };
    const passportBtn = document.getElementById("openPassportBtn");
    if (passportBtn) {
      passportBtn.disabled = false;
      passportBtn.title = `Сформировать агрономический паспорт для поля «${field.name || field.id}»`;
    }

    updateKPIs(data.kpis);
    updateDataSources(data.data_sources, field.name);
    renderCharts(data.timeseries);
    renderAnomalies(data.anomalies);
  } catch (err) {
    console.error("Ошибка анализа произвольного полигона:", err);
    showToast("Ошибка при обработке контура. Попробуйте еще раз.", true);
  } finally {
    hideAnalyticsLoading();
  }
}

// Обновление карточки источников спутниковых и метеорологических данных
function updateDataSources(sources, fieldName) {
  const dsSat = document.getElementById("dsSatellite");
  const dsWeather = document.getElementById("dsWeather");
  const dsModel = document.getElementById("dsModel");
  const dsClim = document.getElementById("dsClim");
  const badge = document.getElementById("activeFieldBadge");

  if (sources) {
    if (dsSat && sources.satellite) dsSat.textContent = sources.satellite;
    if (dsWeather && sources.weather) dsWeather.textContent = sources.weather;
    if (dsModel && sources.model) dsModel.textContent = sources.model;
    if (dsClim && sources.climatology) dsClim.textContent = sources.climatology;
  }

  if (badge) {
    badge.textContent = fieldName || "Активное поле";
  }
}

// Всплывающее информационное уведомление (Toast)
function showToast(msg, isError = false) {
  const toast = document.getElementById("appToast");
  const text = document.getElementById("toastMsg");
  text.textContent = msg;
  toast.className = `app-toast ${isError ? "error" : ""}`;
  setTimeout(() => {
    toast.classList.add("hidden");
  }, 4000);
}

// 7. Модальное окно настройки поля (название и цвет)
function initFieldModal() {
  const modal = document.getElementById("fieldModal");
  const nameInput = document.getElementById("fieldNameInput");
  const saveBtn = document.getElementById("saveFieldModalBtn");
  const cancelBtn = document.getElementById("cancelFieldModalBtn");
  const closeBtn = document.getElementById("closeFieldModalBtn");

  if (!modal) return;

  // Обработка клика по палитре предустановленных цветов
  const palette = document.getElementById("colorPaletteGroup");
  if (palette) {
    palette.querySelectorAll(".color-swatch-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        setModalColor(btn.dataset.color);
      });
    });
  }

  // Выбор произвольного цвета через color picker
  const nativePicker = document.getElementById("nativeColorPicker");
  if (nativePicker) {
    nativePicker.addEventListener("input", (e) => {
      setModalColor(e.target.value);
    });
  }

  // Сохранение изменений параметров поля
  saveBtn.addEventListener("click", () => {
    saveFieldFromModal();
  });

  nameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      saveFieldFromModal();
    }
  });

  // Отмена изменений и закрытие окна
  cancelBtn.addEventListener("click", () => {
    if (currentModalContext) currentModalContext.cancelled = true;
    closeFieldModal();
  });

  closeBtn.addEventListener("click", () => {
    if (currentModalContext) currentModalContext.cancelled = true;
    closeFieldModal();
  });

  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      if (currentModalContext) currentModalContext.cancelled = true;
      closeFieldModal();
    }
  });
}

function openFieldModal(ctx) {
  currentModalContext = ctx;
  const modal = document.getElementById("fieldModal");
  const nameInput = document.getElementById("fieldNameInput");
  const areaEl = document.getElementById("modalFieldArea");
  const coordsEl = document.getElementById("modalFieldCoords");
  const titleEl = document.getElementById("fieldModalTitle");

  if (!modal) return;

  const deleteModalBtn = document.getElementById("deleteFieldModalBtn");
  if (ctx.isNew) {
    titleEl.innerHTML = `<i class="fa-solid fa-plus-circle"></i> Сохранить новое поле`;
    if (deleteModalBtn) deleteModalBtn.style.display = "none";
  } else {
    titleEl.innerHTML = `<i class="fa-solid fa-palette"></i> Настройка поля`;
    if (deleteModalBtn) {
      deleteModalBtn.style.display = "inline-flex";
      deleteModalBtn.onclick = () => {
        if (currentModalContext && currentModalContext.id) {
          deleteSelectedField(currentModalContext.id);
        }
      };
    }
  }

  nameInput.value = ctx.name || ctx.defaultName || "";
  areaEl.textContent = `${Number(ctx.areaHa).toFixed(1)} га`;
  coordsEl.textContent = `${Number(ctx.centerLat).toFixed(3)}° N, ${Number(ctx.centerLon).toFixed(3)}° E`;

  setModalColor(ctx.color || "#00f0ff");

  modal.classList.remove("hidden");
  setTimeout(() => {
    nameInput.focus();
    nameInput.select();
  }, 100);
}

function closeFieldModal() {
  const modal = document.getElementById("fieldModal");
  if (modal) modal.classList.add("hidden");

  if (currentModalContext && currentModalContext.isNew && currentModalContext.cancelled) {
    if (currentModalContext.layer) {
      drawnItems.removeLayer(currentModalContext.layer);
    }
  }
  currentModalContext = null;
}

function setModalColor(color) {
  currentModalColor = color;
  const palette = document.getElementById("colorPaletteGroup");
  if (!palette) return;

  const swatches = palette.querySelectorAll(".color-swatch-btn");
  let matched = false;
  swatches.forEach(btn => {
    if (btn.dataset.color.toLowerCase() === color.toLowerCase()) {
      btn.classList.add("active");
      matched = true;
    } else {
      btn.classList.remove("active");
    }
  });

  const nativePicker = document.getElementById("nativeColorPicker");
  if (nativePicker) {
    nativePicker.value = color;
    if (!matched) {
      nativePicker.parentElement.style.borderColor = color;
      nativePicker.parentElement.style.color = color;
    } else {
      nativePicker.parentElement.style.borderColor = "";
      nativePicker.parentElement.style.color = "";
    }
  }
}

function saveFieldFromModal() {
  if (!currentModalContext) return;

  const nameInput = document.getElementById("fieldNameInput");
  const chosenName = nameInput.value.trim() || currentModalContext.defaultName || `Поле (${currentModalContext.areaHa.toFixed(1)} га)`;
  const chosenColor = currentModalColor || "#00f0ff";

  if (currentModalContext.isNew) {
    const fieldId = `FIELD-${String(fieldCounter++).padStart(2, '0')}`;
    registerCustomField(
      fieldId,
      chosenName,
      chosenColor,
      currentModalContext.geojson,
      currentModalContext.areaHa,
      currentModalContext.centerLat,
      currentModalContext.centerLon,
      currentModalContext.layer
    );

    // Если сейчас активна пользовательская группа полей, привязываем поле к группе
    if (activeGroupId && customFieldGroups[activeGroupId]) {
      const group = customFieldGroups[activeGroupId];
      if (!group.fields) group.fields = {};
      group.fields[fieldId] = {
        id: fieldId,
        name: chosenName,
        color: chosenColor,
        geojson: currentModalContext.geojson,
        areaHa: currentModalContext.areaHa,
        centerLat: currentModalContext.centerLat,
        centerLon: currentModalContext.centerLon,
        layer: currentModalContext.layer
      };
      saveGroupsToStorage();
      renderGroupsInRegionSelect();
    }

    activateAndAnalyzeField(fieldId);
    showToast(`Поле «${chosenName}» сохранено и принято в обработку!`);
  } else {
    // Режим редактирования существующего поля
    const fieldId = currentModalContext.id;
    const field = userSavedFields[fieldId];
    if (field) {
      field.name = chosenName;
      field.color = chosenColor;

      // Если поле принадлежит активной группе, обновляем в группе
      if (activeGroupId && customFieldGroups[activeGroupId] && customFieldGroups[activeGroupId].fields && customFieldGroups[activeGroupId].fields[fieldId]) {
        customFieldGroups[activeGroupId].fields[fieldId].name = chosenName;
        customFieldGroups[activeGroupId].fields[fieldId].color = chosenColor;
        saveGroupsToStorage();
        renderGroupsInRegionSelect();
      }

      // Обновление названия поля в выпадающем списке
      const select = document.getElementById("polygonSelect");
      const opt = select.querySelector(`option[value="${fieldId}"]`);
      if (opt) {
        opt.textContent = `● ${chosenName} (${field.areaHa.toFixed(1)} га)`;
      }

      // Обновление цвета и прозрачности полигона на карте
      if (field.layer && field.layer.setStyle) {
        field.layer.setStyle({
          color: chosenColor,
          fillColor: chosenColor,
          weight: 3.5,
          fillOpacity: 0.5
        });
      }

      // Обновление плашки активного поля в карточке источников
      const badge = document.getElementById("activeFieldBadge");
      if (badge) badge.textContent = chosenName;

      showToast(`Параметры поля «${chosenName}» обновлены!`);
    }
  }

  closeFieldModal();
}

// 8. Обновление ключевых агрономических показателей (KPI)
function updateKPIs(kpis) {
  const statusEl = document.getElementById("statusText");
  const beaconEl = document.getElementById("statusIndicator");
  const ndviEl = document.getElementById("currentNdvi");
  const cropSub = document.getElementById("cropTypeSub");
  const zscoreEl = document.getElementById("currentZscore");
  const gapsEl = document.getElementById("gapsCount");

  statusEl.textContent = kpis.current_status;
  beaconEl.className = "status-beacon";
  
  if (kpis.current_status === "Штатное развитие") {
    beaconEl.classList.add("normal");
    statusEl.className = "kpi-value highlight-green";
  } else if (kpis.current_status === "Угнетение биомассы") {
    beaconEl.classList.add("stress");
    statusEl.className = "kpi-value highlight-amber";
  } else {
    beaconEl.classList.add("critical");
    statusEl.className = "kpi-value highlight-rose";
  }

  ndviEl.textContent = kpis.current_ndvi.toFixed(3);
  cropSub.textContent = `Культура: ${kpis.crop_type}`;
  zscoreEl.textContent = `${kpis.current_zscore > 0 ? "+" : ""}${kpis.current_zscore.toFixed(2)} σ`;
  gapsEl.textContent = kpis.total_gaps_filled;
}

// Расчет количества видимых дней на временной шкале графиков
function getVisibleDays(ctx, defaultCount) {
  if (!ctx || !ctx.chart || !ctx.chart.scales || !ctx.chart.scales.x) return defaultCount;
  const x = ctx.chart.scales.x;
  if (typeof x.min === 'number' && typeof x.max === 'number' && !isNaN(x.min) && !isNaN(x.max)) {
    return Math.max(1, Math.round(x.max - x.min + 1));
  }
  return defaultCount;
}

// Синхронизация масштабирования шкалы дат между графиками NDVI и погоды
function syncChartScales(sourceChart) {
  if (isSyncingScales) return;
  if (!sourceChart || !sourceChart.scales || !sourceChart.scales.x) return;

  const target = (sourceChart === ndviChartInstance) ? weatherChartInstance : ndviChartInstance;
  const x = sourceChart.scales.x;
  const min = x.min;
  const max = x.max;

  isSyncingScales = true;
  try {
    if (target && target.scales && target.scales.x) {
      target.options.scales.x.min = min;
      target.options.scales.x.max = max;
      target.update('none'); // Мгновенная синхронизация без анимации задержки
    }
    updateDaysBadge(min, max);
  } finally {
    isSyncingScales = false;
  }
}

// Обновление бейджа с количеством отображаемых дней периода
function updateDaysBadge(min, max) {
  if (!currentTimeseriesData || !currentTimeseriesData.length) return;
  const totalDays = currentTimeseriesData.length;
  const badge = document.getElementById("chartDaysBadge");
  const text = document.getElementById("chartDaysText");
  if (!badge || !text) return;

  let visibleDays = totalDays;
  if (typeof min === 'number' && typeof max === 'number' && !isNaN(min) && !isNaN(max)) {
    visibleDays = Math.max(1, Math.min(totalDays, Math.round(max - min + 1)));
  }

  if (visibleDays < totalDays) {
    badge.classList.add("zoomed");
    text.textContent = `${visibleDays} из ${totalDays} дн.`;
    badge.title = `Масштаб увеличен: отображено ${visibleDays} дн. из ${totalDays}. (Двойной клик или кнопка «Сброс» для возврата)`;
  } else {
    badge.classList.remove("zoomed");
    text.textContent = `${totalDays} дн.`;
    badge.title = `Длительность анализируемого периода вегетации: ${totalDays} дн.`;
  }
}

// Сброс масштаба обоих графиков к исходному полному диапазону дат
function resetBothChartsZoom() {
  isSyncingScales = true;
  try {
    if (ndviChartInstance) {
      if (typeof ndviChartInstance.resetZoom === 'function') {
        ndviChartInstance.resetZoom();
      }
      if (ndviChartInstance.options && ndviChartInstance.options.scales && ndviChartInstance.options.scales.x) {
        delete ndviChartInstance.options.scales.x.min;
        delete ndviChartInstance.options.scales.x.max;
        ndviChartInstance.update();
      }
    }
    if (weatherChartInstance) {
      if (typeof weatherChartInstance.resetZoom === 'function') {
        weatherChartInstance.resetZoom();
      }
      if (weatherChartInstance.options && weatherChartInstance.options.scales && weatherChartInstance.options.scales.x) {
        delete weatherChartInstance.options.scales.x.min;
        delete weatherChartInstance.options.scales.x.max;
        weatherChartInstance.update();
      }
    }
  } finally {
    isSyncingScales = false;
  }
  if (currentTimeseriesData) {
    updateDaysBadge(0, currentTimeseriesData.length - 1);
  }
}

// 9. Отрисовка графиков динамики NDVI и метеоусловий (Chart.js)
function renderCharts(ts) {
  if (!ts || !ts.length) return;
  currentTimeseriesData = ts;

  const daysCount = ts.length;
  updateDaysBadge(0, daysCount - 1);

  const dates = ts.map(r => r.date);
  const s2Points = ts.map(r => r.s2_ndvi !== null ? r.s2_ndvi : null);
  const lsPoints = ts.map(r => r.landsat_ndvi !== null ? r.landsat_ndvi : null);
  const modPoints = ts.map(r => r.modis_ndvi !== null ? r.modis_ndvi : null);
  const recLine = ts.map(r => r.primary_ndvi_reconstructed);
  const climUpper = ts.map(r => r.clim_upper);
  const climLower = ts.map(r => r.clim_lower);
  const climMean = ts.map(r => r.clim_mean);

  const temps = ts.map(r => r.temp_c);
  const precips = ts.map(r => r.precip_mm);

  // Динамический расчет нижней границы шкалы NDVI с запасом для зимних наблюдений со снегом
  const allNdviVals = [...recLine, ...s2Points, ...lsPoints, ...modPoints].filter(v => v !== null && !isNaN(v));
  const minObservedNdvi = allNdviVals.length > 0 ? Math.min(...allNdviVals) : 0.0;
  const yAxisMin = minObservedNdvi < -0.01 ? Math.max(-0.2, Math.floor((minObservedNdvi - 0.05) * 10) / 10) : 0.0;

  // Определение охвата периода: многолетний или в пределах одного года
  const multiYearSpan = ts.length > 0 && (ts[0].date.slice(0, 4) !== ts[ts.length - 1].date.slice(0, 4));
  const tickLimit = daysCount <= 15 ? daysCount : (daysCount <= 45 ? 15 : 12);

  const dateTickCallback = function(val, index) {
    const raw = this.getLabelForValue(val);
    if (!raw || typeof raw !== 'string') return raw;
    const parts = raw.split('-');
    if (parts.length === 3) {
      return multiYearSpan ? `${parts[2]}.${parts[1]}.${parts[0].slice(2)}` : `${parts[2]}.${parts[1]}`;
    }
    return raw;
  };

  // Проверка наличия и настройка плагина масштабирования
  const hasZoomPlugin = typeof Chart !== 'undefined' && Chart.registry && !!Chart.registry.plugins.get('zoom');
  const zoomConfig = hasZoomPlugin ? {
    zoom: {
      pan: {
        enabled: true,
        mode: 'x',
        threshold: 4,
        onPan: ({ chart }) => syncChartScales(chart)
      },
      zoom: {
        wheel: {
          enabled: true,
          speed: 0.08
        },
        pinch: {
          enabled: true
        },
        mode: 'x',
        onZoom: ({ chart }) => syncChartScales(chart)
      },
      limits: {
        x: { min: 0, max: ts.length - 1, minRange: 3 }
      }
    }
  } : {};

  // Адаптивный радиус точек наблюдений в зависимости от масштаба времени
  const getPointRadiusS2 = (ctx) => {
    const cnt = getVisibleDays(ctx, daysCount);
    return cnt <= 25 ? 6.5 : (cnt <= 60 ? 5 : (cnt <= 120 ? 4 : 3.2));
  };
  const getPointRadiusLS = (ctx) => {
    const cnt = getVisibleDays(ctx, daysCount);
    return cnt <= 25 ? 6.5 : (cnt <= 60 ? 5 : (cnt <= 120 ? 4 : 3.2));
  };
  const getPointRadiusMOD = (ctx) => {
    const cnt = getVisibleDays(ctx, daysCount);
    return cnt <= 25 ? 5.5 : (cnt <= 60 ? 4 : (cnt <= 120 ? 3 : 2.2));
  };
  const getPointRadiusRec = (ctx) => {
    const cnt = getVisibleDays(ctx, daysCount);
    return cnt <= 25 ? 3.5 : (cnt <= 60 ? 2 : 0);
  };

  const ctxNdvi = document.getElementById("ndviChart").getContext("2d");
  if (ndviChartInstance) ndviChartInstance.destroy();

  ndviChartInstance = new Chart(ctxNdvi, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [
        {
          label: 'Sentinel-2 (10m)',
          data: s2Points,
          borderColor: '#00f0ff',
          backgroundColor: '#00f0ff',
          pointRadius: getPointRadiusS2,
          pointHoverRadius: (ctx) => getPointRadiusS2(ctx) + 2.5,
          showLine: false,
          pointStyle: 'circle'
        },
        {
          label: 'Landsat 8/9 (30m)',
          data: lsPoints,
          borderColor: '#10b981',
          backgroundColor: '#10b981',
          pointRadius: getPointRadiusLS,
          pointHoverRadius: (ctx) => getPointRadiusLS(ctx) + 2.5,
          showLine: false,
          pointStyle: 'triangle'
        },
        {
          label: 'MODIS (250m)',
          data: modPoints,
          borderColor: '#f59e0b',
          backgroundColor: '#f59e0b',
          pointRadius: getPointRadiusMOD,
          pointHoverRadius: (ctx) => getPointRadiusMOD(ctx) + 2.5,
          showLine: false,
          pointStyle: 'rectRot'
        },
        {
          label: 'Восстановленный primary_ndvi',
          data: recLine,
          borderColor: '#ffffff',
          backgroundColor: '#ffffff',
          borderWidth: 2,
          pointRadius: getPointRadiusRec,
          pointHoverRadius: 5.5,
          tension: 0.15
        },
        {
          label: 'Климатическая норма (μ)',
          data: climMean,
          borderColor: 'rgba(16, 185, 129, 0.75)',
          borderDash: [5, 5],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false
        },
        {
          label: 'Норма (+1σ)',
          data: climUpper,
          borderColor: 'transparent',
          pointRadius: 0,
          fill: '+1',
          backgroundColor: 'rgba(16, 185, 129, 0.12)'
        },
        {
          label: 'Норма (-1σ)',
          data: climLower,
          borderColor: 'transparent',
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.96)',
          titleFont: { family: 'JetBrains Mono', size: 13.5, weight: 'bold' },
          bodyFont: { family: 'Inter', size: 13, weight: '500' },
          borderColor: 'rgba(0, 240, 255, 0.35)',
          borderWidth: 1.5,
          padding: 12,
          boxPadding: 6,
          usePointStyle: true,
          callbacks: {
            title: (items) => {
              if (!items || !items.length) return '';
              return `📅 Дата: ${formatDateToRu(items[0].label)}`;
            },
            label: (item) => {
              if (item.raw === null || item.raw === undefined || isNaN(item.raw)) return null;
              return ` ${item.dataset.label}: ${Number(item.raw).toFixed(3)}`;
            }
          }
        },
        ...zoomConfig
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.07)' },
          ticks: {
            color: '#cbd5e1',
            font: { size: 12.5, family: 'JetBrains Mono', weight: '500' },
            maxTicksLimit: tickLimit,
            callback: dateTickCallback
          }
        },
        y: {
          min: yAxisMin,
          max: 1.0,
          grid: { color: 'rgba(255, 255, 255, 0.07)' },
          ticks: { color: '#cbd5e1', font: { size: 12.5, family: 'Inter', weight: '600' } }
        }
      }
    }
  });

  const ctxWeather = document.getElementById("weatherChart").getContext("2d");
  if (weatherChartInstance) weatherChartInstance.destroy();

  weatherChartInstance = new Chart(ctxWeather, {
    data: {
      labels: dates,
      datasets: [
        {
          type: 'line',
          label: 'Температура (°C)',
          data: temps,
          borderColor: '#f97316',
          borderWidth: 2,
          pointRadius: (ctx) => {
            const cnt = getVisibleDays(ctx, daysCount);
            return cnt <= 20 ? 3 : 0;
          },
          pointHoverRadius: 5,
          yAxisID: 'yTemp'
        },
        {
          type: 'bar',
          label: 'Осадки (мм)',
          data: precips,
          backgroundColor: 'rgba(56, 189, 248, 0.65)',
          borderColor: '#38bdf8',
          borderWidth: 1,
          yAxisID: 'yPrecip'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.96)',
          titleFont: { family: 'JetBrains Mono', size: 13.5, weight: 'bold' },
          bodyFont: { family: 'Inter', size: 13, weight: '500' },
          borderColor: 'rgba(249, 115, 22, 0.35)',
          borderWidth: 1.5,
          padding: 12,
          boxPadding: 6,
          usePointStyle: true,
          callbacks: {
            title: (items) => {
              if (!items || !items.length) return '';
              return `📅 Дата: ${formatDateToRu(items[0].label)}`;
            },
            label: (item) => {
              if (item.raw === null || item.raw === undefined || isNaN(item.raw)) return null;
              const unit = item.dataset.yAxisID === 'yTemp' ? '°C' : 'мм';
              return ` ${item.dataset.label}: ${Number(item.raw).toFixed(1)} ${unit}`;
            }
          }
        },
        ...zoomConfig
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.07)' },
          ticks: {
            color: '#cbd5e1',
            font: { size: 12.5, family: 'JetBrains Mono', weight: '500' },
            maxTicksLimit: tickLimit,
            callback: dateTickCallback
          }
        },
        yTemp: {
          type: 'linear',
          position: 'left',
          grid: { color: 'rgba(255, 255, 255, 0.07)' },
          ticks: { color: '#fb923c', font: { size: 12, family: 'JetBrains Mono', weight: '600' } },
          title: { display: true, text: 'T (°C)', color: '#fb923c', font: { size: 12.5, weight: 'bold' } }
        },
        yPrecip: {
          type: 'linear',
          position: 'right',
          grid: { display: false },
          ticks: { color: '#38bdf8', font: { size: 12, family: 'JetBrains Mono', weight: '600' } },
          title: { display: true, text: 'Осадки (мм)', color: '#38bdf8', font: { size: 12.5, weight: 'bold' } }
        }
      }
    }
  });
}

// 10. Отображение карточек выявленных аномалий и рекомендаций
function renderAnomalies(anomalies) {
  const container = document.getElementById("anomaliesList");
  container.innerHTML = "";

  if (!anomalies || anomalies.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <i class="fa-solid fa-circle-check" style="color: #10b981; font-size: 24px; margin-bottom: 8px;"></i><br>
        Периодов аномального угнетения биомассы не зафиксировано.<br>
        Вегетационное развитие протекает в рамках многолетней климатической нормы.
      </div>
    `;
    return;
  }

  anomalies.forEach((a) => {
    const card = document.createElement("div");
    card.className = `anomaly-card ${a.status === "Критическая аномалия" ? "critical" : ""}`;

    card.innerHTML = `
      <div class="anomaly-card-header">
        <span class="anomaly-date-badge">
          <i class="fa-regular fa-calendar"></i> ${formatDateToRu(a.start_date)} &mdash; ${formatDateToRu(a.end_date)} (${a.duration_days} дн.)
        </span>
        <span class="anomaly-zscore-badge">Z: ${a.min_zscore} σ (${a.status})</span>
      </div>
      <div class="anomaly-cause">
        <strong>Причина:</strong> ${a.primary_cause}
      </div>
      <div class="anomaly-recommendation">
        <i class="fa-solid fa-lightbulb" style="color: #f59e0b; margin-top: 2px;"></i>
        <span>${a.recommendation}</span>
      </div>
    `;
    container.appendChild(card);
  });
}


// ============================================================================
// 12. АГРОНОМИЧЕСКИЙ ПАСПОРТ ПОЛЯ // ГЕНЕРАЦИЯ, ПЕЧАТЬ В PDF И ЭКСПОРТ (GEOJSON/CSV)
// Реализует практическую ценность для агрономов и интеграцию с ГИС-системами
// ============================================================================

function initPassportModal() {
  const openBtn = document.getElementById("openPassportBtn");
  const closeBtn = document.getElementById("closePassportModalBtn");
  const modal = document.getElementById("passportModal");
  const printBtn = document.getElementById("printPassportBtn");
  const geoJsonBtn = document.getElementById("exportGeoJsonBtn");
  const csvBtn = document.getElementById("exportCsvBtn");

  if (openBtn) {
    openBtn.addEventListener("click", () => {
      openPassportModal();
    });
  }

  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      closePassportModal();
    });
  }

  if (printBtn) {
    printBtn.addEventListener("click", () => {
      printPassportDocument();
    });
  }

  if (geoJsonBtn) {
    geoJsonBtn.addEventListener("click", () => {
      exportPassportGeoJson();
    });
  }

  if (csvBtn) {
    csvBtn.addEventListener("click", () => {
      exportPassportCsv();
    });
  }

  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        closePassportModal();
      }
    });
  }
}

function openPassportModal() {
  if (!currentActiveAnalysis || !currentActiveAnalysis.field || !currentActiveAnalysis.data) {
    showToast("Сначала выберите или нарисуйте контур поля для выполнения спутникового анализа.", true);
    return;
  }

  const { field, data } = currentActiveAnalysis;
  const kpis = data.kpis || {};
  const anomalies = data.anomalies || [];
  const timeseries = data.timeseries || [];

  const modal = document.getElementById("passportModal");
  if (!modal) return;

  const now = new Date();
  const dateNumStr = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`;
  const cleanFieldId = (field.id || 'AOI').replace(/[^a-zA-Z0-9_-]/g, '');
  const regNumber = `PASSPORT-${cleanFieldId}-${dateNumStr}`;

  // Форматирование даты на русском языке
  const monthsRu = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"];
  const formattedDate = `${now.getDate()} ${monthsRu[now.getMonth()]} ${now.getFullYear()} г., ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

  // Заполнение шапки и сводных учетных параметров
  const regEl = document.getElementById("docRegNumber");
  const genEl = document.getElementById("docGeneratedAt");
  const calHeaderEl = document.getElementById("docCalendarPeriodHeader");
  const nameEl = document.getElementById("docFieldName");
  const cropEl = document.getElementById("docCropType");
  const areaEl = document.getElementById("docArea");
  const centroidEl = document.getElementById("docCentroid");
  const periodEl = document.getElementById("docPeriod");
  const statusEl = document.getElementById("docStatus");
  const ndviZEl = document.getElementById("docNdviZ");
  const gapsEl = document.getElementById("docGapsCount");
  const signDateEl = document.getElementById("docSignDate");

  // Определение актуального периода календаря (с проверкой input и fallback на selectedStartDate/EndDate)
  const startInput = document.getElementById("startDateInput");
  const endInput = document.getElementById("endDateInput");
  const sRu = (startInput && startInput.value) ? startInput.value.trim() : formatDateToRu(selectedStartDate);
  const eRu = (endInput && endInput.value) ? endInput.value.trim() : formatDateToRu(selectedEndDate);
  const daysCount = timeseries.length;
  const calPeriodStr = `с ${sRu} по ${eRu}`;
  const calPeriodWithDays = `с ${sRu} по ${eRu} (${daysCount} календарных дней)`;

  if (regEl) regEl.textContent = regNumber;
  if (genEl) genEl.textContent = formattedDate;
  if (calHeaderEl) calHeaderEl.textContent = `${calPeriodStr} (${daysCount} дн.)`;
  if (nameEl) nameEl.textContent = field.name || field.id || "Пользовательский контур";
  if (cropEl) cropEl.textContent = kpis.crop_type || "Зерновые культуры";
  if (areaEl) areaEl.textContent = `${Number(field.areaHa || 0).toFixed(1)} га`;
  if (centroidEl) centroidEl.textContent = `${Number(field.centerLat || 0).toFixed(4)}° N, ${Number(field.centerLon || 0).toFixed(4)}° E`;
  if (periodEl) periodEl.textContent = calPeriodWithDays;
  if (statusEl) statusEl.textContent = kpis.current_status || "Штатное развитие";
  if (ndviZEl) {
    const ndviVal = Number(kpis.current_ndvi || 0).toFixed(3);
    const zVal = Number(kpis.current_zscore || 0).toFixed(2);
    ndviZEl.textContent = `NDVI = ${ndviVal} (Z-Score: ${zVal > 0 ? '+' : ''}${zVal} σ)`;
  }
  if (gapsEl) gapsEl.textContent = `${kpis.total_gaps_filled || 0} точек (Ансамбль LightGBM + CatBoost)`;
  if (signDateEl) signDateEl.textContent = `«${now.getDate()}» ${monthsRu[now.getMonth()]} ${now.getFullYear()} г.`;

  // Снимок графика динамики вегетации для вставки в печатную форму
  const chartImg = document.getElementById("docChartImage");
  const ndviCanvas = document.getElementById("ndviChart");
  if (chartImg && ndviCanvas) {
    try {
      chartImg.src = ndviCanvas.toDataURL("image/png");
    } catch (e) {
      console.warn("Не удалось создать снимок холста графика:", e);
    }
  }

  // Заполнение таблицы аномалий
  const anomContainer = document.getElementById("docAnomaliesContainer");
  if (anomContainer) {
    if (!anomalies || anomalies.length === 0) {
      anomContainer.innerHTML = `
        <div class="doc-empty-hint">
          <i class="fa-solid fa-circle-check"></i> За анализируемый период аномальных отклонений и угнетения биомассы не обнаружено.
          Вегетационный процесс протекает в пределах многолетней климатической нормы (±1σ).
        </div>
      `;
    } else {
      let tableHtml = `
        <table class="doc-table">
          <thead>
            <tr>
              <th>Период наблюдения</th>
              <th>Длительность</th>
              <th>Z-Score / Статус</th>
              <th>Агроклиматическая первопричина</th>
              <th>Прогноз риска урожайности</th>
            </tr>
          </thead>
          <tbody>
      `;
      anomalies.forEach(a => {
        const isCritical = a.status === "Критическая аномалия";
        const riskText = isCritical ? "Высокий риск потерь (15–25%)" : "Умеренный риск (5–12%)";
        const statusClass = isCritical ? "text-rose" : "text-amber";
        tableHtml += `
          <tr>
            <td><strong>${formatDateToRu(a.start_date)} &mdash; ${formatDateToRu(a.end_date)}</strong></td>
            <td>${a.duration_days} дн.</td>
            <td><span class="${statusClass}"><strong>${a.min_zscore} σ</strong> (${a.status})</span></td>
            <td>${a.primary_cause || "Гидротермический стресс"}</td>
            <td>${riskText}</td>
          </tr>
        `;
      });
      tableHtml += `</tbody></table>`;
      anomContainer.innerHTML = tableHtml;
    }
  }

  // Заполнение директивного плана рекомендаций с гарантированным исключением дублирования критериев
  const dirContainer = document.getElementById("docDirectivesContainer");
  if (dirContainer) {
    dirContainer.innerHTML = "";
    const directives = [];
    const usedCriteria = new Set(); // Реестр уже включенных критериев для исключения повторов

    // Вспомогательная функция определения агрономического критерия рекомендации
    function identifyCriterion(text) {
      const lower = (text || "").toLowerCase();
      if (/полив|влаго|засух|осадк|мелиорац/.test(lower)) return "water";
      if (/аминокислот|адаптоген|термическ|температур|жар|стресс/.test(lower)) return "heat_stress";
      if (/скаутинг|фитосанитарн|вредител|болезн|патолог|обследован/.test(lower)) return "scouting";
      if (/азот|питан|микроэлемент|npk|карбамид|подкормк/.test(lower)) return "nutrition";
      if (/спутников|мониторинг|дзз/.test(lower)) return "monitoring";
      return "specific_" + lower.slice(0, 35);
    }

    // 1. Извлечение специфических предписаний из зафиксированных аномалий
    if (anomalies && anomalies.length > 0) {
      anomalies.forEach((a) => {
        if (!a.recommendation) return;

        // Разделяем составные рекомендации (если есть несколько законченных предложений)
        const sentences = a.recommendation.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 10);
        const partsToProcess = sentences.length > 0 ? sentences : [a.recommendation];

        partsToProcess.forEach(part => {
          const crit = identifyCriterion(part);
          // Если данный агрономический критерий уже добавлен в директивный план — НЕ повторяем его!
          if (usedCriteria.has(crit)) {
            return;
          }
          usedCriteria.add(crit);

          let tag = "Агрономическое предписание";
          if (crit === "water") tag = "Водный режим и орошение";
          else if (crit === "heat_stress") tag = "Антистрессовая обработка";
          else if (crit === "scouting") tag = "Фитосанитарный скаутинг";
          else if (crit === "nutrition") tag = "Минеральное питание";
          else if (crit === "monitoring") tag = "Контроль динамики ДЗЗ";

          directives.push({
            tag: tag,
            text: part.trim()
          });
        });
      });
    }

    // 2. Дополнение базовыми критериями регламента (ТОЛЬКО если данный критерий еще не был добавлен выше)
    const baselineCriteria = [
      {
        crit: "scouting",
        tag: "Фитосанитарный аудит",
        text: "Провести инструментальное полевое обследование (скаутинг) контрольных зон на предмет выявления скрытых очагов фитопатологий и листогрызущих вредителей."
      },
      {
        crit: "nutrition",
        tag: "Агрохимия и питание",
        text: "Оценить динамику доступного азота и микроэлементов (Zn, B, Fe); при признаках хлороза провести адресную листовую подкормку."
      },
      {
        crit: "water",
        tag: "Водный баланс",
        text: "Осуществлять непрерывный мониторинг дефицита продуктивной влаги по метеоданным ERA5 и при необходимости запланировать влагозарядковые мероприятия."
      },
      {
        crit: "monitoring",
        tag: "Спутниковый мониторинг",
        text: "Отслеживать динамику вегетационного индекса NDVI и Z-Score после проведения агротехнических работ для оценки темпов нормализации биомассы."
      }
    ];

    baselineCriteria.forEach(item => {
      // Строго проверяем, что критерий еще не представлен в плане
      if (!usedCriteria.has(item.crit)) {
        usedCriteria.add(item.crit);
        directives.push({
          tag: item.tag,
          text: item.text
        });
      }
    });

    // Сохраняем очищенный от повторов план в объект анализа для экспорта
    data.directives = directives;

    // 3. Рендеринг финального очищенного списка уникальных мероприятий
    directives.forEach((d, idx) => {
      const item = document.createElement("div");
      item.className = "doc-directive-item";
      item.innerHTML = `
        <span class="doc-directive-badge">${idx + 1}. ${d.tag}</span>
        <span>${d.text}</span>
      `;
      dirContainer.appendChild(item);
    });
  }

  modal.classList.remove("hidden");
}

function closePassportModal() {
  const modal = document.getElementById("passportModal");
  if (modal) modal.classList.add("hidden");
}

function printPassportDocument() {
  window.print();
}

// Вспомогательная функция безопасного скачивания клиентских файлов
function downloadBlob(content, filename, contentType) {
  const blob = new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 150);
}

async function exportPassportGeoJson() {
  if (!currentActiveAnalysis) return;
  const { field, data } = currentActiveAnalysis;
  showToast("Формирование GeoJSON паспорта поля...");

  try {
    const payload = {
      field_id: field.id,
      field_name: field.name || field.id,
      crop_type: data.kpis.crop_type || "зерновые",
      area_ha: field.areaHa,
      geometry: field.geojson.geometry,
      kpis: data.kpis,
      anomalies: data.anomalies || [],
      directives: data.directives || [],
      period: {
        start_date: formatDateToRu(selectedStartDate),
        end_date: formatDateToRu(selectedEndDate)
      }
    };

    const resp = await fetch("/api/export/field-geojson", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      throw new Error(`Ошибка сервера: ${resp.status}`);
    }

    const geojsonData = await resp.json();
    const safeName = (field.name || field.id).replace(/[^\wа-яА-ЯёЁ-]/g, '_');
    const filename = `Паспорт_${safeName}_${formatDateToRu(selectedStartDate)}_${formatDateToRu(selectedEndDate)}.geojson`;
    downloadBlob(JSON.stringify(geojsonData, null, 2), filename, "application/geo+json;charset=utf-8;");
    showToast(`Файл «${filename}» успешно сформирован и загружен!`);
  } catch (err) {
    console.error("Ошибка экспорта GeoJSON:", err);
    showToast("Не удалось экспортировать GeoJSON. Проверьте соединение с сервером.", true);
  }
}

async function exportPassportCsv() {
  if (!currentActiveAnalysis) return;
  const { field, data } = currentActiveAnalysis;
  showToast("Выгрузка суточного ряда наблюдений в CSV...");

  try {
    const payload = {
      field_id: field.id,
      field_name: field.name || field.id,
      timeseries: data.timeseries || []
    };

    const resp = await fetch("/api/export/field-csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      throw new Error(`Ошибка сервера: ${resp.status}`);
    }

    const res = await resp.json();
    downloadBlob(res.csv_content, res.filename, "text/csv;charset=utf-8;");
    showToast(`Файл «${res.filename}» успешно сформирован и загружен!`);
  } catch (err) {
    console.error("Ошибка экспорта CSV:", err);
    showToast("Не удалось экспортировать CSV. Проверьте соединение с сервером.", true);
  }
}

// ============================================================================
// 13. ДИНАМИЧЕСКИЙ РАЗМЕР СТРАНИЦЫ И АВТОМАТИЧЕСКАЯ АДАПТАЦИЯ КАРТЫ И ГРАФИКОВ
// Синхронизирует габариты Leaflet и холстов Chart.js при приближении/отдалении страницы (Ctrl+/-)
// ============================================================================
let windowResizeDebounceTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(windowResizeDebounceTimer);
  windowResizeDebounceTimer = setTimeout(() => {
    // Инвалидация и пересчет тайлов карты без скачков и сдвигов
    if (typeof map !== 'undefined' && map && typeof map.invalidateSize === 'function') {
      map.invalidateSize({ animate: false });
    }
    // Пересчет габаритов графиков вегетации и метеоусловий
    if (typeof ndviChartInstance !== 'undefined' && ndviChartInstance && typeof ndviChartInstance.resize === 'function') {
      ndviChartInstance.resize();
    }
    if (typeof weatherChartInstance !== 'undefined' && weatherChartInstance && typeof weatherChartInstance.resize === 'function') {
      weatherChartInstance.resize();
    }
  }, 120);
});
