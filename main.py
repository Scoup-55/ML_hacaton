import os
import base64
import json
import re
import time
import requests
import numpy as np
import cv2
import pandas as pd
from typing import Optional, Dict, Any
from ultralytics import YOLO
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from rapidfuzz import process, fuzz

# ─────────────────────────────────────────────────────────────────
# КОНСТАНТЫ БАЗЫ ДАННЫХ
# ─────────────────────────────────────────────────────────────────

DB_PATH         = "D:\\хакатон\\db_hack.csv"          # Путь к БД (можно переопределить)
DB_ENCODING     = "cp1251"               # Кодировка файла
FUZZY_THRESHOLD = 60                     # Минимальный score для нечёткого совпадения (0–100)
FUZZY_LIMIT     = 3                      # Сколько лучших вариантов возвращать при нечётком поиске

# Паттерн для извлечения штрихкода: последовательности 8–14 цифр
BARCODE_RE = re.compile(r'\b(\d{8,14})\b')

# Мусорные слова, которые мешают нечёткому поиску (цены, единицы, сервисные слова)
NOISE_RE = re.compile(
    r'\b\d+[\.,]?\d*\s*(?:₽|руб|р\.?)\b'          # цены
    r'|\b(?:цена|товар|артикул|штрихкод|скидка)\b'  # сервисные слова
    r'|\b\d+\s*(?:кг|г|мл|л|шт|упак)\b',            # количества
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────
# КЛАСС БАЗЫ ДАННЫХ
# ─────────────────────────────────────────────────────────────────

class ProductDB:
    """
    Загружает CSV-базу товаров и предоставляет быстрый поиск по коду
    и нечёткий поиск по названию.
    """

    def __init__(self, path: str = DB_PATH):
        print(f"📂 Загрузка базы товаров: {path}")
        t0 = time.time()

        df = pd.read_csv(
            path,
            encoding=DB_ENCODING,
            sep=None,
            engine="python",
            dtype={"code": str},       # штрихкоды храним как строки
        )

        # Нормализуем колонки
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.dropna(subset=["fullname", "code"])
        df["fullname"] = df["fullname"].str.strip()
        df["code"]     = df["code"].str.strip()

        # Словарь code → fullname для мгновенного поиска
        self._by_code: dict[str, str] = dict(
            zip(df["code"], df["fullname"])
        )

        # Список названий + список кодов для rapidfuzz
        self._names: list[str] = df["fullname"].tolist()
        self._codes: list[str] = df["code"].tolist()

        elapsed = time.time() - t0
        print(f"✅ База загружена: {len(self._names):,} товаров за {elapsed:.1f}с")

    # ── Публичный интерфейс ───────────────────────────────────────

    def match(self, ocr_text: str) -> dict:
        """
        Основной метод. Принимает сырой OCR-текст, возвращает словарь:
            {
              "fullname":   str,   # название из БД
              "code":       str,   # штрихкод из БД
              "match_type": str,   # "fuzzy" | "barcode" | "none"
              "score":      float, # 0–100 для fuzzy, 100.0 для barcode
              "ocr_query":  str,   # что искали (для отладки)
            }

        Приоритет: сначала нечёткое совпадение по названию,
        и только если не нашли — точный поиск по штрихкоду.
        """
        # 1. Нечёткий поиск по названию (убираем цены/единицы/шум)
        clean_query = NOISE_RE.sub(" ", ocr_text)
        clean_query = " ".join(clean_query.split()).strip()

        if len(clean_query) >= 3:
            fuzzy_result = self._match_by_name(clean_query)
            if fuzzy_result:
                return fuzzy_result

        # 2. Резервный поиск по штрихкоду (если имя не дало результата)
        barcode_result = self._match_by_barcode(ocr_text)
        if barcode_result:
            return barcode_result

        return {
            "fullname":   "",
            "code":       "",
            "match_type": "none",
            "score":      0.0,
            "ocr_query":  ocr_text[:80],
        }

    def lookup_barcode(self, barcode: str) -> Optional[str]:
        """Прямой поиск по коду. Возвращает fullname или None."""
        return self._by_code.get(barcode.strip())

    # ── Внутренние методы ────────────────────────────────────────

    def _match_by_barcode(self, text: str) -> Optional[dict]:
        """Ищет все числовые последовательности 8–14 цифр в тексте."""
        candidates = BARCODE_RE.findall(text)
        for barcode in candidates:
            fullname = self._by_code.get(barcode)
            if fullname:
                return {
                    "fullname":   fullname,
                    "code":       barcode,
                    "match_type": "barcode",
                    "score":      100.0,
                    "ocr_query":  barcode,
                }
        return None

    def _match_by_name(self, query: str) -> Optional[dict]:
        """
        Нечёткий поиск через rapidfuzz.
        Использует WRatio — комбинацию нескольких стратегий.
        """
        results = process.extract(
            query,
            self._names,
            scorer=fuzz.WRatio,
            limit=FUZZY_LIMIT,
            score_cutoff=FUZZY_THRESHOLD,
        )

        if not results:
            return None

        # Берём лучший результат
        best_name, best_score, best_idx = results[0]
        return {
            "fullname":   best_name,
            "code":       self._codes[best_idx],
            "match_type": "fuzzy",
            "score":      round(best_score, 1),
            "ocr_query":  query[:80],
            # Дополнительно: топ-3 кандидата (полезно для отладки)
            "candidates": [
                {"fullname": n, "code": self._codes[i], "score": round(s, 1)}
                for n, s, i in results
            ],
        }

# ─────────────────────────────────────────────────────────────────
# НАСТРОЙКИ ОСНОВНОГО СКРИПТА (ВИДЕО, YOLO, OCR, УЛУЧШЕНИЯ)
# ─────────────────────────────────────────────────────────────────

VIDEO_PATH        = r"D:\хакатон\видео\25_12-20.mp4"
OUTPUT_DIR        = r"D:\хакатон\detected_frames"
IMPROVED_ROI_DIR  = r"D:\хакатон\improved_roi"
YOLO_MODEL_PATH   = r"D:\хакатон\datasets\best.pt"
ESRGAN_MODEL_PATH = "RealESRGAN_x4plus.pth"
ESRGAN_URL        = (
    "https://github.com/xinntao/Real-ESRGAN/releases/"
    "download/v0.1.0/RealESRGAN_x4plus.pth"
)

# ========== НАСТРОЙКИ YANDEX VISION OCR ==========
YANDEX_API_KEY    = "API_key"
YANDEX_FOLDER_ID  = "FOLDER_ID"
YANDEX_OCR_URL    = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

APPLY_DB_MATCH    = True   # вкл/выкл сопоставление с БД

TARGET_FPS        = 5.0
CONF_THRESHOLD    = 0.25
ROTATE            = True

DETECTION_SCALE   = 1.0   # масштаб для детекции (1.0 = оригинал)
DISPLAY_SCALE     = 0.25  # масштаб для окна просмотра
SAVE_SCALE        = 1.0   # масштаб для сохраняемых кадров

APPLY_DENOISE     = True
APPLY_CONTRAST    = True
APPLY_AI_ENHANCE  = True  # Real-ESRGAN (медленно, зато красиво)
APPLY_OCR         = True  # Yandex Vision OCR — распознавание текста

# ─────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОСНОВНОГО СКРИПТА
# ─────────────────────────────────────────────────────────────────

def _load_esrgan(model_path: str, url: str) -> Optional[RealESRGANer]:
    """Загружает Real-ESRGAN, при необходимости скачивает модель."""
    try:
        if not os.path.exists(model_path):
            print("⬇️  Скачивание Real-ESRGAN (~64 МБ)...")
            response = requests.get(url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("   Готово.")

        arch = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4
        )
        model = RealESRGANer(
            scale=4, model_path=model_path, model=arch,
            tile=0, tile_pad=10, pre_pad=0, half=False
        )
        print("✅ Real-ESRGAN загружена (CPU)")
        return model
    except Exception as e:
        print(f"⚠️  Real-ESRGAN не загружена: {e}")
        return None


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    """Масштабирует кадр, если scale != 1.0."""
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def sharpen_text(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Unsharp mask — усиливает резкость текста."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)


def apply_clahe(image: np.ndarray, clip: float = 2.5) -> np.ndarray:
    """CLAHE (адаптивный контраст) по каналу L в LAB."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_full_frame(frame: np.ndarray) -> np.ndarray:
    """Лёгкое улучшение полного кадра перед сохранением."""
    result = frame.copy()
    if APPLY_DENOISE:
        result = cv2.bilateralFilter(result, 9, 10, 10)
    if APPLY_CONTRAST:
        result = apply_clahe(result)
    return result


def enhance_roi(roi: np.ndarray) -> np.ndarray:
    """
    Полный пайплайн улучшения вырезанного региона (ROI):
    1. Апскейл мелких ROI
    2. Шумоподавление
    3. Контраст (CLAHE)
    4. Резкость текста (unsharp mask)
    5. Real-ESRGAN (если включён)
    """
    h, w = roi.shape[:2]
    if w < 300 or h < 100:
        scale = max(2.0, 300 / w)
        roi = cv2.resize(roi, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    result = cv2.bilateralFilter(roi, 9, 75, 75)
    result = apply_clahe(result)
    result = sharpen_text(result, strength=1.5)

    if APPLY_AI_ENHANCE and upsampler is not None:
        try:
            rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            enhanced_rgb, _ = upsampler.enhance(rgb, outscale=1)
            result = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"⚠️  Real-ESRGAN ошибка: {e}")

    return result


def parse_price_tag_text(raw_text: str) -> Dict[str, str]:
    """
    Парсит текст из Yandex OCR и извлекает:
    - название товара
    - цену (число + валюта)
    - единицу измерения (кг/шт/л/г/мл)
    """
    if not raw_text:
        return {"name": "", "price": "", "unit": "", "extra": ""}

    # 1. Поиск цены
    price_pattern = r'(\d+[\.,]?\d*)\s*(?:₽|руб|р\.?|р)'
    price_match = re.search(price_pattern, raw_text, re.IGNORECASE)
    price = ""
    if price_match:
        price_num = price_match.group(1).replace(',', '.')
        if not re.search(r'₽|руб|р\.', price_match.group(0)):
            price = f"{price_num} ₽"
        else:
            price = price_match.group(0).strip()
        remaining_text = raw_text.replace(price_match.group(0), '')
    else:
        remaining_text = raw_text

    # 2. Поиск единицы измерения
    unit_pattern = r'\b(\d+(?:[.,]\d+)?)?\s*(кг|кгумм|к|г|мл|литр|л|шт|штук|упак|пак)\b'
    unit_match = re.search(unit_pattern, remaining_text, re.IGNORECASE)
    unit = ""
    if unit_match:
        unit = unit_match.group(2).lower()
        unit_map = {
            "кг": "кг", "кгумм": "кг", "к": "кг",
            "г": "г", "мл": "мл", "литр": "л", "л": "л",
            "шт": "шт", "штук": "шт", "упак": "упак", "пак": "упак"
        }
        unit = unit_map.get(unit, unit)
        remaining_text = remaining_text.replace(unit_match.group(0), '')

    # 3. Название — всё что осталось, без мусора
    cleanup_patterns = [
        r'\b(?:цена|товар|наименование|артикул|штрихкод|скидка|акция)\b',
        r'[^\w\sа-яА-ЯёЁa-zA-Z0-9\.,/%\-()]'
    ]
    for pat in cleanup_patterns:
        remaining_text = re.sub(pat, ' ', remaining_text, flags=re.IGNORECASE)
    name = ' '.join(remaining_text.split()).strip() or "Не распознано"
    if len(name) > 80:
        name = name[:80] + "..."

    return {"name": name, "price": price, "unit": unit, "extra": ""}


def recognize_with_yandex_ocr(roi: np.ndarray, frame_num: int, track_id: int, product_db: Optional[ProductDB]) -> dict:
    """
    Отправляет ROI в Yandex Vision OCR, парсит результат,
    сопоставляет с базой товаров и сохраняет JSON.

    Приоритет сопоставления с БД:
      1. Нечёткий поиск по названию (rapidfuzz)
      2. Точный поиск по штрихкоду (резерв)
    """
    if not APPLY_OCR:
        return {}

    try:
        # 1. Кодируем ROI в JPEG Base64
        _, encoded_img = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(encoded_img).decode("utf-8")

        # 2. Запрос к Yandex OCR
        request_body = {
            "mimeType": "JPEG",
            "languageCodes": ["ru", "en"],
            "model": "page",
            "content": img_base64
        }
        headers = {
            "Authorization": f"Api-Key {YANDEX_API_KEY}",
            "x-folder-id": YANDEX_FOLDER_ID,
            "Content-Type": "application/json"
        }

        response = requests.post(YANDEX_OCR_URL, headers=headers,
                                 json=request_body, timeout=15)
        response.raise_for_status()

        # 3. Извлекаем текст
        result_json = response.json()
        full_text = (result_json
                     .get("result", {})
                     .get("textAnnotation", {})
                     .get("fullText", ""))

        if not full_text:
            print(f"   ⚠️ Кадр {frame_num}, объект {track_id}: OCR не вернул текст")
            return {"raw_text": ""}

        # 4. Парсим OCR-текст → {name, price, unit}
        parsed_data = parse_price_tag_text(full_text)
        parsed_data["raw_text"] = full_text

        # 5. Сопоставляем с базой товаров
        if APPLY_DB_MATCH and product_db is not None:
            db_result = product_db.match(full_text)
            parsed_data["db_match"] = db_result

            mt = db_result["match_type"]
            if mt == "fuzzy":
                print(f"   🔍 Кадр {frame_num}, объект {track_id}: "
                      f"[{db_result['score']}%] {db_result['fullname']}")
            elif mt == "barcode":
                print(f"   🔢 Кадр {frame_num}, объект {track_id}: "
                      f"[штрихкод] {db_result['fullname']}")
            else:
                if parsed_data.get("price"):
                    print(f"   🏷️  Кадр {frame_num}, объект {track_id}: "
                          f"{parsed_data['name']} — {parsed_data['price']} "
                          f"/ {parsed_data['unit']} [не в базе]")
                else:
                    print(f"   ❌ Кадр {frame_num}, объект {track_id}: не найдено")
        else:
            if parsed_data.get("price"):
                print(f"   🏷️  Кадр {frame_num}, объект {track_id}: "
                      f"{parsed_data['name']} — {parsed_data['price']} / {parsed_data['unit']}")
            else:
                print(f"   🏷️  Кадр {frame_num}, объект {track_id}: распознано, но цена не найдена")

        # 6. Сохраняем JSON рядом с ROI
        json_path = os.path.join(IMPROVED_ROI_DIR, f"f{frame_num}_t{track_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)

        return parsed_data

    except requests.exceptions.RequestException as e:
        print(f"⚠️  Ошибка запроса к Yandex OCR: {e}")
        if e.response is not None:
            print(f"   Текст ответа: {e.response.text}")
    except Exception as e:
        print(f"⚠️  Непредвиденная ошибка при распознавании: {e}")

    return {}

# ─────────────────────────────────────────────────────────────────
# ИНИЦИАЛИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMPROVED_ROI_DIR, exist_ok=True)
print(f"📁 Кадры:       {OUTPUT_DIR}")
print(f"📁 ROI:         {IMPROVED_ROI_DIR}")

yolo = YOLO(YOLO_MODEL_PATH)
print("✅ YOLO загружена")

if APPLY_OCR and (YANDEX_API_KEY == "ВАШ_API_КЛЮЧ" or YANDEX_FOLDER_ID == "ВАШ_FOLDER_ID"):
    print("⚠️  ВНИМАНИЕ: не указаны YANDEX_API_KEY или YANDEX_FOLDER_ID. OCR отключён.")
    APPLY_OCR = False

upsampler = None
if APPLY_AI_ENHANCE:
    upsampler = _load_esrgan(ESRGAN_MODEL_PATH, ESRGAN_URL)

product_db = None
if APPLY_DB_MATCH:
    try:
        product_db = ProductDB(DB_PATH)
    except FileNotFoundError:
        print(f"⚠️  Файл БД не найден: {DB_PATH} — сопоставление отключено")
        APPLY_DB_MATCH = False

# ─────────────────────────────────────────────────────────────────
# ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ВИДЕО
# ─────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Не удалось открыть видео: {VIDEO_PATH}")

video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_step = max(1, round(video_fps / TARGET_FPS))
print(f"🎬 FPS видео: {video_fps:.2f} → каждый {frame_step}-й кадр")

frame_num   = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    if ROTATE:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Детекция (возможно на уменьшенном кадре)
    detection_frame = resize_frame(frame, DETECTION_SCALE)
    results = yolo.track(
        detection_frame,
        conf=CONF_THRESHOLD,
        iou=0.5,
        persist=True,
        verbose=False,
    )

    # Показываем с аннотациями
    annotated = results[0].plot()
    cv2.imshow("Детектор ценников", resize_frame(annotated, DISPLAY_SCALE))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Остановлено пользователем.")
        break

    # Сохраняем полный кадр раз в frame_step
    if frame_num % frame_step == 0:
        save_frame = enhance_full_frame(annotated)
        save_frame = resize_frame(save_frame, SAVE_SCALE)
        out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_num:06d}.jpg")
        cv2.imwrite(out_path, save_frame)
        saved_count += 1

    # Обрабатываем каждый задетектированный объект
    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        continue

    sx = 1.0 / DETECTION_SCALE
    sy = 1.0 / DETECTION_SCALE

    for i in range(len(boxes)):
        track_id = int(boxes.id[i].cpu().numpy())
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

        x1 = max(0, int(x1 * sx))
        y1 = max(0, int(y1 * sy))
        x2 = min(frame.shape[1], int(x2 * sx))
        y2 = min(frame.shape[0], int(y2 * sy))

        if x2 <= x1 or y2 <= y1:
            continue

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Улучшаем и сохраняем ROI
        enhanced_roi = enhance_roi(roi)
        roi_path = os.path.join(IMPROVED_ROI_DIR, f"f{frame_num}_t{track_id}.jpg")
        cv2.imwrite(roi_path, enhanced_roi)

        # Распознаём текст и сопоставляем с базой
        if APPLY_OCR:
            recognize_with_yandex_ocr(enhanced_roi, frame_num, track_id, product_db)

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Готово! Кадров обработано: {frame_num}, сохранено: {saved_count}")
