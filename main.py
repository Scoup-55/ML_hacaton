import os
import base64
import json
import re
import time
import requests
import numpy as np
import cv2
import csv
import gc
import pandas as pd
from collections import defaultdict
from typing import Optional, Dict, Any, List
from ultralytics import YOLO
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from rapidfuzz import process, fuzz

# ------------------------------------------------------------
# ВЫБОР ВИДЕО (интерактивно)
# ------------------------------------------------------------
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

def select_video_file() -> str:
    if HAS_TK:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(title="Выберите видео", filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv")])
        root.destroy()
        if path:
            return path
    path = input("Введите путь к видеофайлу: ").strip()
    while not os.path.exists(path):
        print("Файл не найден. Попробуйте снова.")
        path = input("Введите путь к видеофайлу: ").strip()
    return path

# ------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ОЦЕНКИ ПРАВДОПОДОБИЯ
# ------------------------------------------------------------
STOP_WORDS = {'и', 'в', 'на', 'с', 'по', 'из', 'у', 'о', 'для', 'а', 'но', 'за', 'над', 'под', 'без', 'до', 'к', 'от', 'или', 'же', 'бы', 'да', 'нет', 'не', 'ни', 'то', 'это', 'при', 'через', 'между'}

def extract_significant_words(text: str) -> set:
    words = re.findall(r'[а-яёa-z]+', text.lower())
    return {w for w in words if len(w) >= 3 and w not in STOP_WORDS}

def word_overlap(text1: str, text2: str) -> float:
    words1 = extract_significant_words(text1)
    words2 = extract_significant_words(text2)
    if not words1 or not words2:
        return 0.0
    inter = len(words1 & words2)
    union = len(words1 | words2)
    return inter / union if union > 0 else 0.0

# Категориальная проверка (можно расширять)
CATEGORY_KEYWORDS = {
    'напиток': ['напиток', 'сок', 'вода', 'лимонад', 'квас', 'компот', 'cola', 'fanta', 'sprite', 'вино'],
    'молочка': ['молоко', 'кефир', 'йогурт', 'сметана', 'творог', 'ряженка', 'сливки'],
    'мясо': ['сардельки', 'колбаса', 'ветчина', 'бекон', 'фарш', 'стейк'],
    'хлеб': ['хлеб', 'батон', 'булка', 'лаваш', 'лепёшка'],
    'фрукты': ['яблоко', 'груша', 'банан', 'апельсин', 'мандарин', 'лимон'],
    'овощи': ['помидор', 'огурец', 'капуста', 'морковь', 'лук', 'картофель'],
    'бакалея': ['рис', 'гречка', 'макароны', 'мука', 'сахар', 'соль'],
    'гигиена': ['тампоны', 'прокладки', 'шампунь', 'мыло', 'паста зубная'],
    'канцтовары': ['стикер', 'ручка', 'карандаш', 'тетрадь', 'бумага']
}

def same_category(ocr_text: str, db_text: str) -> bool:
    ocr_lower = ocr_text.lower()
    db_lower = db_text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        in_ocr = any(k in ocr_lower for k in keywords)
        in_db = any(k in db_lower for k in keywords)
        if in_ocr and in_db:
            return True
    return False

# ------------------------------------------------------------
# КОНСТАНТЫ БАЗЫ ДАННЫХ
# ------------------------------------------------------------
DB_PATH         = r"D:\хакатон\db_hack.csv"
DB_ENCODING     = "cp1251"
FUZZY_THRESHOLD = 60
FUZZY_LIMIT     = 3
BARCODE_RE = re.compile(r'\b(\d{8,14})\b')
NOISE_RE = re.compile(
    r'\b\d+[\.,]?\d*\s*(?:₽|руб|р\.?)\b'
    r'|\b(?:цена|товар|артикул|штрихкод|скидка)\b'
    r'|\b\d+\s*(?:кг|г|мл|л|шт|упак)\b',
    re.IGNORECASE
)

# ------------------------------------------------------------
# КЛАСС БАЗЫ ДАННЫХ (с жёсткими фильтрами)
# ------------------------------------------------------------
class ProductDB:
    def __init__(self, path: str = DB_PATH):
        print(f"📂 Загрузка базы товаров: {path}")
        t0 = time.time()
        df = pd.read_csv(path, encoding=DB_ENCODING, sep=None, engine="python", dtype={"code": str})
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.dropna(subset=["fullname", "code"])
        df["fullname"] = df["fullname"].str.strip()
        df["code"] = df["code"].str.strip()
        self._by_code = dict(zip(df["code"], df["fullname"]))
        self._names = df["fullname"].tolist()
        self._codes = df["code"].tolist()
        elapsed = time.time() - t0
        print(f"✅ База загружена: {len(self._names):,} товаров за {elapsed:.1f}с")

    def match(self, ocr_text: str) -> dict:
        # Сначала штрихкод – всегда доверяем
        barcode_result = self._match_by_barcode(ocr_text)
        if barcode_result:
            barcode_result["word_overlap"] = 1.0
            barcode_result["reliability"] = 1.0
            return barcode_result

        clean_query = NOISE_RE.sub(" ", ocr_text)
        clean_query = " ".join(clean_query.split()).strip()
        if len(clean_query) < 3:
            return self._none_result(ocr_text)

        fuzzy_result = self._match_by_name(clean_query)
        if not fuzzy_result:
            return self._none_result(ocr_text)

        ov = word_overlap(clean_query, fuzzy_result["fullname"])
        if not same_category(clean_query, fuzzy_result["fullname"]):
            ov = 0.0

        fuzzy_result["word_overlap"] = ov
        reliability = (fuzzy_result["score"] / 100.0) * ov
        fuzzy_result["reliability"] = reliability

        # ЖЁСТКИЕ ФИЛЬТРЫ
        if ov == 0.0:
            return self._none_result(ocr_text)
        if ov < 0.2 and fuzzy_result["score"] < 80:
            return self._none_result(ocr_text)

        return fuzzy_result

    def _none_result(self, ocr_text: str) -> dict:
        return {
            "fullname": "", "code": "", "match_type": "none",
            "score": 0.0, "ocr_query": ocr_text[:80],
            "word_overlap": 0.0, "reliability": 0.0
        }

    def lookup_barcode(self, barcode: str) -> Optional[str]:
        return self._by_code.get(barcode.strip())

    def _match_by_barcode(self, text: str) -> Optional[dict]:
        for barcode in BARCODE_RE.findall(text):
            fullname = self._by_code.get(barcode)
            if fullname:
                return {"fullname": fullname, "code": barcode, "match_type": "barcode", "score": 100.0, "ocr_query": barcode}
        return None

    def _match_by_name(self, query: str) -> Optional[dict]:
        clean_query = re.sub(r'[^\w\sа-яА-ЯёЁ]', ' ', query)
        clean_query = ' '.join(clean_query.split()).lower()
        results = process.extract(clean_query, self._names, scorer=fuzz.WRatio, limit=FUZZY_LIMIT*2, score_cutoff=FUZZY_THRESHOLD)
        if not results:
            return None
        query_words = extract_significant_words(clean_query)
        if not query_words:
            return None
        filtered = []
        for name, score, idx in results:
            cand_words = extract_significant_words(name)
            if query_words & cand_words:
                filtered.append((name, score, idx))
        if not filtered:
            return None
        best_name, best_score, best_idx = filtered[0]
        return {
            "fullname": best_name, "code": self._codes[best_idx], "match_type": "fuzzy",
            "score": round(best_score, 1), "ocr_query": query[:80],
            "candidates": [{"fullname": n, "code": self._codes[i], "score": round(s,1)} for n,s,i in filtered[:3]]
        }

# ------------------------------------------------------------
# НАСТРОЙКИ ОБРАБОТКИ
# ------------------------------------------------------------
OUTPUT_DIR       = r"D:\хакатон\detected_frames"
IMPROVED_ROI_DIR = r"D:\хакатон\improved_roi"
YOLO_MODEL_PATH  = r"D:\хакатон\datasets\best.pt"
ESRGAN_MODEL_PATH = "RealESRGAN_x4plus.pth"
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

YANDEX_API_KEY   = "YANDEX_API_KEy"
YANDEX_FOLDER_ID = "YANDEX_FOLDER_ID"
YANDEX_OCR_URL   = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

APPLY_DB_MATCH   = True
TARGET_FPS       = 5.0
CONF_THRESHOLD   = 0.25
ROTATE           = True
DETECTION_SCALE  = 1.0
DISPLAY_SCALE    = 0.25
SAVE_SCALE       = 1.0
APPLY_DENOISE    = True
APPLY_CONTRAST   = True
APPLY_AI_ENHANCE = True   # если вылетает, поставьте False
APPLY_OCR        = True
TRACK_TIMEOUT    = 30
USE_PRICE_REGION = False

# ------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (загрузка моделей, улучшение, OCR)
# ------------------------------------------------------------
def _load_esrgan(model_path: str, url: str) -> Optional[RealESRGANer]:
    try:
        if not os.path.exists(model_path):
            print("⬇️ Скачивание Real-ESRGAN...")
            r = requests.get(url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model = RealESRGANer(scale=4, model_path=model_path, model=arch, tile=0, tile_pad=10, pre_pad=0, half=False)
        print("✅ Real-ESRGAN загружена")
        return model
    except Exception as e:
        print(f"⚠️ ESRGAN ошибка: {e}")
        return None

def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def sharpen_text(img: np.ndarray, strength: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=2)
    return cv2.addWeighted(img, 1+strength, blurred, -strength, 0)

def apply_clahe(img: np.ndarray, clip: float = 2.5) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def extract_price_region(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    if h > w:
        return roi[int(2*h/3):, :]
    else:
        return roi[:, int(w/2):]

def enhance_full_frame(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    if APPLY_DENOISE:
        out = cv2.bilateralFilter(out, 9, 10, 10)
    if APPLY_CONTRAST:
        out = apply_clahe(out)
    return out

def enhance_roi(roi: np.ndarray, upsampler: Optional[RealESRGANer]) -> np.ndarray:
    h, w = roi.shape[:2]
    if w < 300 or h < 100:
        scale = max(2.0, 300/w)
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    out = cv2.bilateralFilter(roi, 9, 75, 75)
    out = apply_clahe(out)
    out = sharpen_text(out, 1.5)
    # Ограничиваем применение Real-ESRGAN: только если область достаточно большая
    if APPLY_AI_ENHANCE and upsampler and (h * w > 15000):
        try:
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            enhanced, _ = upsampler.enhance(rgb, outscale=1)
            out = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"⚠️ ESRGAN apply error: {e}")
    return out

def parse_price_tag_text(raw_text: str) -> Dict[str, str]:
    if not raw_text:
        return {"name": "", "price": "", "unit": "", "extra": ""}
    price = ""
    unit = ""
    remaining = raw_text
    # Цена с валютой
    price_pattern = r'(?<!\d)(\d{1,5}(?:[\.,]\d{1,2})?)\s*(?:₽|руб|р\.?|р)(?!\d)'
    match = re.search(price_pattern, remaining, re.IGNORECASE)
    if match:
        price_str = match.group(1).replace(',', '.')
        try:
            val = float(price_str)
            if val <= 100000:
                price = f"{price_str} ₽"
                remaining = remaining.replace(match.group(0), '')
        except:
            pass
    # Если не нашли – ищем последнее число
    if not price:
        numbers = re.findall(r'(?<!\d)(\d{1,5}(?:[\.,]\d{1,2})?)(?!\d)', remaining)
        for num in reversed(numbers):
            num_clean = num.replace(',', '.')
            try:
                val = float(num_clean)
                if val <= 100000:
                    if not re.search(rf'{num}\s*(кг|г|мл|л|шт)', remaining, re.IGNORECASE):
                        price = f"{num_clean} ₽"
                        remaining = remaining.replace(num, '', 1)
                        break
            except:
                continue
    # Единица измерения
    unit_pattern = r'\b(\d+(?:[.,]\d+)?)?\s*(кг|кгумм|к|г|мл|литр|л|шт|штук|упак|пак)\b'
    unit_match = re.search(unit_pattern, remaining, re.IGNORECASE)
    if unit_match:
        unit = unit_match.group(2).lower()
        unit_map = {"кг":"кг","кгумм":"кг","к":"кг","г":"г","мл":"мл","литр":"л","л":"л","шт":"шт","штук":"шт","упак":"упак","пак":"упак"}
        unit = unit_map.get(unit, unit)
        remaining = remaining.replace(unit_match.group(0), '')
    # Очистка названия
    cleanup = re.compile(r'\b(?:цена|товар|наименование|артикул|штрихкод|скидка|акция)\b', re.IGNORECASE)
    remaining = cleanup.sub(' ', remaining)
    remaining = re.sub(r'[^\w\sа-яА-ЯёЁa-zA-Z0-9\.,/%\-()]', ' ', remaining)
    name = ' '.join(remaining.split()).strip()
    if not name:
        name = "Не распознано"
    if len(name) > 80:
        name = name[:80] + "..."
    return {"name": name, "price": price, "unit": unit, "extra": ""}

def recognize_with_yandex_ocr(roi: np.ndarray, frame_num: int, track_id: int, product_db: Optional[ProductDB]) -> Optional[dict]:
    if not APPLY_OCR:
        return None
    if USE_PRICE_REGION:
        roi = extract_price_region(roi)
    try:
        _, enc = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(enc).decode("utf-8")
        headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}", "x-folder-id": YANDEX_FOLDER_ID, "Content-Type": "application/json"}
        body = {"mimeType": "JPEG", "languageCodes": ["ru","en"], "model": "page", "content": b64}
        resp = requests.post(YANDEX_OCR_URL, headers=headers, json=body, timeout=15)
        resp.raise_for_status()
        full_text = resp.json().get("result", {}).get("textAnnotation", {}).get("fullText", "")
        if not full_text:
            return None
        parsed = parse_price_tag_text(full_text)
        parsed["raw_text"] = full_text
        parsed["frame_num"] = frame_num
        parsed["track_id"] = track_id
        if APPLY_DB_MATCH and product_db:
            db_res = product_db.match(full_text)
            parsed["db_match"] = db_res
            mt = db_res["match_type"]
            if mt == "fuzzy":
                print(f"   🔍 Кадр {frame_num} т{track_id}: [{db_res['score']}% | overlap={db_res.get('word_overlap',0):.2f}] {db_res['fullname']}")
            elif mt == "barcode":
                print(f"   🔢 Кадр {frame_num} т{track_id}: [штрихкод] {db_res['fullname']}")
            else:
                if parsed.get("price"):
                    print(f"   🏷️ Кадр {frame_num} т{track_id}: {parsed['name']} — {parsed['price']} / {parsed['unit']} [не в БД]")
                else:
                    print(f"   ❌ Кадр {frame_num} т{track_id}: не найдено")
        else:
            if parsed.get("price"):
                print(f"   🏷️ Кадр {frame_num} т{track_id}: {parsed['name']} — {parsed['price']} / {parsed['unit']}")
            else:
                print(f"   🏷️ Кадр {frame_num} т{track_id}: распознано, но цена не найдена")
        json_path = os.path.join(IMPROVED_ROI_DIR, f"f{frame_num}_t{track_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        return parsed
    except Exception as e:
        print(f"⚠️ OCR ошибка: {e}")
        return None

def aggregate_track_results(track_id: int, results_list: List[dict]) -> Optional[dict]:
    if not results_list:
        return None
    def quality(r):
        has_price = 1.0 if r.get("price") else 0.0
        rel = r.get("db_match", {}).get("reliability", 0.0) if isinstance(r.get("db_match"), dict) else 0.0
        if not r.get("db_match") or r["db_match"].get("match_type") == "none":
            rel = 0.4 if has_price else 0.0
        name_ok = 0.5 if r.get("name") and r["name"] != "Не распознано" else 0.0
        return has_price * 1.5 + rel * 1.0 + name_ok
    best = max(results_list, key=quality)
    if not best.get("price") and quality(best) < 1.0:
        with_price = [r for r in results_list if r.get("price")]
        if with_price:
            best = max(with_price, key=quality)
    return best

def save_results_csv(all_results: List[dict], csv_path: str):
    if not all_results:
        print("Нет результатов для CSV")
        return
    fieldnames = ["track_id", "name", "price", "unit", "raw_text",
                  "db_fullname", "db_code", "db_match_type", "db_score", "db_reliability",
                  "roi_image_path"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in all_results:
            db = res.get("db_match", {}) if isinstance(res.get("db_match"), dict) else {}
            row = {
                "track_id": res.get("track_id", ""),
                "name": res.get("name", ""),
                "price": res.get("price", ""),
                "unit": res.get("unit", ""),
                "raw_text": res.get("raw_text", "").replace("\n", " "),
                "db_fullname": db.get("fullname", ""),
                "db_code": db.get("code", ""),
                "db_match_type": db.get("match_type", ""),
                "db_score": db.get("score", 0),
                "db_reliability": db.get("reliability", 0.0),
                "roi_image_path": os.path.join(IMPROVED_ROI_DIR, f"track_{res.get('track_id', '')}_final.jpg")
            }
            writer.writerow(row)
    print(f"✅ CSV сохранён: {csv_path}")

# ------------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ
# ------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMPROVED_ROI_DIR, exist_ok=True)

VIDEO_PATH = select_video_file()
print(f"🎥 Выбрано видео: {VIDEO_PATH}")

yolo = YOLO(YOLO_MODEL_PATH)
print("✅ YOLO загружена")

if APPLY_OCR and (YANDEX_API_KEY == "AQVN01X7SsqPYYWmaN-wO2fCpKlenaLVQPIv1twc" or YANDEX_FOLDER_ID == "b1g1f5c07ffpj7v067d0"):
    print("⚠️ Используются демонстрационные ключи Yandex OCR. Замените на свои для продакшена.")

upsampler = None
if APPLY_AI_ENHANCE:
    upsampler = _load_esrgan(ESRGAN_MODEL_PATH, ESRGAN_URL)

product_db = None
if APPLY_DB_MATCH:
    try:
        product_db = ProductDB(DB_PATH)
    except Exception as e:
        print(f"⚠️ БД не загружена: {e}. Поиск по БД отключён.")
        APPLY_DB_MATCH = False

track_history = defaultdict(list)
track_last_seen = {}

# ------------------------------------------------------------
# ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ВИДЕО
# ------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Не удалось открыть видео: {VIDEO_PATH}")

fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_step = max(1, round(fps_video / TARGET_FPS))
print(f"🎬 FPS видео: {fps_video:.2f} → обрабатываем каждый {frame_step}-й кадр")

frame_num = 0
saved_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
    if ROTATE:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    det_frame = resize_frame(frame, DETECTION_SCALE)
    results = yolo.track(det_frame, conf=CONF_THRESHOLD, iou=0.5, persist=True, verbose=False)

    annotated = results[0].plot()
    cv2.imshow("Детектор ценников", resize_frame(annotated, DISPLAY_SCALE))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if frame_num % frame_step == 0:
        out_frame = enhance_full_frame(annotated)
        out_frame = resize_frame(out_frame, SAVE_SCALE)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_num:06d}.jpg"), out_frame)
        saved_frames += 1

    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        to_remove = [tid for tid, last in track_last_seen.items() if (frame_num - last) > TRACK_TIMEOUT]
        for tid in to_remove:
            final = aggregate_track_results(tid, track_history.get(tid, []))
            if final:
                final["track_id"] = tid
                with open(os.path.join(IMPROVED_ROI_DIR, f"track_{tid}_final.json"), "w", encoding="utf-8") as f:
                    json.dump(final, f, ensure_ascii=False, indent=2)
                print(f"✅ Трек {tid} завершён: {final.get('name')} — {final.get('price')}")
            del track_last_seen[tid]
            if tid in track_history:
                del track_history[tid]
            gc.collect()
        continue

    sx = 1.0 / DETECTION_SCALE
    sy = 1.0 / DETECTION_SCALE
    current_ids = set()

    for i in range(len(boxes)):
        track_id = int(boxes.id[i].cpu().numpy())
        current_ids.add(track_id)
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
        enhanced = enhance_roi(roi, upsampler)
        roi_path = os.path.join(IMPROVED_ROI_DIR, f"f{frame_num}_t{track_id}.jpg")
        cv2.imwrite(roi_path, enhanced)

        if APPLY_OCR:
            parsed = recognize_with_yandex_ocr(enhanced, frame_num, track_id, product_db)
            if parsed:
                track_history[track_id].append(parsed)
        track_last_seen[track_id] = frame_num

    to_remove = [tid for tid in track_last_seen if tid not in current_ids and (frame_num - track_last_seen[tid]) > TRACK_TIMEOUT]
    for tid in to_remove:
        final = aggregate_track_results(tid, track_history.get(tid, []))
        if final:
            final["track_id"] = tid
            with open(os.path.join(IMPROVED_ROI_DIR, f"track_{tid}_final.json"), "w", encoding="utf-8") as f:
                json.dump(final, f, ensure_ascii=False, indent=2)
            print(f"✅ Трек {tid} завершён: {final.get('name')} — {final.get('price')}")
        del track_last_seen[tid]
        if tid in track_history:
            del track_history[tid]
        gc.collect()

cap.release()
cv2.destroyAllWindows()
gc.collect()

# ------------------------------------------------------------
# ФИНАЛЬНАЯ АГРЕГАЦИЯ И СОХРАНЕНИЕ CSV
# ------------------------------------------------------------
print("\n🏁 Завершение видео. Агрегация оставшихся треков...")
final_results = []
for tid, hist in track_history.items():
    final = aggregate_track_results(tid, hist)
    if final:
        final["track_id"] = tid
        final_results.append(final)
        with open(os.path.join(IMPROVED_ROI_DIR, f"track_{tid}_final.json"), "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        print(f"✅ Трек {tid}: {final.get('name')} — {final.get('price')}")

# Собираем также треки, уже сохранённые по таймауту
for fname in os.listdir(IMPROVED_ROI_DIR):
    if fname.startswith("track_") and fname.endswith("_final.json"):
        try:
            tid = int(fname.split("_")[1])
            if tid not in [r.get("track_id") for r in final_results]:
                with open(os.path.join(IMPROVED_ROI_DIR, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "track_id" not in data:
                        data["track_id"] = tid
                    final_results.append(data)
        except:
            pass

# Удаляем дубликаты
uniq = {}
for r in final_results:
    tid = r.get("track_id")
    if tid not in uniq:
        uniq[tid] = r
final_results = list(uniq.values())

if final_results:
    csv_path = os.path.join(OUTPUT_DIR, "final_results.csv")
    save_results_csv(final_results, csv_path)
else:
    print("Нет результатов для сохранения.")

print(f"\n✅ Готово! Обработано кадров: {frame_num}, сохранено полных кадров: {saved_frames}")
