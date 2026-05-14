from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import os

# ================= НАСТРОЙКИ =================
VIDEO_PATH = r"D:\хакатон\видео\25_12-20.mp4"
OUTPUT_DIR = r"D:\хакатон\detected_frames"
TARGET_FPS = 5.0
CONF_THRESHOLD = 0.25
ROTATE = True

# --- МАСШТАБИРОВАНИЕ ДЛЯ УСКОРЕНИЯ ---
# Если видео высокого разрешения, можно уменьшить его перед детекцией (например, 0.5 = вдвое)
DETECTION_SCALE = 1.0  # масштаб для детекции (1.0 = оригинал, 0.5 = половина ширины/высоты)
DISPLAY_SCALE = 0.25  # масштаб для отображения (только окно)
SAVE_SCALE = 1.0  # масштаб для сохранения (1.0 = оригинал после поворота)

# --- УЛУЧШЕНИЯ (применяются ТОЛЬКО к сохраняемым кадрам) ---
APPLY_CONTRAST_ON_SAVE = True  # CLAHE только при сохранении (медленно, но один раз на кадр)
APPLY_DENOISE_ON_SAVE = True  # шумоподавление только при сохранении (очень медленно)
DENOISE_STRENGTH = 10  # сила шумоподавления (3-10)

# ================= ПОДГОТОВКА =================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Кадры сохраняются в: {OUTPUT_DIR}")

model_path = hf_hub_download(
    repo_id="openfoodfacts/price-tag-detection",
    filename="weights/best.pt"
)
model = YOLO(model_path)
print("Модель YOLO загружена")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Ошибка: не удалось открыть {VIDEO_PATH}")
    exit()

video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0:
    video_fps = 30
step = max(1, int(round(video_fps / TARGET_FPS)))
print(f"FPS видео: {video_fps:.2f}, сохраняем каждый {step}-й кадр")


# ================= ФУНКЦИИ УЛУЧШЕНИЙ =================
def enhance_frame(frame, do_contrast, do_denoise, denoise_strength):
    """Применяет улучшения к кадру (только если нужно)"""
    result = frame.copy()
    if do_denoise:
        # Более быстрый шумодав (bilateral filter)
        result = cv2.bilateralFilter(result, 9, denoise_strength, denoise_strength)
        # Или очень быстрый GaussianBlur (но размывает)
        # result = cv2.GaussianBlur(result, (5,5), 1.0)
    if do_contrast:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


# ================= ОСНОВНОЙ ЦИКЛ =================
frame_num = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # 1. Поворот (если нужно)
    if ROTATE:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 2. Масштабирование для детекции (ускорение)
    frame_for_detection = frame
    if DETECTION_SCALE != 1.0:
        new_w = int(frame.shape[1] * DETECTION_SCALE)
        new_h = int(frame.shape[0] * DETECTION_SCALE)
        frame_for_detection = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. Трекинг (на уменьшенном кадре для скорости)
    results = model.track(frame_for_detection,
                          conf=CONF_THRESHOLD,
                          iou=0.5,
                          persist=True,
                          verbose=False)

    # 4. Отрисовка рамок на том же кадре, что и детекция (уменьшенном)
    annotated_frame = results[0].plot()  # это кадр размера frame_for_detection

    # 5. Если детекция была на уменьшенном кадре, а сохранять хотим в оригинальном размере,
    #    нужно пересчитать координаты рамок. Проще всего:
    #    - либо не уменьшать для детекции (DETECTION_SCALE=1)
    #    - либо отрисовывать на оригинальном кадре, пересчитав боксы.
    #    Для простоты предлагаю: если DETECTION_SCALE != 1, то отображаем и сохраняем как есть (уменьшенный кадр).
    #    А если вам критичен оригинальный размер, то ставьте DETECTION_SCALE=1.0.
    #    Ниже я реализую сохранение в том же разрешении, что и детекция (для простоты).

    # Для отображения масштабируем (если нужно)
    display_frame = annotated_frame
    if DISPLAY_SCALE != 1.0:
        dw = int(display_frame.shape[1] * DISPLAY_SCALE)
        dh = int(display_frame.shape[0] * DISPLAY_SCALE)
        display_frame = cv2.resize(display_frame, (dw, dh), interpolation=cv2.INTER_AREA)

    cv2.imshow("Detection (optimized)", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Сохраняем кадр (только каждый step-й)
    if frame_num % step == 0:
        # Берём кадр с рамками (в том размере, в котором детектили)
        save_frame = annotated_frame

        # Применяем улучшения (только на сохраняемых кадрах)
        if APPLY_CONTRAST_ON_SAVE or APPLY_DENOISE_ON_SAVE:
            save_frame = enhance_frame(save_frame, APPLY_CONTRAST_ON_SAVE, APPLY_DENOISE_ON_SAVE, DENOISE_STRENGTH)

        # Дополнительное масштабирование при сохранении (если нужно)
        if SAVE_SCALE != 1.0:
            sw = int(save_frame.shape[1] * SAVE_SCALE)
            sh = int(save_frame.shape[0] * SAVE_SCALE)
            save_frame = cv2.resize(save_frame, (sw, sh), interpolation=cv2.INTER_AREA)

        save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_num:06d}.jpg")
        cv2.imwrite(save_path, save_frame)
        saved_count += 1
        print(f"Сохранён кадр {frame_num} (размер: {save_frame.shape[1]}x{save_frame.shape[0]})")

cap.release()
cv2.destroyAllWindows()
print(f"\nГотово! Обработано {frame_num} кадров, сохранено {saved_count}")