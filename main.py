from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import os

# НАСТРОЙКИ 
VIDEO_PATH = r"D:\хакатон\видео\25_12-20.mp4"
OUTPUT_DIR = r"D:\хакатон\detected_frames"
TARGET_FPS = 5.0
CONF_THRESHOLD = 0.25
ROTATE = True

DETECTION_SCALE = 1.0  # масштаб для детекции 
DISPLAY_SCALE = 0.25  # масштаб для отображения 
SAVE_SCALE = 1.0  # масштаб для сохранения 


APPLY_CONTRAST_ON_SAVE = True  # CLAHE только при сохранении 
APPLY_DENOISE_ON_SAVE = True  # шумоподавление только при сохранении 
DENOISE_STRENGTH = 10  # сила шумоподавления 

# ================= ПОДГОТОВКА =================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Кадры сохраняются в: {OUTPUT_DIR}")

model_path = r"E:\Train\runs\detect\finetuned_price_tag_model\exp1\weights\best.pt"
model = YOLO(model_path)
print("Модель зYOLO агружена")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Ошибка: не удалось открыть {VIDEO_PATH}")
    exit()

video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0:
    video_fps = 30
step = max(1, int(round(video_fps / TARGET_FPS)))
print(f"FPS видео: {video_fps:.2f}, сохраняем каждый {step}-й кадр")


#  ФУНКЦИИ УЛУЧШЕНИЙ 
def enhance_frame(frame, do_contrast, do_denoise, denoise_strength):
    """Применяет улучшения к кадру (только если нужно)"""
    result = frame.copy()
    if do_denoise:
        
        result = cv2.bilateralFilter(result, 9, denoise_strength, denoise_strength)
    
       
    if do_contrast:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


# ОСНОВНОЙ ЦИКЛ 
frame_num = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # 1. Поворот 
    if ROTATE:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 2. Масштабирование для детекции 
    frame_for_detection = frame
    if DETECTION_SCALE != 1.0:
        new_w = int(frame.shape[1] * DETECTION_SCALE)
        new_h = int(frame.shape[0] * DETECTION_SCALE)
        frame_for_detection = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. Трекинг 
    results = model.track(frame_for_detection,
                          conf=CONF_THRESHOLD,
                          iou=0.5,
                          persist=True,
                          verbose=False)

   
    annotated_frame = results[0].plot()  # это кадр размера frame_for_detection

    # Для отображения масштабируем 
    display_frame = annotated_frame
    if DISPLAY_SCALE != 1.0:
        dw = int(display_frame.shape[1] * DISPLAY_SCALE)
        dh = int(display_frame.shape[0] * DISPLAY_SCALE)
        display_frame = cv2.resize(display_frame, (dw, dh), interpolation=cv2.INTER_AREA)

    cv2.imshow("Detection (optimized)", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Сохраняем кадр 
    if frame_num % step == 0:
        # Берём кадр с рамками 
        save_frame = annotated_frame

        # Применяем улучшения 
        if APPLY_CONTRAST_ON_SAVE or APPLY_DENOISE_ON_SAVE:
            save_frame = enhance_frame(save_frame, APPLY_CONTRAST_ON_SAVE, APPLY_DENOISE_ON_SAVE, DENOISE_STRENGTH)

        # Дополнительное масштабирование при сохранении 
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
