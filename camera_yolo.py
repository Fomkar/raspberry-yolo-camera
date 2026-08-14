from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import threading
import time


# =========================================================
# AYARLAR
# =========================================================

CAMERA_WIDTH = 1270
CAMERA_HEIGHT = 720

YOLO_WIDTH = 640
YOLO_HEIGHT = 360

YOLO_IMAGE_SIZE = 256
CONFIDENCE = 0.5

MODEL_PATH = "models/yolo11n.pt"


# =========================================================
# GLOBAL DEĞİŞKENLER
# =========================================================

latest_frame = None
running = True

frame_lock = threading.Lock()


# =========================================================
# KAMERA THREAD
# =========================================================

def camera_thread(picam2):

    global latest_frame
    global running

    while running:

        frame = picam2.capture_array()

        # RGB -> BGR
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        # Sadece en son görüntüyü sakla
        with frame_lock:
            latest_frame = frame


# =========================================================
# KAMERA
# =========================================================

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "size": (
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        ),
        "format": "RGB888"
    },
    buffer_count=2
)

picam2.configure(camera_config)

picam2.start()

time.sleep(2)


# =========================================================
# YOLO
# =========================================================

print("YOLO modeli yükleniyor...")

model = YOLO(MODEL_PATH)

print("YOLO hazır.")


# =========================================================
# CAMERA THREAD BAŞLAT
# =========================================================

camera_thread_obj = threading.Thread(
    target=camera_thread,
    args=(picam2,),
    daemon=True
)

camera_thread_obj.start()

print("Kamera başladı.")
print("Çıkmak için Q tuşuna bas.")


# =========================================================
# FPS
# =========================================================

fps_start = time.time()
fps_counter = 0
fps = 0


# =========================================================
# ANA DÖNGÜ
# =========================================================

while True:

    # En son frame'i al
    with frame_lock:

        if latest_frame is None:
            continue

        frame = latest_frame.copy()


    # =====================================================
    # YOLO İÇİN KÜÇÜLT
    # =====================================================

    yolo_frame = cv2.resize(
        frame,
        (
            YOLO_WIDTH,
            YOLO_HEIGHT
        )
    )


    # =====================================================
    # YOLO
    # =====================================================

    results = model(
        yolo_frame,
        imgsz=YOLO_IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )


    # =====================================================
    # SONUCU ÇİZ
    # =====================================================

    result_frame = results[0].plot()


    # =====================================================
    # EKRAN İÇİN TEKRAR BÜYÜT
    # =====================================================

    display_frame = cv2.resize(
        result_frame,
        (
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        )
    )


    # =====================================================
    # FPS
    # =====================================================

    fps_counter += 1

    elapsed = time.time() - fps_start

    if elapsed >= 1.0:

        fps = fps_counter / elapsed

        fps_counter = 0
        fps_start = time.time()


    cv2.putText(
        display_frame,
        f"YOLO FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # =====================================================
    # GÖRÜNTÜ
    # =====================================================

    cv2.imshow(
        "Raspberry Pi - YOLO",
        display_frame
    )


    # =====================================================
    # Q İLE ÇIKIŞ
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        running = False

        break


# =========================================================
# KAPAT
# =========================================================

running = False

time.sleep(0.2)

picam2.stop()

cv2.destroyAllWindows()

print("Program kapatıldı.")
