from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (1280, 720),
        "format": "BGR888"
    },
    buffer_count=2
)

picam2.configure(config)
picam2.start()

time.sleep(2)

print("Kamera başladı.")
print("Çıkmak için Q'ya bas.")

prev_time = time.time()
fps = 0

while True:

    frame = picam2.capture_array()

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "OV5647 Camera",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
