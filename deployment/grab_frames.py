import cv2
import os
import time

# === CONFIGURATION ===
# <-- Replace with your phone's IP stream URL
STREAM_URL = "http://192.168.55.7:8080/video"
SAVE_DIR = "captured_png_frames"
CAPTURE_DURATION = 5       # seconds
FPS = 5                    # frames per second
MAX_FRAMES = CAPTURE_DURATION * FPS

# === SETUP ===
os.makedirs(SAVE_DIR, exist_ok=True)
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print(f"Capturing {MAX_FRAMES} PNG frames...")

frame_count = 0
interval = 1.0 / FPS

try:
    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        filename = os.path.join(SAVE_DIR, f"frame_{frame_count:04d}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

        frame_count += 1
        time.sleep(interval)

except KeyboardInterrupt:
    print("Interrupted by user.")

cap.release()
print("All frames saved.")
