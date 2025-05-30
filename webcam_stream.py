import cv2
import os
import time

# Configuration
STREAM_URL = "http://10.5.34.206:4747/video"  # Replace with your iPhone stream URL
SAVE_DIR = "captured_frames"
FPS = 5                      # Save 5 frames per second
CAPTURE_DURATION = 10        # Record for 10 seconds
MAX_FRAMES = FPS * CAPTURE_DURATION

# Create directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Open video stream
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print("❌ Failed to open video stream.")
    exit()

print("✅ Stream opened. Capturing frames...")

frame_count = 0
start_time = time.time()

while frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    # Save frame as image
    filename = os.path.join(SAVE_DIR, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(filename, frame)
    print(f"📸 Saved {filename}")
    frame_count += 1
    time.sleep(1 / FPS)

cap.release()
print("✅ Capture complete.")
