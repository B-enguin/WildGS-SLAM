import cv2
import os
import time
import requests

# === CONFIGURATION ===
# Replace with your iPhone's DroidCam stream IP
STREAM_URL = "http://10.5.225.69:4747/video"  # ← Replace this!
SAVE_DIR = "captured_png_frames"
CAPTURE_DURATION = 5       # seconds
FPS = 5                    # frames per second
MAX_FRAMES = CAPTURE_DURATION * FPS

# === SETUP ===
print("Checking stream availability...")
try:
    r = requests.get(STREAM_URL, timeout=5)
    if r.status_code != 200:
        print("Warning: Stream responded but not 200 OK.")
except requests.exceptions.RequestException as e:
    print(f"Cannot connect to stream: {e}")
    exit(1)

os.makedirs(SAVE_DIR, exist_ok=True)
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print(f"Capturing up to {MAX_FRAMES} PNG frames at {FPS} FPS...")
frame_count = 0
interval = 1.0 / FPS
start_time = time.time()

try:
    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Show the live video feed
        cv2.imshow("DroidCam Feed", frame)

        # Save the frame
        filename = os.path.join(SAVE_DIR, f"frame_{frame_count:04d}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        frame_count += 1

        # Wait to maintain target FPS
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break
        time.sleep(interval)

except KeyboardInterrupt:
    print("Interrupted by user.")

cap.release()
cv2.destroyAllWindows()
print("Finished capturing frames.")
