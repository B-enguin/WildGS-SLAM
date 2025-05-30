import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datetime import datetime

# Assume this is your LiveStreamDataset from earlier
from datasets import LiveStreamDataset  # replace with actual path

# Configuration
cfg = {
    'dataset': 'livestream',
    'cam': {
        'H': 480,
        'W': 640,
        'fx': 525,
        'fy': 525,
        'cx': 320,
        'cy': 365.55963134765625,
        'H_out': 360,
        'W_out': 640,
        'H_edge': 0,
        'W_edge': 0,
    },
    'data': {
        'stream_url': 'http://10.5.82.7:4747/video',  # Replace with your stream
    }
}

# Create output folder with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_folder = f"stream_output_{timestamp}"
os.makedirs(output_folder, exist_ok=True)

# Initialize dataset and dataloader
dataset = LiveStreamDataset(cfg)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# Capture N frames
num_samples = 100
# for i, (index, frame_tensor) in enumerate(dataloader):
for i in range(len(dataset)):
    index, frame_tensor = dataset[i]  # Assuming dataset returns a single frame tensor
    frame_tensor = frame_tensor.squeeze(0)  # remove batch dimension: (3, H, W)
    save_path = os.path.join(output_folder, f"frame_{i:04d}.jpg")
    # save_image(frame_tensor, save_path)
    print(f"✅ Saved {save_path}")

    if i + 1 >= num_samples:
        break
