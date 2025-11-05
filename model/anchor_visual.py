import torch
import torchvision
import torch.nn as nn
import torchvision.io as io
from anchor_boxes import *
import dotenv
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

dotenv.load_dotenv()
TEST_IMG = os.getenv("TEST_IMG")

if TEST_IMG is None:
    raise ValueError("TEST_IMG environment variable is not set.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = io.read_image(TEST_IMG)
image = image.to(device).unsqueeze(0)
image = torchvision.transforms.Resize(20)(image) # 20 * 35

anchor_boxes = generate_anchor_boxes([image])

idx = []
[idx.extend(range(i * 6, (i + 1) * 6)) for i in range(0, 700, 25)]
sample_boxes = anchor_boxes[0][idx]

plt.figure(figsize=(10,10))
plt.imshow(image.cpu()[0].permute(1, 2, 0), extent=(0, 35, 20, 0))

for box in sample_boxes:
    x_min, y_min, x_max, y_max = box.cpu().numpy()
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    scaled_width = box_width * 0.25
    scaled_height = box_height * 0.25
    
    new_x_min = center_x - scaled_width / 2
    new_y_min = center_y - scaled_height / 2
    new_x_max = center_x + scaled_width / 2
    new_y_max = center_y + scaled_height / 2
    
    # convert to pixel coordinates
    width_px = (new_x_max - new_x_min) * image.shape[3]
    height_px = (new_y_max - new_y_min) * image.shape[2]
    x_px = new_x_min * image.shape[3]
    y_px = new_y_min * image.shape[2]
    
    rect = Rectangle((x_px, y_px), width_px, height_px, fill=False, color='red', linewidth=1)
    plt.gca().add_patch(rect)

plt.savefig("../img/anchor_visualization.png")