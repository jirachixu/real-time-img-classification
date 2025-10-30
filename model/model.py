import torch
import torchvision
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")