import torch
from os import listdir


# Function that returns the device (CPU or GPU)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


# Function that returns the class names and the number of classes
def get_classes(train_path: str):
    class_names = sorted(listdir(train_path))
    n_classes = len(class_names)

    return class_names, n_classes
