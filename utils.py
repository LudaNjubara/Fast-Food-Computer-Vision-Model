import numpy as np
import torch
from os import listdir

from matplotlib import pyplot as plt


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


def show_image(image, class_names, predicted_item):
    # show image and label and prediction without numbered axes
    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    height, width, _ = image.shape
    plt.imshow(image, aspect=width / height)
    plt.title("Predicted: {}".format(class_names[predicted_item]))
    plt.axis('off')
    plt.show()


def show_images(images_to_show, class_names, num_cols=4):
    # show images and label and prediction in a grid
    num_images = len(images_to_show)
    num_rows = int(np.ceil(num_images / num_cols))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axs = axs.flatten()

    for i, (image, label, predicted_item) in enumerate(images_to_show):
        image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        height, width, _ = image.shape
        axs[i].imshow(image, aspect=width / height)
        axs[i].set_title("Label: {}\nPrediction: {}".format(class_names[label], class_names[predicted_item]))
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
