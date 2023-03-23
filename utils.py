from os import listdir

import cv2
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import plotly.express as px
import numpy as np
from matplotlib import pyplot as plt


def get_classes(train_path: str):
    class_names = sorted(listdir(train_path))
    n_classes = len(class_names)

    return class_names, n_classes


def calculate_class_distribution(path: str, class_names: list):
    class_dis = [len(listdir(path + name)) for name in class_names]

    return class_dis


# Function that visualizes the class distribution for training and validation data
def visualize_class_distribution(class_names: list, class_dis: list, title: str):
    fig = px.pie(names=class_names, values=class_dis, hole=0.3)
    fig.update_layout(
        {
            "title":
                {
                    'text': title,
                    'x': 0.48
                }
        }
    )
    fig.show()

    fig = px.bar(x=class_names, y=class_dis, color=class_names)
    fig.show()


def load_image(image_path: str, image_size: int, device: torch.device):
    # load image using cv2
    image = cv2.imread(image_path)

    # convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize image
    image = cv2.resize(image, (image_size, image_size))

    # convert image to tensor
    image = transforms.ToTensor()(image).to(device)

    # add batch dimension
    image = image.unsqueeze(0)

    return image


def get_data_generator(path: str, batch_size: int, image_size: int):
    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    data = ImageFolder(path, transform=data_transform)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return data_loader


def predict_one_image(model, image_path: str, class_names: list, image_size: int, device: torch.device):
    model.eval()
    with torch.no_grad():
        output = model(load_image(image_path, image_size, device))
        _, preds = torch.max(output, 1)
        prediction = class_names[preds[0]]
        accuracy = torch.nn.functional.softmax(output, dim=1)[0] * 100
        actual = image_path.split('/')[-2]

        return prediction, actual, accuracy[preds[0]].item()


# Function that visualizes some images
def show_images(data, class_names, model=None):
    # Get images and labels from data
    images, labels = next(iter(data))

    # Convert images to numpy
    images = images.numpy()

    # Plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, int(20 / 2), idx + 1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title(class_names[labels[idx]])

        # model prediction
        if model is not None:
            model.eval()
            with torch.no_grad():
                output = model(images)
                _, preds = torch.max(output, 1)
                ax.set_title("{} ({})".format(class_names[preds[idx]], class_names[labels[idx]]),
                             color=("green" if preds[idx] == labels[idx].item() else "red"))
        else:
            ax.set_title(class_names[labels[idx]])

    plt.show()
