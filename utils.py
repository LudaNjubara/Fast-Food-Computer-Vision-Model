import glob
from os import listdir

import cv2
import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import plotly.express as px


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


# Function that calculates the class distribution for training and validation data
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


# Function that loads an image and returns a tensor of the image and its label
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


# Function that loads all images from a folder and returns a list of tensors of the images and their labels
def load_images_from_folder(folder, image_size: int, device: torch.device):
    images = []
    for filename in glob.glob(folder + '*.jpeg'):
        img = load_image(filename, image_size, device)
        if img is not None:
            images.append(img)

    return images


# Function that returns the data in the form of a data loader so that it can be used for training and validation
def get_data_generator(path: str, batch_size: int, image_size: int) -> DataLoader:
    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    data = ImageFolder(path, transform=data_transform)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return data_loader


# Function that performs a prediction on a single image and returns the prediction and prediction accuracy
def predict(model, image, class_names: list) -> [str, str, float]:
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, preds = torch.max(output, 1)
        prediction = class_names[preds[0]]
        accuracy = torch.nn.functional.softmax(output, dim=1)[0] * 100

        return prediction, accuracy[preds[0]].item()
