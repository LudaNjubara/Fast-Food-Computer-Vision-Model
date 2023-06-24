import os
from random import random

# Model
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights

# Image processing
from PIL import Image

# Custom imports
import utils
from constants import const


# Train the model
def train_model():
    # Initialize variables
    best_valid_loss = 0.56
    early_stop_counter = 0

    for epoch in range(const["n_epochs"]):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(const["device"]), labels.to(const["device"])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)

        # Validation phase
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            valid_correct = 0
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(const["device"]), labels.to(const["device"])
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                valid_correct += (predicted == labels).sum().item()

            valid_loss /= len(valid_dataset)
            valid_acc = valid_correct / len(valid_dataset)

            # Save the model if the validation loss has decreased
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                early_stop_counter = 0
                print(
                    'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_valid_loss,
                                                                                              valid_loss))
                save_model()

            # Early stopping if the validation loss has not decreased for 4 consecutive epochs
            else:
                early_stop_counter += 1
                if early_stop_counter >= const["early_stop"]:
                    print("Early stopping after {} epochs".format(epoch))
                    print("\nSTATS:")
                    print('Valid Loss: {:.4f}, Valid Accuracy: {:.2f}%\n'.format(valid_loss, valid_acc * 100))
                    break

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Valid Loss: {:.4f}, Valid Accuracy: {:.2f}%'
              .format(epoch + 1, const["n_epochs"], train_loss, train_acc * 100, valid_loss, valid_acc * 100))


def load_existing_model(path: str = "fast_food_model_w_resnet50_improved.pth"):
    model.load_state_dict(torch.load(path))


def save_model(path: str = "fast_food_model_w_resnet50_improved.pth"):
    torch.save(model.state_dict(), path)


def predict_many():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        images_to_show = []
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(const["device"]), labels.to(const["device"])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for j in range(images.size(0)):
                if len(images_to_show) < 20 and random() < 0.05:
                    images_to_show.append((images[j], labels[j], predicted[j].item()))

        accuracy = 100 * correct / total
        print('Test Accuracy: {:.2f} %'.format(accuracy))

        utils.show_images(images_to_show, class_names)


def predict_one(image_path: str):
    # Predict a single image
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path)
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(const["device"])
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        print('Predicted: {}'.format(class_names[predicted.item()]))

        utils.show_image(image, class_names, predicted.item())


# Main
if __name__ == "__main__":
    # Flags
    TRAINING_MODE = False
    OPTIMIZE_MODE = False

    # initialize variables
    class_names, n_classes = utils.get_classes(const["train_path"])
    single_test_image_path = const["single_test_image_path"]

    # Define the transformation function
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load the dataset
    train_dataset = datasets.ImageFolder(root=const["train_path"], transform=transform)
    valid_dataset = datasets.ImageFolder(root=const["valid_path"], transform=transform)
    test_dataset = datasets.ImageFolder(root=const["test_path"], transform=transform)

    # Load the pre-trained model
    model = torchvision.models.resnet50(weights=ResNet50_Weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_classes)

    # Move the model to the device (GPU)
    model.to(const["device"])

    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=const["batch_size"], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=const["batch_size"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=const["batch_size"], shuffle=False)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=const["learning_rate"])

    if TRAINING_MODE or OPTIMIZE_MODE:
        if OPTIMIZE_MODE:
            # Freeze the pre-trained layers and train only the last layer
            for name, param in model.named_parameters():
                if 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # Update the optimizer
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=const["learning_rate"])

            load_existing_model()

        train_model()
    else:
        load_existing_model()
        predict_one(single_test_image_path)
        # Predict multiple images from the test set
        predict_many()
