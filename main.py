# Global
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Utils
import utils

# Model
from classifiers import FastFoodClassifier

# Constants
from constants import const


def train_model():
    # Initialize the best validation loss to a large value
    best_valid_loss = float('inf')
    # best_valid_loss = 1.43
    early_stop_counter = 0

    # Train the model
    for epoch in range(const["n_epochs"]):
        # Initialize the loss
        train_loss = 0.0
        valid_loss = 0.0

        # Train the model
        model.train()
        for data, target in train_data:
            # Move the data and target to the device
            data, target = data.to(const["device"]), target.to(const["device"])

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)
            loss += model.l2_loss() * model.l2_reg

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Calculate the loss
            train_loss += loss.item() * data.size(0)

        # Validate the model
        model.eval()
        with torch.no_grad():
            for data, target in valid_data:
                # Move the data and target to the device
                data, target = data.to(const["device"]), target.to(const["device"])

                # Forward pass
                output = model(data)

                # Calculate the loss
                loss = criterion(output, target)

                # Calculate the loss
                valid_loss += loss.item() * data.size(0)

        # Calculate the average losses
        train_loss = train_loss / len(train_data.dataset)
        valid_loss = valid_loss / len(valid_data.dataset)

        # Print the progress
        print("Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(epoch + 1, train_loss, valid_loss))

        # Step the scheduler
        scheduler.step(valid_loss)

        # Save the model if validation loss has decreased
        if valid_loss <= best_valid_loss:
            print("Validation loss decreased from {:.6f} to {:.6f}. Saving model...".format(best_valid_loss, valid_loss))
            best_valid_loss = valid_loss
            early_stop_counter = 0
            save_model()
        else:
            early_stop_counter += 1
            if early_stop_counter >= 10 and epoch >= 25:
                print('Early stopping after {} epochs'.format(epoch + 1))
                break

    print("Training complete.")


def save_model(path: str = "fast_food_classifier.pth"):
    torch.save(model.state_dict(), path)


def load_model(path: str = "fast_food_classifier.pth"):
    model.load_state_dict(torch.load(path))


def predict_one(image_path: str):
    print("Predicting image...", end="\n")

    image = utils.load_image(image_path, const["image_size"], const["device"])
    prediction, accuracy = utils.predict(model, image, class_names)

    print(f"Prediction: {prediction}, Accuracy: {round(accuracy, 2)}%", end="\n\n")


def predict_many(path: str):
    print("Predicting many images...", end="\n")

    avg_accuracy = 0.0
    images = utils.load_images_from_folder(path, const["image_size"], const["device"])

    for image in images:
        prediction, accuracy = utils.predict(model, image, class_names)
        avg_accuracy += accuracy
        print(f"Prediction: {prediction}, Accuracy: {round(accuracy, 2)}%")

    avg_accuracy /= len(images)
    print(f"\nAverage accuracy: {round(avg_accuracy, 2)}%", end="\n\n")


# Main
if __name__ == "__main__":
    # Flags
    TRAINING_MODE = False
    OPTIMIZE_MODE = False

    # Get the class names
    class_names, n_classes = utils.get_classes(const["train_path"])

    # Paths
    single_test_image_path = "Fast Food Classification V2/test_image.jpg"
    many_test_image_path = const["test_path"] + "Burger/"

    # Initialize the model
    model = FastFoodClassifier(n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=const["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6)
    model.to(const["device"])

    # Set up the data generators
    train_data = utils.get_data_generator(const["train_path"], const["batch_size"], const["image_size"])
    valid_data = utils.get_data_generator(const["valid_path"], const["batch_size"], const["image_size"])

    if TRAINING_MODE:
        if OPTIMIZE_MODE:
            # Load the model
            load_model()

        # Train the model
        train_model()

        # Save the model
        save_model()
    else:
        # Load the model
        load_model()

        # Predict a single image
        predict_one(single_test_image_path)

        # Predict multiple images
        predict_many(many_test_image_path)
