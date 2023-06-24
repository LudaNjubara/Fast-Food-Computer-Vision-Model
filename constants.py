from utils import get_device

const = {
    "device": get_device(),
    "train_path": "./Fast Food Classification V2/Train/",
    "valid_path": "./Fast Food Classification V2/Valid/",
    "test_path": "./Fast Food Classification V2/Test/",
    "single_test_image_path": "./Fast Food Classification V2/test_image.png",
    "image_size": 224,
    "batch_size": 64,
    "n_epochs": 15,
    "learning_rate": 0.0001,
    "l2_reg": 0.0001,
    "dropout": 0.45,
    "early_stop": 4,
}