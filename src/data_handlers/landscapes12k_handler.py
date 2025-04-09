#!/usr/bin/env python3

from src.utils.dependencies import *

LANDSCAPE_TRAIN_IMAGES_DIR = "/Users/jordan/Data/Landscapes12k/Train/"
LANDSCAPE_TEST_IMAGES_DIR = "/Users/jordan/Data/Landscapes12k/Test/"
LANDSCAPE_VAL_IMAGES_DIR = "/Users/jordan/Data/Landscapes12k/Val/"

class LandscapesDataset(Dataset):
    def __init__(self, img_size=128, image_dir="/Users/jordan/Data/Landscapes12k/Train/"):
        super().__init__()
        self.img_size = img_size
        self.img_dir = image_dir
        self.dataset = ImageFolder(self.img_dir,transform = transforms.Compose(
            [transforms.Resize((self.img_size, self.img_size)),transforms.ToTensor()]
        ))
        self.labels = ["Coast", "Desert", "Forest", "Glacier", "Mountain"]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]


if __name__ == "__main__":
    print("Data Handler was run directly, initiating example...")
    dataset = LandscapesDataset()

    print("Selecting a random image from the dataset...")
    x = dataset[random.randrange(0, len(dataset))]

    plt.figure(figsize=(5, 5))
    plt.imshow(x[0].permute(1, 2, 0).numpy())
    plt.title(f"{dataset.labels[x[1]]} Class")
    plt.show()