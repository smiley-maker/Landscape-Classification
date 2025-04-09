#!/usr/bin/env python3

from src.utils.dependencies import *
from src.architectures.cnn import CNN
from src.data_handlers.landscapes12k_handler import LandscapesDataset
from src.trainers.train_cnn import TrainCNN

if __name__ == "__main__":
    # Parameters
    batch_size = 32
    num_epochs = 100

    # Create device
    print("Creating device")
    device = torch.device("mps")
    print(device.type)

    # Create CNN model
    print("Creating model")
    model = CNN().to(device)
    print("Created model")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    print("Created optimizer")

    # Create datasets for training, testing, and validation
    train_data = LandscapesDataset(img_size=64)
#    test_data = LandscapesDataset(image_dir="/Users/jordan/Data/Landscapes12k/Test/")
#    val_data = LandscapesDataset(image_dir="/Users/jordan/Data/Landscapes12k/Val/")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
#    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    print(f"Created dataset with length {len(train_data)}")

    # Create trainer
    trainer = TrainCNN(
        model=model,
        epochs=num_epochs,
        batch_size=batch_size,
        data=train_loader,
        optimizer=optimizer,
        device=device,
        write_results_to="src/results/training/100epochsCNN"
    ) 

    print("Created trainer")

    print("training")
    trainer.train_model()