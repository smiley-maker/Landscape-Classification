#!/usr/bin/env python3

from src.utils.dependencies import *

class TrainCNN:
    def __init__(self,
                 model : nn.Module,
                 optimizer,
                 epochs : int,
                 batch_size : int,
                 data,
                 device,
                 write_results_to : str,
                 extra_losses = None
                ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = data
        self.device = device
        self.extra_losses = extra_losses
        self.writer = SummaryWriter(log_dir=write_results_to)

        self.criterion = nn.CrossEntropyLoss()
        self.labels = ["Coast", "Desert", "Forest", "Glacier", "Mountain"]


    def train_model(self):
        print(f"Training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            overall_loss = 0

            self.model.train()
            self.model.zero_grad()

            for batch_num, x in enumerate(self.data):
                y_true = x[1].to(self.device).long()
                x = x[0].to(self.device)
            
                self.optimizer.zero_grad()

                y_hat = self.model(x)
                loss = self.criterion(y_hat, y_true)

                if self.extra_losses != None:
                    for l in self.extra_losses:
                        loss += l(y_true, y_hat)

                overall_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            # Log loss to TensorBoard
            self.writer.add_scalar("Loss", overall_loss/(batch_num*self.batch_size), epoch)
            print(f"Epoch {epoch}: {overall_loss/(batch_num)}")

            # Log a batch of images with predicted labels to tensorboard. 
            # I think we can just use the last batch of images and labels. 
            with torch.no_grad():
                y_pred = torch.argmax(y_hat, dim=1)

                # Take the images and move them to CPU
                images = x.cpu()
                preds = y_pred.cpu()

                # Number of images in the batch
                num_images = images.size(0)
                num_cols = 4  # number of columns in the grid
                num_rows = (num_images + num_cols - 1) // num_cols

                # Create a matplotlib figure
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

                # If batch size is smaller than grid, flatten axes properly
                axes = axes.flatten()

                for img, pred, ax in zip(images, preds, axes):
                    # img shape: [C, H, W] --> convert to numpy
                    img = img.permute(1, 2, 0).numpy()

                    # Normalize image for display
                    img = (img - img.min()) / (img.max() - img.min())

                    ax.imshow(img)
                    ax.set_title(f'Pred: {self.labels[pred.item()]}')
                    ax.axis('off')

                # Hide any leftover axes if batch is small
                for ax in axes[num_images:]:
                    ax.axis('off')

                # Add to TensorBoard
                self.writer.add_figure("Predictions", fig, epoch)

                plt.close(fig)  # Close the figure to free memory
        
        self.writer.close()
        return overall_loss