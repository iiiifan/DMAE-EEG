import torch
from torch import nn
from torch.utils.data import DataLoader
from config import FineTuneConfig, EEGDataset, load_mae_checkpoint
from models.finetune_model import EEGFineTuner


class TrainingConfig:
    """Training configuration class - required for checkpoint loading compatibility"""

    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FineTuneTrainer:
    """Fine-tuning trainer - simplified version"""

    def __init__(self, config=None):
        self.config = config or FineTuneConfig()
        self.device = self.config.device

        # Setup model
        self.model = EEGFineTuner(self.config, self.device).to(self.device)

        # Setup loss function for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Setup optimizer
        trainable_params = self.model.get_trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.training_config['learning_rate'],
            weight_decay=self.config.training_config['weight_decay']
        )

    def load_pretrained_weights(self, checkpoint_path):
        """Load pre-trained MAE weights"""
        mae_state_dict, load_success = load_mae_checkpoint(checkpoint_path, self.device)
        if load_success:
            self.model.load_mae_weights(mae_state_dict)
            print("Pre-trained weights loaded successfully")
        else:
            print("Failed to load pre-trained weights")

    def setup_data(self, data_path=None, mode='dummy'):
        """Setup data loader"""
        dataset = EEGDataset(data_path, mode, num_classes=2)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.training_config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        self.electrode_groups = dataset.electrode_groups

    def forward_and_loss(self, signals, labels):
        """Forward process and loss calculation"""
        # Forward propagation
        logits = self.model(signals, self.electrode_groups)

        # Prepare labels for binary classification
        labels = labels.float().unsqueeze(1)

        # Calculate loss
        loss = self.criterion(logits, labels)

        return logits, loss

    def train_step(self, signals, labels):
        """Single training step"""
        self.model.train()

        # Forward pass and loss calculation
        logits, loss = self.forward_and_loss(signals, labels)

        # Backward propagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_parameters(),
            max_norm=self.config.training_config['max_grad_norm']
        )

        self.optimizer.step()

        return loss.item()


def train_finetune_model(checkpoint_path=None, epochs=10):
    """Simple fine-tuning training function"""

    # Create trainer
    trainer = FineTuneTrainer()

    # Load pre-trained weights
    if checkpoint_path:
        trainer.load_pretrained_weights(checkpoint_path)

    # Setup data
    trainer.setup_data(mode='dummy')

    print("Starting fine-tuning training...")

    # Training loop
    for epoch in range(epochs):
        epoch_losses = []

        for batch_idx, batch_data in enumerate(trainer.train_loader):
            signals = batch_data['signal'].to(trainer.device).float()
            labels = torch.tensor([
                batch_data['label'][i] for i in range(len(batch_data['label']))
            ]).to(trainer.device)

            # Training step
            loss = trainer.train_step(signals, labels)
            epoch_losses.append(loss)

            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss:.4f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}")

    return trainer.model


if __name__ == "__main__":
    # Usage example
    checkpoint_path = "ckpt/pretrained/epoch_test.ckpt"

    model = train_finetune_model(
        checkpoint_path=checkpoint_path,
        epochs=5
    )

    print("Fine-tuning completed!")