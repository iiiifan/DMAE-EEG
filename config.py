import torch
from torch.utils.data import Dataset
import os


class FineTuneConfig:
    """Fine-tuning training configuration"""

    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model configuration
        self.model_config = {
            'num_patches': 40,
            'vit_dim': 768,
            'vit_depth': 8,
            'vit_heads': 16,
            'vit_mlp_dim': 768,
            'masking_ratio': 0.75,
            'decoder_dim': 512,
            'decoder_depth': 4,
            'temporal_dim': 100,
            'spatial_groups': 10,
            'dataset_names': ["PhysionetMI"],
            'num_classes': 2,  # Default to binary classification
        }

        # Training configuration
        self.training_config = {
            'batch_size': 8,
            'epochs': 10,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'step_size': 5,
            'gamma': 0.8,
            'max_grad_norm': 1.0,
            'freeze_encoder': True,
            'print_freq': 5
        }

        # Path configuration
        self.paths = {
            'checkpoint': "./ckpt/pretrained/epoch_test.ckpt",
            'save_dir': "./ckpt",
            'save_name': "finetuned_model.pth"
        }


class EEGDataset(Dataset):
    """EEG dataset base class"""

    def __init__(self, data_path=None, mode='dummy', num_classes=2):
        self.mode = mode
        self.data_path = data_path
        self.num_classes = num_classes

        # Electrode grouping configuration
        self.electrode_groups = [
            [36, 2, 1, 32, 0],
            [36, 35, 34, 32, 33],
            [37, 3, 4, 5, 6, 46, 10, 9, 8, 7],
            [37, 38, 39, 40, 41, 46, 45, 44, 43, 42],
            [47, 11, 12, 13, 14],
            [47, 48, 49, 50, 51],
            [30, 19, 20, 21, 22, 23],
            [30, 56, 57, 58, 59],
            [28, 26, 29, 25, 24],
            [28, 63, 29, 62, 61]
        ]

        if mode == 'dummy':
            self._init_dummy_data()
        else:
            self._load_real_data()

    def _init_dummy_data(self):
        """Initialize dummy data"""
        self.length = 100

    def _load_real_data(self):
        """Load real data - implement real data loading logic here"""
        # TODO: Implement real data loading
        self.length = 200

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'dummy':
            return {
                'signal': torch.randn(64, 400),
                'label': torch.randint(0, self.num_classes, (1,)).item(),
            }
        else:
            # TODO: Implement real data retrieval
            return {
                'signal': torch.randn(64, 400),
                'label': torch.randint(0, self.num_classes, (1,)).item(),
            }


def load_mae_checkpoint(checkpoint_path, device):
    """Load MAE checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print("Checkpoint file does not exist")
        return None, False

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract state_dict
        state_dict = None
        if isinstance(checkpoint, dict):
            for key in ['model_state_dict', 'state_dict', 'model']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break

        if state_dict is not None:
            print("Successfully loaded checkpoint")
            return state_dict, True
        else:
            print("No valid state_dict found")
            return None, False

    except Exception as e:
        print(f"Loading failed: {e}")
        return None, False