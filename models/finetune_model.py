import torch
from torch import nn
import sys
import os

# Add project root directory to path
sys.path.append('..')

from models.model import EEGDenoisingMAE, EEGTransformerEncoder, GraphAwarePooling, SpatioTemporalPatch


class EEGFineTuner(nn.Module):
    """Fine-tuning model based on MAE encoder with configurable number of classes"""

    def __init__(self, config, device='cpu'):
        super().__init__()

        self.device = device
        self.config = config

        # Extract parameters from configuration
        model_cfg = config.model_config
        self.spatial_groups = model_cfg['spatial_groups']
        self.temporal_dim = model_cfg['temporal_dim']
        self.num_classes = model_cfg.get('num_classes', 2)  # Default to 2 classes

        # Encoder components - Copy MAE structure
        self.patch_embed = nn.Linear(self.temporal_dim, model_cfg['vit_dim'])
        self.pos_embedding = nn.Parameter(torch.randn(1, model_cfg['num_patches'] + 1, model_cfg['vit_dim']))
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_cfg['vit_dim']))
        self.input_norm = nn.LayerNorm(model_cfg['vit_dim'])

        self.encoder = EEGTransformerEncoder(
            dim=model_cfg['vit_dim'],
            depth=model_cfg['vit_depth'],
            heads=model_cfg['vit_heads'],
            dim_head=model_cfg['vit_dim'] // model_cfg['vit_heads'],
            mlp_dim=model_cfg['vit_mlp_dim']
        )

        # Preprocessing components
        self.graph_pool = GraphAwarePooling()
        self.st_patch = SpatioTemporalPatch(self.graph_pool)

        # Classification head - New trainable part
        if self.num_classes == 2:
            # Binary classification - single output with sigmoid
            self.classifier = nn.Sequential(
                nn.LayerNorm(model_cfg['vit_dim']),
                nn.Linear(model_cfg['vit_dim'], model_cfg['vit_dim'] // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(model_cfg['vit_dim'] // 2, model_cfg['vit_dim'] // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(model_cfg['vit_dim'] // 4, 1)  # Single output for binary classification
            )
        else:
            # Multi-class classification
            self.classifier = nn.Sequential(
                nn.LayerNorm(model_cfg['vit_dim']),
                nn.Linear(model_cfg['vit_dim'], model_cfg['vit_dim'] // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(model_cfg['vit_dim'] // 2, self.num_classes)  # Configurable number of classes
            )

        # Freeze encoder based on configuration
        if config.training_config['freeze_encoder']:
            self._freeze_encoder()

    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        frozen_components = [
            self.patch_embed, self.input_norm, self.encoder,
            self.graph_pool, self.st_patch
        ]

        for component in frozen_components:
            for param in component.parameters():
                param.requires_grad = False

        # Freeze positional embedding and CLS token
        self.pos_embedding.requires_grad = False
        self.cls_token.requires_grad = False

        print("Encoder parameters frozen")

    def load_mae_weights(self, mae_state_dict):
        """Load pre-trained weights from MAE model"""
        print("Loading MAE pre-trained weights...")

        # Create temporary MAE model to load weights
        mae_model = EEGDenoisingMAE(
            dataset_names=self.config.model_config['dataset_names'],
            num_patches=self.config.model_config['num_patches'],
            vit_dim=self.config.model_config['vit_dim'],
            vit_depth=self.config.model_config['vit_depth'],
            vit_heads=self.config.model_config['vit_heads'],
            vit_mlp_dim=self.config.model_config['vit_mlp_dim'],
            masking_ratio=self.config.model_config['masking_ratio'],
            decoder_dim=self.config.model_config['decoder_dim'],
            decoder_depth=self.config.model_config['decoder_depth'],
            device=self.device
        )

        # Load weights to temporary model
        missing_keys, unexpected_keys = mae_model.load_state_dict(mae_state_dict, strict=False)

        # Copy weights to fine-tuning model
        try:
            self.patch_embed.load_state_dict(mae_model.patch_embed.state_dict())
            self.pos_embedding.data.copy_(mae_model.pos_embedding.data)
            self.cls_token.data.copy_(mae_model.cls_token.data)
            self.input_norm.load_state_dict(mae_model.input_norm.state_dict())
            self.encoder.load_state_dict(mae_model.encoder.state_dict())
            self.graph_pool.load_state_dict(mae_model.graph_pool.state_dict())
            self.st_patch.load_state_dict(mae_model.st_patch.state_dict())

            print("MAE weights loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
        finally:
            del mae_model

    def forward(self, eeg_signals, electrode_groups):
        """Forward propagation"""
        # Generate spatio-temporal patches
        patches = self.st_patch(eeg_signals, electrode_groups)

        # Embedding and encoding
        embeddings = self.patch_embed(patches)
        embeddings = self.input_norm(embeddings)

        # Add positional encoding
        if patches.size(1) <= self.pos_embedding.size(1) - 1:
            embeddings = embeddings + self.pos_embedding[:, 1:patches.size(1) + 1]

        # Transformer encoding
        encoded_features = self.encoder(embeddings)

        # Global average pooling
        global_features = encoded_features.mean(dim=1)

        # Classification
        logits = self.classifier(global_features)

        return logits

    def predict_proba(self, eeg_signals, electrode_groups):
        """Get prediction probabilities"""
        with torch.no_grad():
            logits = self.forward(eeg_signals, electrode_groups)
            if self.num_classes == 2:
                # Binary classification - use sigmoid
                probabilities = torch.sigmoid(logits)
                # Return probabilities for both classes
                probs_class_0 = 1 - probabilities
                probs_class_1 = probabilities
                return torch.cat([probs_class_0, probs_class_1], dim=1)
            else:
                # Multi-class classification - use softmax
                probabilities = torch.softmax(logits, dim=1)
                return probabilities

    def predict(self, eeg_signals, electrode_groups, threshold=0.5):
        """Get predictions"""
        with torch.no_grad():
            logits = self.forward(eeg_signals, electrode_groups)
            if self.num_classes == 2:
                # Binary classification
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > threshold).long().squeeze()
            else:
                # Multi-class classification
                predictions = torch.argmax(logits, dim=1)
            return predictions

    def get_trainable_parameters(self):
        """Get trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]