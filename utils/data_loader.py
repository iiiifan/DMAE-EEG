import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional
from utils.region import electrodes_to_patch  # Assuming this is your existing electrode grouping function


class EEGDataset(Dataset):
    def __init__(
            self,
            dataset_name: str,
            data_root: str = "./dataset",
            load_mode: str = 'memmap',
            transform=None,
            writable: bool = False
    ):
        """
        Universal EEG dataset class for loading raw and ICA processed data pairs

        Args:
            dataset_name: Name of the dataset (e.g., Cho2017_DualData)
            data_root: Root directory of datasets (default: ./dataset)
            load_mode: Data loading mode ('memmap' or 'ram')
            transform: Optional data augmentation transforms
            writable: Whether tensors need to be writable (requires memory copy)
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.load_mode = load_mode
        self.transform = transform
        self.writable = writable

        # Initialize paths
        self.dataset_path = os.path.join(data_root, dataset_name)
        self.position_file = os.path.join(self.dataset_path, "electrode_positions.csv")

        # Validate dataset directory exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")

        # Load electrode grouping information
        _, _, self.group_indices = electrodes_to_patch(self.position_file)

        # Discover data file pairs
        self.file_pairs = self._discover_file_pairs()
        if not self.file_pairs:
            raise ValueError(f"No valid data pairs found in {self.dataset_path}")

        # Pre-load data if RAM mode is selected
        self.data_cache = None
        if load_mode == 'ram':
            self._preload_data()

    def _discover_file_pairs(self) -> List[Dict[str, str]]:
        """
        Discover all valid file pairs in the dataset directory

        Returns:
            List of dictionaries containing file paths and metadata
        """
        file_pairs = []

        # Scan for raw data files (non-ICA files)
        for filename in os.listdir(self.dataset_path):
            if not filename.endswith(".npy") or "_ica" in filename:
                continue

            # Check if corresponding ICA file exists
            base_name = os.path.splitext(filename)[0]
            ica_filename = f"{base_name}_ica.npy"
            ica_path = os.path.join(self.dataset_path, ica_filename)

            if not os.path.exists(ica_path):
                continue

            # Extract subject and session information from filename
            # Expected format: subXX_sesX_*.npy
            parts = base_name.split('_')
            if len(parts) < 2:
                continue

            subject_id = parts[0][3:] if parts[0].startswith('sub') else parts[0]
            session_id = parts[1][3:] if parts[1].startswith('ses') else parts[1]

            file_pairs.append({
                'raw': os.path.join(self.dataset_path, filename),
                'ica': ica_path,
                'subject': subject_id,
                'session': session_id,
                'base_name': base_name
            })

        return sorted(file_pairs, key=lambda x: (x['subject'], x['session']))

    def _preload_data(self) -> None:
        """Preload all data pairs into memory"""
        print(f"Preloading {len(self.file_pairs)} file pairs into RAM...")
        self.data_cache = []
        for pair in self.file_pairs:
            self.data_cache.append(self._load_data_pair(pair))

    def _load_data_pair(self, pair: Dict[str, str]) -> Dict:
        """
        Load a single data pair (raw and ICA processed)

        Args:
            pair: Dictionary containing file paths and metadata

        Returns:
            Dictionary containing loaded data and metadata
        """
        # Choose loading strategy based on mode
        if self.load_mode == 'memmap':
            # Use copy-on-write mode for memory efficiency
            raw_data = np.load(pair['raw'], mmap_mode='c')
            ica_data = np.load(pair['ica'], mmap_mode='c')
        else:
            # Load directly into memory
            raw_data = np.load(pair['raw'])
            ica_data = np.load(pair['ica'])

        # Create writable copies if needed
        if self.writable:
            raw_data = raw_data.copy()
            ica_data = ica_data.copy()

        # Validate data shapes match
        if raw_data.shape != ica_data.shape:
            raise ValueError(
                f"Shape mismatch between raw {raw_data.shape} and ICA {ica_data.shape} "
                f"in {pair['base_name']}"
            )

        return {
            'raw': raw_data,
            'ica': ica_data,
            'subject': pair['subject'],
            'session': pair['session'],
            'base_name': pair['base_name'],
            'file_path': pair['raw']
        }

    def _get_data_pair(self, pair_idx: int) -> Dict:
        """Get data pair by index, using cache if available"""
        if self.data_cache is not None:
            return self.data_cache[pair_idx]
        else:
            return self._load_data_pair(self.file_pairs[pair_idx])

    def __len__(self) -> int:
        """Return total number of samples across all files"""
        if hasattr(self, '_total_samples'):
            return self._total_samples

        total = 0
        for i in range(len(self.file_pairs)):
            data = self._get_data_pair(i)
            total += data['raw'].shape[0]

        self._total_samples = total
        return total

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single sample by global index

        Args:
            idx: Global sample index across all files

        Returns:
            Dictionary containing sample data and metadata
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Find which file pair contains this sample
        cumulative_samples = 0
        for pair_idx, pair in enumerate(self.file_pairs):
            data = self._get_data_pair(pair_idx)
            num_samples = data['raw'].shape[0]

            if cumulative_samples + num_samples > idx:
                # Found the correct file pair
                local_idx = idx - cumulative_samples

                # Extract the specific sample
                raw_sample = data['raw'][local_idx]
                ica_sample = data['ica'][local_idx]

                # Convert to tensors with zero-copy when possible
                raw_tensor = torch.from_numpy(
                    raw_sample.astype(np.float32, copy=self.writable)
                )
                ica_tensor = torch.from_numpy(
                    ica_sample.astype(np.float32, copy=self.writable)
                )

                # Apply transforms if provided
                if self.transform:
                    raw_tensor = self.transform(raw_tensor)
                    ica_tensor = self.transform(ica_tensor)

                return {
                    'raw': raw_tensor,
                    'ica': ica_tensor,
                    'group_indices': self.group_indices,
                    'dataset_id': self.dataset_name,
                    'subject': data['subject'],
                    'session': data['session'],
                    'sample_idx': local_idx,
                    'global_idx': idx,
                    'file_idx': pair_idx
                }

            cumulative_samples += num_samples

        # This should never be reached due to the length check above
        raise IndexError(f"Index {idx} out of range")

    @property
    def electrode_groups(self) -> List[List[int]]:
        """Get electrode grouping information"""
        return self.group_indices

    def get_dataset_stats(self) -> Dict[str, any]:
        """
        Get comprehensive dataset statistics

        Returns:
            Dictionary containing dataset statistics
        """
        if not self.file_pairs:
            return {}

        # Get a sample to determine data dimensions
        sample_data = self._get_data_pair(0)

        # Collect unique subjects and sessions
        subjects = set(pair['subject'] for pair in self.file_pairs)
        sessions = set(pair['session'] for pair in self.file_pairs)

        stats = {
            'dataset_name': self.dataset_name,
            'num_files': len(self.file_pairs),
            'num_subjects': len(subjects),
            'num_sessions': len(sessions),
            'total_samples': len(self),
            'num_channels': sample_data['raw'].shape[1] if sample_data['raw'].ndim > 1 else 1,
            'time_points': sample_data['raw'].shape[-1],
            'num_electrode_groups': len(self.group_indices),
            'data_shape': sample_data['raw'].shape[1:],  # Shape excluding sample dimension
            'load_mode': self.load_mode,
            'subjects': sorted(subjects),
            'sessions': sorted(sessions)
        }

        return stats

    def get_subject_data(self, subject_id: str) -> List[int]:
        """
        Get all sample indices for a specific subject

        Args:
            subject_id: Subject identifier

        Returns:
            List of global sample indices for the subject
        """
        indices = []
        cumulative = 0

        for pair_idx, pair in enumerate(self.file_pairs):
            if pair['subject'] == subject_id:
                data = self._get_data_pair(pair_idx)
                num_samples = data['raw'].shape[0]
                indices.extend(range(cumulative, cumulative + num_samples))
            else:
                data = self._get_data_pair(pair_idx)
                num_samples = data['raw'].shape[0]

            cumulative += num_samples

        return indices

    def __repr__(self) -> str:
        stats = self.get_dataset_stats()
        return (
            f"EEGDataset('{self.dataset_name}', "
            f"subjects={stats.get('num_subjects', 0)}, "
            f"samples={stats.get('total_samples', 0)}, "
            f"mode='{self.load_mode}')"
        )