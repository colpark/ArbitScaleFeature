import glob
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    A base class for fMRI datasets. Handles argument registration, data scaling,
    sequence loading, and DDP-safe caching of the data index.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration), 1)
        # The 'distributed' flag is added here for DDP-safe caching
        self.distributed = dist.is_available() and dist.is_initialized()
        self.data = self._set_data()

    def register_args(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs

    def scale_input(self, y, subject_path):
        """Applies normalization to the input tensor based on pre-computed stats."""
        if self.input_scaling_method == 'none':
            return y
        
        stats_path = os.path.join(subject_path, 'global_stats.pt')
        if not os.path.exists(stats_path):
            # If stats are required but not found, return original or raise error
            print(f"Warning: Stats file not found at {stats_path}. Returning unscaled data.")
            return y
            
        stats_dict = torch.load(stats_path)
        background = (y == 0)

        if self.input_scaling_method == 'minmax':
            # Use .get() for safety in case key is missing
            y = y / stats_dict.get('global_max', 1.0)
        elif self.input_scaling_method in ['znorm_zeroback', 'znorm_minback']:
            y = (y - stats_dict['global_mean']) / stats_dict['global_std']
            if self.input_scaling_method == 'znorm_zeroback':
                y[background] = 0
        elif self.input_scaling_method == 'robust':
            y = (y - stats_dict['median']) / stats_dict['iqr']
        return y

    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None):
        """Loads a sequence of fMRI frames from individual .pt files."""
        if self.shuffle_time_sequence:
            # Ensure we don't sample more frames than available
            num_to_sample = sample_duration // self.stride_within_seq
            frames_to_load = random.sample(range(num_frames), k=num_to_sample)
        else:
            frames_to_load = range(start_frame, start_frame + sample_duration, self.stride_within_seq)

        fnames = [f'frame_{frame}.pt' for frame in frames_to_load]
        
        loaded_frames = [torch.load(os.path.join(subject_path, fname)).unsqueeze(0) for fname in fnames]
        y = torch.cat(loaded_frames, dim=4)
        y = self.scale_input(y, subject_path)
        return y

    def __len__(self):
        return len(self.data)

    def _get_cache_filepath(self) -> str:
        """Generates a unique, descriptive filepath for the dataset cache CSV."""
        image_name = Path(self.root).name
        filename = (
            f"{self.downstream_task}_{image_name}_{self.split}"
            f"_seqlen{self.sequence_length}_within{self.stride_within_seq}"
            f"_between{self.stride_between_seq}.csv"
        )
        if self.split == 'train' and self.num_train_fMRI_segments is not None:
            filename = filename.replace('.csv', f'_segments{self.num_train_fMRI_segments}.csv')
        
        cache_dir = Path("./data/data_tuple/")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / filename)

    def _load_data_from_cache(self):
        """Tries to load the data list from a cached CSV file."""
        if not self.use_subj_dict:
            return None
            
        cache_file = self._get_cache_filepath()
        if os.path.exists(cache_file):
            print(f"Loading cached dataset list from {cache_file}")
            df = pd.read_csv(cache_file)
            df.subject = df.subject.astype(str)
            # Ensure the cache belongs to the same set of subjects
            if set(df.subject) == set(self.subject_dict.keys()):
                return df.values.tolist()
            else:
                print("Cache is stale (subject mismatch). Regenerating...")
        return None

    def _save_data_to_cache(self, data):
        """Saves the data list to a CSV file from rank 0 only."""
        if not self.use_subj_dict:
            return

        cache_file = self._get_cache_filepath()
        # Only rank 0 should write the file to prevent race conditions
        if not self.distributed or dist.get_rank() == 0:
            df = pd.DataFrame(data, columns=['i', 'subject', 'subject_path', 'start_frame', 'sample_duration', 'num_frames', 'target', 'sex'])
            df.to_csv(cache_file, index=False)
            print(f"[RANK 0] Saved dataset cache to {cache_file}")
        
        # All processes wait here to ensure the file is written before proceeding.
        if self.distributed:
            dist.barrier()

    def _make_data_tuple_list(self, subject_path, i, subject_name, target, sex):
        """Creates a list of all possible fMRI segments for a single subject."""
        num_frames = len(glob.glob(os.path.join(subject_path, 'frame_*.pt')))
        session_duration = num_frames - self.sample_duration + 1
        
        if self.train and self.num_train_fMRI_segments is not None:
            max_segments = session_duration // self.stride
            num_segments = min(self.num_train_fMRI_segments, max_segments)
            session_duration = num_segments * self.stride

        start_frames = []
        for start_offset in range(self.stride_within_seq):
            start_frames.extend(range(start_offset, session_duration, self.stride))
            
        return [(i, subject_name, subject_path, sf, self.sample_duration, num_frames, target, sex) for sf in start_frames]

    def __getitem__(self, index):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def _set_data(self):
        raise NotImplementedError("This method must be implemented by a subclass.")

class S1200(BaseDataset):
    """Dataset class for the S1200 dataset."""
    def _set_data(self):
        # 1. Try loading from cache first
        data = self._load_data_from_cache()
        
        # 2. If cache miss, generate data from scratch
        if data is None:
            data = []
            img_root = os.path.join(self.root, 'img')
            print("Generating dataset list from scratch...")
            for i, subject_name in enumerate(self.subject_dict):
                sex, target = self.subject_dict[subject_name]
                subject_path = os.path.join(img_root, str(subject_name))
                if not os.path.isdir(subject_path):
                    print(f"Warning: Subject directory not found: {subject_path}")
                    continue
                data.extend(self._make_data_tuple_list(subject_path, i, subject_name, target, sex))
            
            # 3. Save to cache for next time
            if self.limit_samples is None: # Only cache if not using a limited subset
                self._save_data_to_cache(data)

        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)
        
        # Note: Padding is hardcoded. Consider making this configurable.
        background_value = y.flatten()[0]
        y = y.permute(0, 4, 1, 2, 3) # To (N, T, H, W, D) -> (N, D, T, H, W)
        y = torch.nn.functional.pad(y, (3, 9, 0, 0, 10, 8), value=background_value)
        y = y.permute(0, 2, 3, 4, 1) # Back to (N, T, H, W, D)

        return {
            "fmri_sequence": y, "subject_name": subject_name,
            "target": target, "TR": start_frame, "sex": sex
        }

class ConcatDataset(Dataset):
    """
    Custom ConcatDataset that also concatenates the 'target_values' attribute
    for tasks like regression.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(d) for d in datasets])
        if hasattr(datasets[0], "target_values"):
            self.target_values = np.concatenate([d.target_values for d in datasets if hasattr(d, 'target_values')], axis=0)

    def __len__(self):
        return self.cumulative_sizes[-1] if len(self.cumulative_sizes) > 0 else 0

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        # Find which dataset the index belongs to
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        # Find the index within that dataset
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
