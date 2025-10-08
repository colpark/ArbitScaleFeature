import os
import pickle
import random
import yaml
from argparse import Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# Assuming hcp_datasets.py is in the same directory or in the python path
from datasets import register
from .fmri_datasets import S1200, S1200test, ConcatDataset

@register('fmri_datamodule')
class DataModule:
    """
    Organizes the data loading pipeline by reading a YAML configuration file.
    Handles dataset creation, splitting, and DDP-aware DataLoader instantiation.
    """
    def __init__(self, config_path: str):
        """
        Initializes the DataModule by loading a configuration from a YAML file.

        Args:
            config_path (str): The path to the data_module_cfg.yaml file.
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = Namespace(**config_dict)
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.distributed = dist.is_available() and dist.is_initialized()

    def setup(self, stage: str = None):
        """Main setup function to prepare datasets for all splits."""
        base_params = self._get_base_params()
        dataset_splits = {'train': [], 'val': [], 'test': []}
        
        for dataset_name, image_path in zip(self.config.dataset_name, self.config.image_path):
            DatasetClass = self._get_dataset_class(dataset_name)
            subject_dict = self.make_subject_dict(dataset_name, image_path)

            split_path = self.define_split_file_path(dataset_name, self.config.dataset_split_num)
            train_names, val_names, test_names = self.load_or_create_split(subject_dict, split_path)
            
            '''
            Original code
            all_splits = {
                'train': {key: subject_dict[key] for key in train_names if key in subject_dict},
                'val': {key: subject_dict[key] for key in val_names if key in subject_dict},
                'test': {key: subject_dict[key] for key in test_names if key in subject_dict},
            }

            for split_name, data_dict in all_splits.items():
                limit = getattr(self.config, f'limit_{split_name}ing_samples', None)
                sampled_dict = self._sample_subject_dict(data_dict, limit)
            
            '''

            ##########
            #Changed to use subject limit number properly
            limit_names = {'train':'training', 'val':'validation', 'test':'test'}
            all_splits = {
                'train': {key: subject_dict[key] for key in train_names if key in subject_dict},
                'val': {key: subject_dict[key] for key in val_names if key in subject_dict},
                'test': {key: subject_dict[key] for key in test_names if key in subject_dict},
            }

            for split_name, data_dict in all_splits.items():
                limit = getattr(self.config, f'limit_{limit_names[split_name]}_samples', None)
                sampled_dict = self._sample_subject_dict(data_dict, limit)
            ###########
                
                params = {**base_params, "root": image_path, "subject_dict": sampled_dict, "train": (split_name == 'train'), "split": split_name, "limit_samples": limit}
                dataset_splits[split_name].append(DatasetClass(**params))
                print(f"Number of subjects for {dataset_name} '{split_name}': {len(sampled_dict)}")

        self.train_dataset = ConcatDataset(dataset_splits['train'])
        self.val_dataset = ConcatDataset(dataset_splits['val'])
        self.test_dataset = ConcatDataset(dataset_splits['test'])

        print(f"\nTotal training segments: {len(self.train_dataset)}")
        print(f"Total validation segments: {len(self.val_dataset)}")
        print(f"Total test segments: {len(self.test_dataset)}")

    def _create_dataloader(self, dataset, batch_size, is_train):
        """
        Helper function to create a DataLoader with or without a DistributedSampler.
        """
        sampler = None
        # The sampler handles shuffling in DDP, so shuffle must be False in DataLoader
        shuffle = is_train 

        if self.distributed:
            # Create a sampler for the given dataset.
            # shuffle=True for training to randomize data order each epoch.
            # shuffle=False for val/test to ensure consistent evaluation order.
            sampler = DistributedSampler(dataset, shuffle=is_train)
            # When using a sampler, the shuffle argument to DataLoader must be False.
            shuffle = False
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            drop_last=is_train # Drop last incomplete batch only for training
        )

    def train_dataloader(self):
        """Creates the DataLoader for the training set."""
        return self._create_dataloader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            is_train=True
        )

    def val_dataloader(self):
        """Creates the DataLoader for the validation set."""
        return self._create_dataloader(
            self.val_dataset, 
            batch_size=self.config.eval_batch_size, 
            is_train=False
        )

    def test_dataloader(self):
        """Creates the DataLoader for the test set."""
        return self._create_dataloader(
            self.test_dataset, 
            batch_size=self.config.eval_batch_size, 
            is_train=False
        )

    # --- Helper methods for data setup (unchanged from previous version) ---

    def _get_dataset_class(self, dataset_name):
        if dataset_name == "S1200": return S1200
        elif dataset_name == "S1200test": return S1200test
        raise NotImplementedError(f"Dataset '{dataset_name}' not implemented.")

    def _get_base_params(self) -> Dict:
        param_keys = ["sequence_length", "stride_between_seq", "stride_within_seq", "downstream_task", "shuffle_time_sequence", "input_scaling_method", "label_scaling_method", "num_train_fMRI_segments", "use_subj_dict"]
        return {key: getattr(self.config, key) for key in param_keys}

    def _sample_subject_dict(self, data_dict: Dict, limit: float = None) -> Dict:
        if limit is None: return data_dict
        num_total = len(data_dict)
        num_to_select = int(limit) if limit >= 1.0 else int(num_total * limit)
        num_to_select = max(1, min(num_total, num_to_select))
        if num_to_select < num_total:
            print(f"Randomly sampling {num_to_select} of {num_total} subjects.")
            keys = list(data_dict.keys())
            sampled_keys = random.sample(keys, num_to_select)
            return {key: data_dict[key] for key in sampled_keys}
        return data_dict

    def _make_s1200_dict(self, image_path: str, available_subjects: set) -> Dict:
        """Creates the subject dictionary specifically for the S1200 dataset."""
        meta_path = os.path.join(image_path, "metadata", "HCP_1200_precise_age.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        meta_df = pd.read_csv(meta_path)

        task_map = {'sex': 'Gender', 'age': 'age'}
        task_col = task_map.get(self.config.downstream_task)
        if not task_col:
            raise NotImplementedError(f"Downstream task '{self.config.downstream_task}' not supported for S1200.")
        meta_df = meta_df[list(set(['subject', task_col, 'sex']))].dropna()
        meta_df['subject'] = meta_df['subject'].map(lambda x: str(int(x)))
        meta_df = meta_df[meta_df['subject'].isin(available_subjects)]

        final_dict = {}
        meta_df['sex'] = meta_df['sex'].apply(lambda x: 1 if x == 'M' else 0)

        for _, row in meta_df.iterrows():
            subject_id = row['subject']
            sex = row['sex']
            target = row[task_col]
            if self.config.downstream_task == 'sex':
                target = sex
            final_dict[subject_id] = (sex, target)
        return final_dict
        
    
    def make_subject_dict(self, dataset_name: str, image_path: str) -> Dict:
        if self.config.use_subj_dict:
            image_name_part = [p for p in image_path.split("/") if 'MNI_to_TRs' in p]
            image_name = image_name_part[0] if image_name_part else Path(image_path).name
            cache_dir = Path("./data/subj_dict/")
            cache_dir.mkdir(parents=True, exist_ok=True)
            subj_dict_path = cache_dir / f"{dataset_name}_{self.config.downstream_task}_{image_name}.pickle"
            if os.path.exists(subj_dict_path):
                print(f"Loading cached subject dictionary from {subj_dict_path}")
                with open(subj_dict_path, 'rb') as f: return pickle.load(f)
        img_root = os.path.join(image_path, 'img')
        available_subjects = {s for s in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, s))}
        if dataset_name == "S1200" or dataset_name == "S1200test" : final_dict = self._make_s1200_dict(image_path, available_subjects)
        else: raise NotImplementedError(f"Subject dictionary creation not implemented for {dataset_name}")
        if self.config.use_subj_dict:
            if not self.distributed or dist.get_rank() == 0:
                with open(subj_dict_path, 'wb') as f: pickle.dump(final_dict, f)
                print(f"[RANK 0] Saved subject dictionary to {subj_dict_path}")
            if self.distributed: dist.barrier()
        return final_dict

    def define_split_file_path(self, dataset_name, split_num):
        os.makedirs("./data/splits", exist_ok=True)
        return f"./data/splits/{dataset_name}_split{split_num}.pkl"

    def load_or_create_split(self, subject_dict, path):
        if os.path.exists(path):
            print(f"Loading splits from {path}")
            with open(path, 'rb') as f: return pickle.load(f)
        print(f"Randomly determining splits and saving to {path}")
        subjects = list(subject_dict.keys())
        random.shuffle(subjects)
        n = len(subjects)
        n_train = int(n * self.config.train_split)
        n_val = int(n * self.config.val_split)
        train_names = subjects[:n_train]
        val_names = subjects[n_train : n_train + n_val]
        test_names = subjects[n_train + n_val :]
        if not self.distributed or dist.get_rank() == 0:
            with open(path, 'wb') as f: pickle.dump((train_names, val_names, test_names), f)
        if self.distributed: dist.barrier()
        return train_names, val_names, test_names