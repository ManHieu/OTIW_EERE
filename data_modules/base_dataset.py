from abc import ABC, abstractmethod
import logging
import math
import os
import pickle
from typing import List
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from data_modules.input_example import InputExample


class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.
    """
    name = None         # name of the dataset

    def __init__(
        self,
        tokenizer: str,
        data_dir: str,
        max_input_length: int,
        seed: int = None,
        split = 'train',
        ) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        self.max_input_length = max_input_length
        self.data_path = data_dir
        self.split = split

        self.examples: List[InputExample] = self.load_data(split=split)
        self.features: List[InputExample] = self.compute_futures()
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, index):
        return self.features[index]
    
    @abstractmethod
    def load_schema(self):
        """
        Load extra dataset information, such as entity/relation types.
        """
        pass

    @abstractmethod
    def load_data(self, split: str) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        pass

    def compute_futures(self) -> List[InputExample]:
        return self.examples
    
    def my_collate(self, batch: List[InputExample]):
        return batch