import os
import json
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset


@LOAD_DATASET.register_module()
class IndonliDataset(BaseDataset):
    @staticmethod
    def load(path):
        return Dataset.from_list([json.loads(line.strip()) for line in open(path)])

