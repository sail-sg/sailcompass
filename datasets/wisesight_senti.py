import os
import json
from .base import BaseDataset
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class WiseSentiDataset(BaseDataset):
    @staticmethod
    def load(path):
        return Dataset.from_list([json.loads(line.strip()) for line in open(path)])