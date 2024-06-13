import os
import re
import json
from .base import BaseDataset
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class XlsumViDataset(BaseDataset):
    @staticmethod
    def load(path):
        dataset = Dataset.from_list([json.loads(line.strip()) for line in open(path)])
        return dataset
        