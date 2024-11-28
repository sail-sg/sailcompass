import re
import json
import random
from datasets import Dataset
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS
from opencompass.openicl.icl_evaluator import BaseEvaluator
from .base import BaseDataset
from opencompass.utils import first_option_parse
import langid


@LOAD_DATASET.register_module()
class BelebeleDataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset = Dataset.from_list([json.loads(line.strip()) for line in open(path)])
        lang = path.split('/')[-1].split('_')[0]
        def preprocess(example):
            example['answer'] = chr(int(example['correct_answer_num'])+64)   

            if lang == 'tha':
                example['mc_answer5'] = 'ฉันไม่รู้คำตอบ'
            elif lang == 'ind':
                example['mc_answer5'] = 'Saya tidak tahu jawabannya.' 
            elif lang == 'vie':
                example['mc_answer5'] = 'Tôi không biết câu trả lời.'
            example['options'] = '\n'.join([f'{chr(64+idx)}. ' + example[f'mc_answer{idx}'] for idx in range(1, 6)])  
            return example

        return dataset.map(preprocess)




@ICL_EVALUATORS.register_module()
class BelebeleEvaluator(BaseEvaluator):
    """Exact match evaluator."""

    def __init__(self, options: str = 'ABCD') -> None:
        super().__init__()
        self.options = options

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        cnt = 0
        details = []
        prompts = set([
            "Context: ",
            "Answer: ",
            "Câu hỏi: ",
            "Bối cảnh: "
        ])
        trans = {
            '1': 'A',
            '2': 'B',
            '3': 'C',
            '4': 'D',            
        }
        for pred, ans in zip(predictions, references):
            detail = {'pred': pred, 'answer': ans}

            pred = re.split(r'[\n]', pred, 1)[0]

            for prompt in prompts:
                if prompt in pred:
                    pred = pred.split(prompt)[-1]
                    break

            pred = first_option_parse(pred, self.options)

            if pred == ans:
                cnt += 1
                detail['correct'] = True
            elif pred in trans and trans[pred] == ans:
                cnt += 1
                detail['correct'] = True
            else:
                detail['correct'] = False
            details.append(detail)

        score = cnt / len(predictions) * 100

        return {'EM': score, 'details': details}