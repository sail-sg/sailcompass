from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import TextGenEvaluator
from opencompass.datasets import Flores200Dataset


flores_reader_cfg = dict(
    input_columns=['target'],
    output_column='source')

system_prompt = '''Dịch câu được cung cấp từ tiếng Việt sang tiếng Anh mà vẫn giữ nguyên nghĩa và ngữ cảnh ban đầu.\n'''

exp_1 = '''Tiếng Việt: Chỉ có chưa tới một ngàn ca bệnh ở người được báo cáo, nhưng một số ca đã dẫn đến tử vong.'''
ans_1 = '''Tiếng Anh: Fewer than a thousand cases have ever been reported in humans, but some of them have been fatal.\n'''

exp_2 = '''Tiếng Việt: Những dịch vụ này thường được sử dụng qua một số điện thoại miễn cước có thể gọi từ hầu hết các điện thoại mà không bị tính phí.'''
ans_2 = '''Tiếng Anh: Access to these services is often through a toll-free telephone number that can be called from most phones without charge.\n'''

exp_3 = '''Tiếng Việt: Bạn không nên cho máy bay không người lái bay gần sân bay hoặc trên đầu một đám đông, ngay cả khi đó là hành vi không phạm pháp tại địa phương của bạn.'''
ans_3 = '''Tiếng Anh: Flying a drone near an airport or over a crowd is almost always a bad idea, even if it's not illegal in your area.\n'''

prompt_input = '''Tiếng Việt: {target}'''
prompt_output = '''Tiếng Anh:'''



flores_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt=system_prompt),
            ],
            round=[
                dict(role='HUMAN', prompt=exp_1),
                dict(role='BOT', prompt=ans_1),
                dict(role='HUMAN', prompt=exp_2),
                dict(role='BOT', prompt=ans_2),
                dict(role='HUMAN', prompt=exp_3),
                dict(role='BOT', prompt=ans_3),
                dict(role='HUMAN', prompt=prompt_input),
                dict(role='BOT', prompt=prompt_output),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, generation_kwargs=dict(do_sample=False), max_out_len=100))


flores_eval_cfg = dict(
    evaluator=dict(type=TextGenEvaluator),
    pred_role='BOT')

flores_datasets = [
    dict(
        type=Flores200Dataset,
        abbr='flores-vi-en',
        path='./data/flores200/vie_Latn_devtest.jsonl',
        reader_cfg=flores_reader_cfg,
        infer_cfg=flores_infer_cfg,
        eval_cfg=flores_eval_cfg)
]




