from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AnsEvaluator
from opencompass.datasets import XnliDataset


xnli_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',)

trans = {
    'neutral' : 'trung lập',
    'entailment': 'kết quả',
    'contradiction': 'mâu thuẫn'
}



system_prompt = '''Xin vui lòng đọc tiền đề và giả thuyết, và xác định mối quan hệ logic giữa chúng. Phản hồi của bạn nên là một trong các lựa chọn sau: trung lập, kết quả, mâu thuẫn.\n'''  

exp_1 = '''Tiền đề: Jones, ở đây là chỉ Ngài William Johnson, đã bình luận rằng, Ngài ấy được yêu thương, vuốt ve, và suýt bị tôn thờ bởi những người da đỏ.
Giả thuyết: Người Ấn Độ đã chỉ cho Sir William Johnson cách trồng một khu vườn.'''
ans_1 = f'''Trả lời: {trans['neutral']}\n'''

exp_2 = '''Tiền đề: Tôi chưa có chỗ cho sáu chai rượu scotch.
Giả thuyết: Tôi vẫn có thể tiêu thụ thêm sáu chai scotch nữa.'''
ans_2 = f'''Trả lời: {trans['entailment']}\n'''

exp_3 = '''Tiền đề: Một tháng đã trôi qua kể từ khi cuộc bầu cử và các đảng viên đảng Cộng hòa và Dân chủ vẫn đang có sự thịnh vượng cao.
Giả thuyết: Mới chỉ 1 tuần sau cuộc bầu cử.'''
ans_3 = f'''Trả lời: {trans['contradiction']}\n'''

prompt_input = '''Tiền đề: {sentence1}
Giả thuyết: {sentence2}'''
prompt_output = '''Trả lời:'''



xnli_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            ans: dict(
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
                    dict(role='BOT', prompt=f'Trả lời: {trans[ans]}'),
                ]) 
            for ans in ['neutral', 'entailment', 'contradiction']
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer, generation_kwargs=dict(do_sample=False)))


xnli_eval_cfg = dict(
    evaluator=dict(type=AnsEvaluator),
    pred_role='BOT')

xnli_datasets = [
    dict(
        type=XnliDataset,
        abbr='xnli-vi-ppl',
        path='./data/xnli/vi_xnli_test.jsonl',
        reader_cfg=xnli_reader_cfg,
        infer_cfg=xnli_infer_cfg,
        eval_cfg=xnli_eval_cfg)
]











