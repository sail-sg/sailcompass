from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AnsEvaluator
from opencompass.datasets import VsmecDataset


vsmec_reader_cfg = dict(
    input_columns=['text'],
    output_column='label',)

system_prompt ='''Đọc văn bản tiếng Việt và xác định nhãn tình cảm của nó. Câu trả lời của bạn nên là một trong các lựa chọn sau: Sợ hãi, Khác, Ghê tởm, Buồn bã, Thích thú, Bất ngờ, Tức giận.\n'''

exp_1 = '''Văn bản: per nhớ ngày tháng cta bên nhau ghê , giờ thì chả còn gì'''
ans_1 = '''Nhãn: Buồn bã\n'''

exp_2 = '''Văn bản: phim này xem nam chính thôi . nữ chính diễn chán vãi lúa lắm lúc tụt hứng lắm . giận không ra giận , vui không ra vui .'''
ans_2 = '''Nhãn: Ghê tởm\n'''

exp_3 = '''Văn bản: bài hát đầu tiên của phim là gì nhỉ quên mất rồi . bạn nào thông não cho mình phát'''
ans_3 = '''Nhãn: Khác\n'''

prompt_input = '''Văn bản: {text}'''
prompt_output = '''Nhãn:'''


vsmec_infer_cfg = dict(
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
                    dict(role='BOT', prompt=f'Nhãn: {ans}'),
                ]) 
            for ans in ['Sợ hãi', 'Ghê tởm', 'Buồn bã', 'Thích thú', 'Khác', 'Tức giận', 'Bất ngờ']
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer, generation_kwargs=dict(do_sample=False)))

vsmec_eval_cfg = dict(
    evaluator=dict(type=AnsEvaluator),
    pred_role="BOT")


vsmec_datasets = [
    dict(
        type=VsmecDataset,
        abbr='vsmec-vi',
        path='./data/vsmec/test.json',
        reader_cfg=vsmec_reader_cfg,
        infer_cfg=vsmec_infer_cfg,
        eval_cfg=vsmec_eval_cfg)
]


