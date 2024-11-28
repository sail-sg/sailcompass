from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import TextGenEvaluator
from opencompass.datasets import Flores200Dataset


flores_reader_cfg = dict(
    input_columns=['target'],
    output_column='source')

system_prompt = '''Terjemahkan kalimat yang disediakan dari Bahasa Indonesia ke Bahasa Inggris dengan tetap menjaga makna dan konteks aslinya.\n'''

exp_1 = '''Bahasa Indonesia: Kurang dari seribu kasus pernah dilaporkan terjadi pada manusia, tetapi beberapa di antaranya merupakan kasus yang fatal.'''
ans_1 = '''Bahasa Inggris: Fewer than a thousand cases have ever been reported in humans, but some of them have been fatal.\n'''


exp_2 = '''Bahasa Indonesia: Akses ke layanan-layanan ini biasanya melalui nomor telepon bebas pulsa yang dapat dihubungi dari sebagian besar telepon tanpa biaya.'''
ans_2 = '''Bahasa Inggris: Access to these services is often through a toll-free telephone number that can be called from most phones without charge.\n'''

exp_3 = '''Bahasa Indonesia: Menerbangkan drone di dekat bandara atau di atas kerumunan hampir selalu ide yang buruk, walau hal ini legal di area Anda.'''
ans_3 = '''Bahasa Inggris: Flying a drone near an airport or over a crowd is almost always a bad idea, even if it's not illegal in your area.\n'''

prompt_input = '''Bahasa Indonesia: {target}'''
prompt_output = '''Bahasa Inggris:'''

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
        abbr='flores-id-en',
        path='./data/flores200/ind_Latn_devtest.jsonl',
        reader_cfg=flores_reader_cfg,
        infer_cfg=flores_infer_cfg,
        eval_cfg=flores_eval_cfg)
]




