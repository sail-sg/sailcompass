from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AnsEvaluator
from opencompass.datasets import IndolemDataset


indolem_reader_cfg = dict(
    input_columns=['text'],
    output_column='label',)

system_prompt ='''Read the provided Indonesian text and determine its sentiment label. Your label should be either positive or negative.\n'''

exp_1 = '''Text: Kawan, Ibu and ayah kau pergi bukan bermakna dia tinggalkan kau Tetapi, Allah sayangkan mereka. Sayang mereka? Jadi org baik Doakn mereka'''
ans_1 = '''Label: positive\n'''

exp_2 = '''Text: Tidak ada pelayanan saat datang dan bantuan bawa barang pada hal kami kesulitan. Saat masuk kamar bau apek sekali hingga dua jam hingga kami bantu pewangian.'''
ans_2 = '''Label: negative\n'''

exp_3 = '''Text: Cinta sejati selalu datang dan takkan pernah pergi untuk selamanya, karena cinta sejati itu, selalu hadir untuk memberi bukan untuk meminta.'''
ans_3 = '''Label: positive\n'''

prompt_input = '''Text: {text}'''
prompt_output = '''Label:'''


trans = {
    'positive' : 'positive',
    'negative': 'negative',
}


indolem_infer_cfg = dict(
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
                    dict(role='BOT', prompt=f'Label: {trans[ans]}'),
                ]) 
            for ans in ['positive', 'negative']
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer, generation_kwargs=dict(do_sample=False)))

indolem_eval_cfg = dict(
    evaluator=dict(type=AnsEvaluator),
    pred_role="BOT")


indolem_datasets = [
    dict(
        type=IndolemDataset,
        abbr='indolem-en',
        path='./data/indolem_senti/test.jsonl',
        reader_cfg=indolem_reader_cfg,
        infer_cfg=indolem_infer_cfg,
        eval_cfg=indolem_eval_cfg)
]


