from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AnsEvaluator
from opencompass.datasets import IndonliDataset

indonli_reader_cfg = dict(
    input_columns=['premise', 'hypothesis'],
    output_column='label',)


system_prompt = '''Silakan baca premis dan hipotesis, dan identifikasi hubungan logis di antara keduanya. Tanggapan Anda harus salah satu dari opsi berikut: netral, implikasi, kontradiksi.\n''' 

exp_1 = '''Premis: Selera Tiongkok yang tak pernah terpuaskan terhadap ayam goreng, dulunya merupakan alasan besar mengapa para investor mencintai induk KFC, Yum Brands.
Hipotesis: Banyak sekali investor yang lihai untuk menanamkan uang di KFC.'''
ans_1 = '''Jawaban: netral\n'''

exp_2 = '''Premis: Madonna awalnya mencoba menjadi vokalis band rock, namun gagal. Dia kemudian menyanyikan genre pop dipadukan dengan dance.
Hipotesis: Madonna menyanyikan genre pop karena gagal sebagai vokalis band rock.'''
ans_2 = '''Jawaban: implikasi\n'''

exp_3 = '''Premis: Sesampainya di dalam kamar, Amanda harus berhadapan dengan Septian yang sedang mencurigainya berselingkuh dengan lelaki lain.
Hipotesis: Septian mencurigai Amanda berselingkuh dengan perempuan lain.'''
ans_3 = '''Jawaban: kontradiksi\n'''

prompt_input = '''Premis: {premise}
Hipotesis: {hypothesis}'''
prompt_output = '''Jawaban:'''

trans = {
    'neutral' : 'netral',
    'entailment': 'implikasi',
    'contradiction': 'kontradiksi'
}


indonli_infer_cfg = dict(
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
                    dict(role='BOT', prompt=f'Jawaban: {trans[ans]}'),
                ]) 
            for ans in ['neutral', 'entailment', 'contradiction']
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer, generation_kwargs=dict(do_sample=False)))


indonli_eval_cfg = dict(
    evaluator=dict(type=AnsEvaluator),
    pred_role='BOT')

indonli_datasets = [
    dict(
        type=IndonliDataset,
        abbr='indonli-id-ppl',
        path='./data/indonli/test.jsonl',
        reader_cfg=indonli_reader_cfg,
        infer_cfg=indonli_infer_cfg,
        eval_cfg=indonli_eval_cfg)
]











