from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import TextGenEvaluator
from opencompass.datasets import IndosumDataset

indosum_reader_cfg = dict(
    input_columns=['input'],
    output_column='target',)

system_prompt = '''Silakan ikuti contoh yang diberikan dan tulis ringkasan singkat dalam bahasa Indonesia untuk teks yang diberikan.\n'''

exp_1 = '''Teks: Merdeka.com - Manajer Manchester United, Jose Mourinho, dikabarkan menuntut nilai transfer 100 juta poundsterling untuk David de Gea, jika memang Real Madrid masih tertarik membelinya. Pemain Spanyol sering dikaitkan dengan Los Blancos di beberapa kesempatan belakangan ini, namun Setan Merah tentu tak ingin melepasnya dengan harga murah. United mungkin siap menurunkan permintaan mereka, jika Real memasukkan nama Alvaro Morata sebagai bagian dari kesepakatan. Striker 24 tahun dikabarkan merupakan salah satu prioritas United di bursa musim panas dan United mungkin siap melepas De Gea dengan harga 45 juta pounds jika sang pemain dimasukkan dalam kesepakatan, menurut Daily Star. Madrid sendiri menginginkan tak kurang dari 60 juta pounds untuk Morata, jadi sepertinya bakal ada negosiasi yang alot mengingat United menuntut harga tinggi untuk De Gea. Pemain AC Milan yang belum lama ini menolak kontrak baru, Gianluigi Donnarumma, juga sempat dikaitkan dengan Madrid, yang mencari kiper baru untuk menggantikan Keylor Navas. (dst / rer)'''
ans_1 = '''Ringkasan: Manajer Manchester United, Jose Mourinho, dikabarkan menuntut nilai transfer 100 juta poundsterling untuk David de Gea, jika memang Real Madrid masih tertarik membelinya. Pemain Spanyol sering dikaitkan dengan Los Blancos di beberapa kesempatan belakangan ini, namun Setan Merah tentu tak ingin melepasnya dengan harga murah . United mungkin siap menurunkan permintaan mereka, jika Real memasukkan nama Alvaro Morata sebagai bagian dari kesepakatan.\n'''

prompt_input = '''Teks: {input}'''
prompt_output = '''Ringkasan:'''

indosum_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt=system_prompt),
            ],
            round=[
                dict(role='HUMAN', prompt=exp_1),
                dict(role='BOT', prompt=ans_1),
                dict(role='HUMAN', prompt=prompt_input),
                dict(role='BOT', prompt=prompt_output),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, generation_kwargs=dict(do_sample=False), max_out_len=240, batch_size=4))


indosum_eval_cfg = dict(
    evaluator=dict(type=TextGenEvaluator),
    pred_role='BOT')

indosum_datasets = [
    dict(
        type=IndosumDataset,
        abbr='indosum-id',
        path='./data/indosum/trunc_test_1000.jsonl',
        reader_cfg=indosum_reader_cfg,
        infer_cfg=indosum_infer_cfg,
        eval_cfg=indosum_eval_cfg)
]











