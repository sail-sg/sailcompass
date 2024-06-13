from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer 
from opencompass.datasets import BelebeleDataset, BelebeleEvaluator


belebele_reader_cfg = dict(
    input_columns=['flores_passage', 'question', 'options'],
    output_column='answer')

system_prompt = '''Berikut ini merupakan soal pilihan ganda, harap dibaca konteksnya dan berikan hanya satu pilihan yang benar, tanpa rincian atau penjelasan lainnya.\n'''

exp_1 = '''Konteks: Simbol nuklir digunakan untuk menulis persamaan nuklir peluruhan radioaktif. Mari kita perhatikan contoh peluruhan beta-minus dari thorium-234 menjadi protaktinium-234. Reaksi ini diwakili oleh persamaan :.
Pertanyaan: Apa yang digunakan untuk menulis persamaan nuklir peluruhan radioaktif?
A. simbol trigonometri
B. simbol radioaktif
C. simbol kritis
D. simbol nuklir'''
ans_1 = '''Jawaban: D\n'''


exp_2 = '''Konteks: Barang obral apa pun yang dibeli dapat dikembalikan untuk mendapatkan kredit toko tetapi tidak untuk pengembalian uang sebesar harga pembelian. Setiap peralatan rumah tangga dan peralatan berkebun dijual bersama dengan peralatan konstruksi pilihan.
Pertanyaan: Jika pernyataan di atas benar, manakah pernyataan berikut yang juga benar?
A. Barang apa pun yang tidak dijual tidak dapat dikembalikan untuk mendapatkan kredit toko.
B. Beberapa alat konstruksi tidak dapat dikembalikan untuk kredit toko.
C. Peralatan berkebun tidak dapat dikembalikan untuk mendapatkan pengembalian uang.
D. Barang-barang yang dapat dikembalikan untuk pengembalian dana tidak ada yang merupakan peralatan konstruksi.'''
ans_2 = '''Jawaban: C\n'''


exp_3 = '''Konteks: Saya punya tujuh tas. Tiga tas berukuran besar, dan empat tas lainnya berukuran kecil. Saya punya beberapa bola basket dan bola voli. Saya memasukkan dua bola voli ke dalam setiap tas kecil. Dan saya menaruh dua bola basket dan dua bola voli di setiap tas besar. Jumlah bola voli adalah usia saya.
Pertanyaan: Apa isi tas besar itu?
A. Enam bola basket dan enam bola voli.
B. Delapan bola basket dan enam bola voli.
C. Dua bola basket dan enam bola voli.
D. Delapan bola basket dan delapan bola voli.'''
ans_3 = '''Jawaban: A\n'''


prompt_input = '''Konteks: {flores_passage}
Pertanyaan: {question}
{options}'''
prompt_output = '''Jawaban:'''


belebele_infer_cfg = dict(
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
    inferencer=dict(type=GenInferencer, generation_kwargs=dict(do_sample=False), max_out_len=10))


belebele_eval_cfg = dict(
    evaluator=dict(type=BelebeleEvaluator),
    pred_role="BOT")

belebele_datasets = [
    dict(
        type=BelebeleDataset,
        abbr='belebele-id',
        path='./data/belebele/ind_Latn.jsonl',
        reader_cfg=belebele_reader_cfg,
        infer_cfg=belebele_infer_cfg,
        eval_cfg=belebele_eval_cfg)
]
