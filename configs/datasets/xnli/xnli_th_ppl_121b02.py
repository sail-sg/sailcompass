from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AnsEvaluator
from opencompass.datasets import XnliDataset


xnli_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',)


trans = {
    'neutral' : 'เป็นกลาง',
    'entailment': 'นำไปสู่',
    'contradiction': 'ขัดแย้ง'
}



system_prompt = '''โปรดอ่านทฤษฎีและสมมติฐาน และระบุความสัมพันธ์ตรรกะของพวกเขา การตอบของคุณควรเป็นหนึ่งในตัวเลือกต่อไปนี้: เป็นกลาง, นำไปสู่, ขัดแย้ง\n'''  

exp_1 = '''สถานที่: โจนส์หมายถึง เซอร์ วิลเลียม จอห์นสัน, กล่าวว่า เขาเป็นที่รัก, เป็นที่เชยชม, และเกือบจะเป็นที่รักของชาวอินเดียนแดง
สมมติฐาน: ชาวอินเดียแสดงให้ Sir William Johnson เห็นถึงวิธีการปลูกต้นไม้ในสวน'''
ans_1 = f'''คำตอบ: {trans['neutral']}\n'''


exp_2 = '''สถานที่: ฉันยังมีที่ว่างสำหรับดื่มสก็อตช์ได้อีกหกแก้ว
สมมติฐาน: ฉันยังสามารถดื่มสก็อตได้มากกว่าอีก 6 ครั้ง'''
ans_2 = f'''คำตอบ: {trans['entailment']}\n'''


exp_3 = '''สถานที่: เดือนผ่านไป ตั้งแต่การเลือกตั้ง พรรครีพับลิกัน และ พรรคเดโมแครต ยังคงไฮว์ไฟว์กันอยู่
สมมติฐาน: เป็นเวลาเพียงหนึ่งสัปดาห์นับตั้งแต่การเลือกตั้ง'''
ans_3 = f'''คำตอบ: {trans['contradiction']}\n'''


prompt_input = '''สถานที่: {sentence1}
สมมติฐาน: {sentence2}'''
prompt_output = '''คำตอบ:'''


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
                    dict(role='BOT', prompt=f'คำตอบ: {trans[ans]}'),
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
        abbr='xnli-th-ppl',
        path='./data/xnli/th_xnli_test.jsonl',
        reader_cfg=xnli_reader_cfg,
        infer_cfg=xnli_infer_cfg,
        eval_cfg=xnli_eval_cfg)
]











