from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer 
from opencompass.datasets import BelebeleDataset, BelebeleEvaluator


belebele_reader_cfg = dict(
    input_columns=['flores_passage', 'question', 'options'],
    output_column='answer')

system_prompt = '''ต่อไปนี้เป็นคำถามแบบปรนัย โปรดอ่านบริบทและให้ตัวเลือกที่ถูกต้องเพียงตัวเลือกเดียว โดยไม่มีรายละเอียดหรือคำอธิบายอื่นใด\n'''

exp_1 = '''บริบท: สัญลักษณ์นิวเคลียร์ใช้ในการเขียนสมการนิวเคลียร์สำหรับการสลายกัมมันตภาพรังสี ลองพิจารณาตัวอย่างของการสลายตัวแบบเบตาลบของทอเรียม-234 ไปเป็นโปรแทกติเนียม-234 ปฏิกิริยานี้แสดงด้วยสมการ:
คำถาม: อะไรคือสิ่งที่ใช้ในการเขียนสมการนิวเคลียร์สำหรับการสลายกัมมันตภาพรังสี?
A. สัญลักษณ์ตรีโกณมิติ
B. สัญลักษณ์กัมมันตภาพรังสี
C. สัญลักษณ์วิกฤต
D. สัญลักษณ์นิวเคลียร์'''
ans_1 = '''คำตอบ: D\n'''


exp_2 = '''บริบท: สินค้าลดราคาใดๆ ที่ซื้อสามารถคืนเป็นเครดิตร้านค้าได้ แต่ไม่สามารถขอคืนเงินตามราคาที่ซื้อได้ จำหน่ายเครื่องใช้ไฟฟ้าภายในบ้านและอุปกรณ์ทำสวนทุกชิ้นพร้อมอุปกรณ์ก่อสร้างที่คัดสรร
คำถาม: หากข้อความข้างต้นเป็นจริง ข้อใดต่อไปนี้จะต้องเป็นจริงด้วย
A. สินค้าที่ไม่ได้ลดราคาไม่สามารถคืนเป็นเครดิตร้านค้าได้
B. เครื่องมือก่อสร้างบางอย่างไม่สามารถคืนเป็นเครดิตร้านค้าได้
C. ไม่มีการคืนอุปกรณ์ทำสวนเพื่อขอเงินคืน
D. ไม่มีสิ่งใดที่สามารถคืนเพื่อขอเงินคืนได้ซึ่งถือเป็นเครื่องมือในการก่อสร้าง'''
ans_2 = '''คำตอบ: C\n'''


exp_3 = '''บริบท: ฉันมีเจ็ดถุง สามถุงใหญ่และอีกสี่ถุงเล็ก ฉันมีบาสเก็ตบอลและวอลเล่ย์บอล ฉันใส่ลูกวอลเลย์บอลสองลูกไว้ในถุงเล็กแต่ละใบ และฉันใส่ลูกบาสเก็ตบอลสองลูกและลูกวอลเล่ย์บอลสองลูกไว้ในกระเป๋าใบใหญ่แต่ละใบ จำนวนวอลเล่ย์บอลคืออายุของฉัน
คำถาม: อะไรอยู่ในถุงใหญ่เหล่านั้น?
A. บาสเก็ตบอลหกลูกและวอลเลย์บอลหกลูก
B. บาสเก็ตบอลแปดลูกและวอลเลย์บอลหกลูก
C. บาสเก็ตบอลสองลูกและวอลเลย์บอลหกลูก
D. บาสเก็ตบอลแปดลูกและวอลเลย์บอลแปดลูก'''
ans_3 = '''คำตอบ: A\n'''


prompt_input = '''บริบท: {flores_passage}
คำถาม: {question}
{options}'''
prompt_output = '''คำตอบ:'''


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
        abbr='belebele-th',
        path='./data/belebele/tha_Thai.jsonl',
        reader_cfg=belebele_reader_cfg,
        infer_cfg=belebele_infer_cfg,
        eval_cfg=belebele_eval_cfg)
]
