from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import TextGenEvaluator
from opencompass.datasets import ThaisumDataset


thaisum_reader_cfg = dict(
    input_columns=['body'],
    output_column='summary',)

system_prompt = '''โปรดปฏิบัติตามตัวอย่างที่ให้ไว้และเขียนบทสรุปสั้นๆ เป็นภาษาไทยสำหรับข้อความที่กำหนด\n'''

exp_1 = '''ข้อความ: นายคมกฤษณ์ ชนะศรี ประธานสภาพนักงานมหาวิทยาลัยสงขลาครินทร์ วิทยาเขตหาดใหญ่ ในฐานะประธานสภาข้าราชการ พนักงานและลูกจ้างมหาวิทยาลัยแห่งประเทศไทย (ปขมท.) เปิดเผยว่า ปขมท.ได้วิเคราะห์ร่าง พ.ร.บ.การอุดมศึกษาพ.ศและการจัดตั้งกระทรวงการอุดมศึกษา เพื่อเสนอคณะทำงานเตรียมจัดตั้งกระทรวงการอุดมศึกษา โดยพบว่ามหาวิทยาลัยมีหลายประเภท แต่ละมหาวิทยาลัยและแต่ละกลุ่มมีศักยภาพต่างกัน สิ่งที่ ปขมท. เรียกร้องและเสนอต่อ คณะทำงานเตรียมการจัดตั้งฯ รวมทั้งสำนักงานคณะกรรมการการอุดมศึกษา (สกอ.) ไปแล้ว คือ 1.อยากให้กำหนดมาตรฐานขั้นต่ำในการดูแลบุคลากรทุกประเภท ทั้งเงินเดือน ค่ารักษาพยาบาล สวัสดิการต่างๆ โดยเป็นมาตรฐานขั้นต่ำที่ทุกมหาวิทยาลัยควรมี ส่วนมหาวิทยาลัยที่มีศักยภาพสูงก็ให้เงินเดือน สวัสดิการที่สูงขึ้น ซึ่งถือเป็นเรื่องดี,ประธาน ปขมท. กล่าวต่อว่า 2. การกำหนดความก้าวหน้าทางวิชาชีพของพนักงานสายสนับสนุนการสอน ซึ่งปัจจุบันขึ้นกับแต่ละมหาวิทยาลัยกำหนด ปขมท.เห็นว่า สกอ.น่าจะตั้งคณะทำงานกลางขึ้นชุดหนึ่ง กำหนดมาตรฐานขึ้นต่ำ ทั้งมีหน้าที่ติดตามการดำเนินการของแต่ละมหาวิทยาลัยว่าบุคลากรแต่ละกลุ่มมีความก้าวหน้าหรือติดขัด ปัญหาใด และอยากให้ช่วยเหลืออย่างไร โดยส่วนตัว เห็นว่าการแยกการอุดมศึกษาออกไปเป็นกระทรวงการอุดมศึกษาจะทำให้การดูแลบุคลากรกลุ่มต่างๆ ได้ดีขึ้น แต่ด้านสวัสดิการต่างๆ ยังไม่ค่อยชัดเจน.'''
ans_1 = '''สรุป: ปขมท.ได้วิเคราะห์ร่าง พ.ร.บ.การอุดมศึกษา พ.ศและการจัดตั้งกระทรวงการอุดมศึกษา เพื่อเสนอคณะทำงานเตรียมจัดตั้งกระทรวงการอุดมศึกษา โดยพบว่ามหาวิทยาลัยมีหลายประเภท แต่ละมหาวิทยาลัยและแต่ละกลุ่มมีศักยภาพต่างกัน\n'''

prompt_input = '''ข้อความ: {body}'''
prompt_output = '''สรุป:'''



thaisum_infer_cfg = dict(
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
    inferencer=dict(type=GenInferencer, generation_kwargs=dict(do_sample=False), max_out_len=550, batch_size=4))


thaisum_eval_cfg = dict(
    evaluator=dict(type=TextGenEvaluator),
    pred_role='BOT')

thaisum_datasets = [
    dict(
        type=ThaisumDataset,
        abbr='thaisum-th',
        path='./data/thaisum/trunc_thaisum_test_1000.jsonl',
        reader_cfg=thaisum_reader_cfg,
        infer_cfg=thaisum_infer_cfg,
        eval_cfg=thaisum_eval_cfg)
]











