from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AnsEvaluator
from opencompass.datasets import WiseSentiDataset


wisesenti_reader_cfg = dict(
    input_columns=['texts'],
    output_column='category',)

system_prompt = '''โปรดอ่านข้อความที่ให้มาเป็นภาษาไทยและระบุป้ายกำกับอารมณ์ของมัน การตอบของคุณควรเป็นหนึ่งในตัวเลือกต่อไปนี้: เป็นกลาง, เชิงบวก, เชิงลบ\n'''

exp_1 = '''ข้อความ: จ้างอีซูซุผลิตให้ครับ บอกลาจากฟอร์ดแล้ว พูดง่ายๆ ดีแมกซ์ในคราบมาสด้า อีก 2 ปีได้เห็นครับ'''
ans_1 = '''คำตอบ: เป็นกลาง\n'''

exp_2 = '''ข้อความ: นิสสันนาวาราขับลุยงาน นี้เจ๋งมาก ซื้อมาไม่เคยมีปัญหาต้องซ่อมเหมือนคนอื่น ประทับใจจริงๆ'''
ans_2 = '''คำตอบ: เชิงบวก\n'''

exp_3 = '''ข้อความ: ตอนนี้พอกินเหล้าละอยากดูด vape ทุกทีเลยว่ะ เพราะเทออะ แม่ง แง'''
ans_3 = '''คำตอบ: เชิงลบ\n'''

prompt_input = '''ข้อความ: {texts}'''
prompt_output = '''คำตอบ:'''

trans = {
    0: 'เชิงบวก',
    1: 'เป็นกลาง',
    2: 'เชิงลบ'
}

wisesenti_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            str(ans): dict(
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
            for ans in range(3)
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer, generation_kwargs=dict(do_sample=False), batch_size=4))

wisesenti_eval_cfg = dict(
    evaluator=dict(type=AnsEvaluator),
    pred_role="BOT")

wisesenti_datasets = [
    dict(
        type=WiseSentiDataset,
        abbr='wisesenti-th',
        path='./data/wisesight_senti/test.jsonl',
        reader_cfg=wisesenti_reader_cfg,
        infer_cfg=wisesenti_infer_cfg,
        eval_cfg=wisesenti_eval_cfg)
]

#  