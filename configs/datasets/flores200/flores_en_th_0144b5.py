from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import TextGenEvaluator
from opencompass.datasets import Flores200Dataset


flores_reader_cfg = dict(
    input_columns=['source'],
    output_column='target')

system_prompt = '''แปลประโยคที่ให้จากภาษาอังกฤษเป็นภาษาไทยโดยยังคงความหมายและบริบทดั้งเดิมไว้\n''' 

exp_1 = '''อังกฤษ: Fewer than a thousand cases have ever been reported in humans, but some of them have been fatal.'''
ans_1 = '''ไทย: มีรายงานผู้ป่วยน้อยกว่าหนึ่งพันรายในมนุษย์ แต่บางรายมีความรุนแรงถึงขั้นทำให้เสียชีวิตได้\n'''

exp_2 = '''อังกฤษ: Access to these services is often through a toll-free telephone number that can be called from most phones without charge.'''
ans_2 = '''ไทย: การเข้าถึงบริการต่าง ๆ เหล่านี้มักใช้หมายเลขโทรศัพท์โทรฟรีที่ใช้โทรศัพท์ส่วนใหญ่โทรได้โดยไม่เสียค่าใช้จ่าย\n'''

exp_3 = '''อังกฤษ: Flying a drone near an airport or over a crowd is almost always a bad idea, even if it's not illegal in your area.'''
ans_3 = '''ไทย: การบังคับโดรนให้บินใกล้ท่าอากาศยานหรือในฝูงชนแทบจะเป็นความคิดที่แย่อยู่เสมอ แม้ว่าการทำเช่นนี้จะไม่ผิดกฎหมายในพื้นที่ของคุณก็ตาม\n'''

prompt_input = '''อังกฤษ: {source}'''
prompt_output = '''ไทย:'''


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
    inferencer=dict(type=GenInferencer, generation_kwargs=dict(do_sample=False), max_out_len=350))


flores_eval_cfg = dict(
    evaluator=dict(type=TextGenEvaluator),
    pred_role='BOT')

flores_datasets = [
    dict(
        type=Flores200Dataset,
        abbr='flores-en-th',
        path='./data/flores200/tha_Thai_devtest.jsonl',
        reader_cfg=flores_reader_cfg,
        infer_cfg=flores_infer_cfg,
        eval_cfg=flores_eval_cfg)
]




