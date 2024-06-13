from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer 
from opencompass.datasets import BelebeleDataset, BelebeleEvaluator


belebele_reader_cfg = dict(
    input_columns=['flores_passage', 'question', 'options'],
    output_column='answer')

system_prompt = '''Dưới đây là những câu hỏi trắc nghiệm, vui lòng đọc ngữ cảnh và chỉ đưa ra một phương án đúng, không kèm theo bất kỳ chi tiết hay giải thích nào khác.\n'''

exp_1 = '''Bối cảnh: Ký hiệu hạt nhân được dùng để viết phương trình hạt nhân của sự phân rã phóng xạ. Hãy xem xét ví dụ về sự phân rã beta-trừ của thorium-234 thành protactinium-234. Phản ứng này được biểu diễn bằng phương trình:.
Câu hỏi: Người ta dùng gì để viết phương trình hạt nhân của sự phân rã phóng xạ?
A. ký hiệu lượng giác
B. ký hiệu phóng xạ
C. biểu tượng quan trọng
D. ký hiệu hạt nhân'''
ans_1 = '''Trả lời: D\n'''


exp_2 = '''Bối cảnh: Bất kỳ mặt hàng giảm giá nào đã mua đều có thể được trả lại để lấy tín dụng của cửa hàng nhưng không được hoàn lại giá mua. Mọi thiết bị gia dụng và mọi thiết bị làm vườn đều được giảm giá cùng với các dụng cụ xây dựng được chọn lọc.
Câu hỏi: Nếu những câu trên là đúng thì câu nào sau đây cũng phải đúng?
A. Bất kỳ mặt hàng nào không được bán đều không thể được trả lại để lấy tín dụng của cửa hàng.
B. Một số công cụ xây dựng không được trả lại dưới dạng tín dụng của cửa hàng.
C. Không có thiết bị làm vườn nào được trả lại để được hoàn lại tiền.
D. Không có thứ nào có thể trả lại để hoàn lại tiền là công cụ xây dựng.'''
ans_2 = '''Trả lời: C\n'''


exp_3 = '''Bối cảnh: Tôi có bảy túi. Ba túi lớn và bốn túi còn lại nhỏ. Tôi có một số quả bóng rổ và bóng chuyền. Tôi để hai quả bóng chuyền vào mỗi túi nhỏ. Và tôi để hai quả bóng rổ và hai quả bóng chuyền vào mỗi chiếc túi lớn. Số quả bóng chuyền bằng tuổi tôi.
Câu hỏi: Trong những chiếc túi lớn đó có gì?
A. Sáu quả bóng rổ và sáu quả bóng chuyền.
B. Tám quả bóng rổ và sáu quả bóng chuyền.
C. Hai quả bóng rổ và sáu quả bóng chuyền.
D. Tám quả bóng rổ và tám quả bóng chuyền.'''
ans_3 = '''Trả lời: A\n'''



prompt_input = '''Bối cảnh: {flores_passage}
Câu hỏi: {question}
{options}'''
prompt_output = '''Trả lời:'''


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
        abbr='belebele-vi',
        path='./data/belebele/vie_Latn.jsonl',
        reader_cfg=belebele_reader_cfg,
        infer_cfg=belebele_infer_cfg,
        eval_cfg=belebele_eval_cfg)
]
