from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import TextGenEvaluator
from opencompass.datasets import XlsumViDataset

xlsum_reader_cfg = dict(
    input_columns=['text'],
    output_column='summary',)


system_prompt = '''Hãy làm theo ví dụ được cung cấp và viết tóm tắt ngắn gọn bằng tiếng Việt cho văn bản đã cho.\n'''

exp_1 = '''Văn bản: Báo The Nation của Thái Lan dẫn lời tổng thư ký hiệp hội, Chidchai Sakormbadee, rằng khoảng 10% của ước đoán 12 triệu khách đến Thái Lan sẽ thay đổi kế hoạch. Nếu điều này xảy ra, nó sẽ gây thiệt hại khoảng 30 tỉ baht. Ông này nói riêng Phuket đã thường tiếp đón 1.5 triệu du khách trong bốn tháng từ 11 đến tháng Hai và nay khu vực này có nguy cơ thiệt hại nhiều nhất. Các tường thuật nói rằng 10 khách sạn và khu nghỉ mát ở Phuket đã bị thiệt hại nặng. Đối với thị trường nội địa, Hiệp hội quảng bá du lịch nội địa Thái cho biết người Thái Lan đã hủy kế hoạch đến Phuket trong tương lai trước mắt. Hiệp hội này cũng ước tính thu nhập du lịch nội địa năm nay sẽ giảm 10%, tương đương 30 tỉ baht.'''
ans_1 = '''Tóm tắt: Hiệp hội các đại lý du lịch Thái Lan nói khoảng 1.2 triệu du khách nước ngoài có thể hủy vé đặt trước đến Thái Lan do hậu quả thiên tai hôm Chủ nhật.\n'''

prompt_input = '''Văn bản: {text}'''
prompt_output = '''Tóm tắt:'''


xlsum_infer_cfg = dict(
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
    inferencer=dict(type=GenInferencer, generation_kwargs=dict(do_sample=False), max_out_len=180, batch_size=4))



xlsum_eval_cfg = dict(
    evaluator=dict(type=TextGenEvaluator),
    pred_role='BOT')


xlsum_datasets = [
    dict(
        type=XlsumViDataset,
        abbr='xlsum-vi',
        path='./data/xlsum_vi/test_1000.jsonl',
        reader_cfg=xlsum_reader_cfg,
        infer_cfg=xlsum_infer_cfg,
        eval_cfg=xlsum_eval_cfg)
]