from mmengine.config import read_base

with read_base():
    #------------------------------------------------------
    from .models.sailor_7b import models as sailor_7b
    from .models.sealion_7b import models as sealion_7b
    from .models.seallm_7b_hybrid import models as seallm_7b_hybrid
    from .models.llama2_7b import models as llama2_7b
    from .models.llama3_8b import models as llama3_8b
    from .models.gemma_7b import models as gemma_7b
    from .models.mistral_7b_v0_1 import models as mistral_7b
    from .models.qwen1_5_7b import models as qwen1_5_7b
    from .models.falcon_7b import models as falcon_7b
    from .models.bloom_7b1 import models as bloom_7b1
    from .models.typhoon_8b_v1_5 import models as typhoon_8b_v1_5
    from .models.vinallama_7b import models as vinallama_7b

    from .models.qwen2_5_0_5b import models as qwen2_5_0_5b
    from .models.qwen2_5_1_5b import models as qwen2_5_1_5b
    from .models.qwen2_5_7b import models as qwen2_5_7b
    from .models.qwen2_5_14b import models as qwen2_5_14b
    from .models.qwen2_5_32b import models as qwen2_5_32b
    from .models.qwen2_5_72b import models as qwen2_5_72b

    #------------------------------------------------------
    # QA
    from .datasets.xquad.xquad_th_34e7ab import xquad_datasets as xquad_th
    from .datasets.tydiqa_id.tydiqa_id_346aa7 import tydiqa_datasets as tydiqa_id
    from .datasets.xquad.xquad_vi_34e7ab import xquad_datasets as xquad_vi
    #------------------------------------------------------
    # MT
    from .datasets.flores200.flores_en_id_ei3232 import flores_datasets as flores_en_id
    from .datasets.flores200.flores_en_th_0144b5 import flores_datasets as flores_en_th
    from .datasets.flores200.flores_en_vi_ei3232 import flores_datasets as flores_en_vi

    from .datasets.flores200.flores_id_en_ei3232 import flores_datasets as flores_id_en
    from .datasets.flores200.flores_th_en_0144b5 import flores_datasets as flores_th_en
    from .datasets.flores200.flores_vi_en_ei3232 import flores_datasets as flores_vi_en
    #------------------------------------------------------
    # TS
    from .datasets.thaisum.thaisum_th_9dd95d import thaisum_datasets as thaisum_th  
    from .datasets.indosum.indosum_id_832342 import indosum_datasets as indosum_id 
    from .datasets.xlsum_vi.xlsum_vi_48d2e3 import xlsum_datasets as xlsum_vi  
    #------------------------------------------------------
    # EXAM - PPL
    from .datasets.m3exam.m3exam_th_ppl2_481ea1 import m3exam_datasets as m3exam_th_ppl2
    from .datasets.m3exam.m3exam_th_ppl4_481ea1 import m3exam_datasets as m3exam_th_ppl4
    from .datasets.m3exam.m3exam_th_ppl5_481ea1 import m3exam_datasets as m3exam_th_ppl5
    from .datasets.m3exam.m3exam_jv_ppl_4fs13f import m3exam_datasets as m3exam_jv_ppl
    from .datasets.m3exam.m3exam_vi_ppl2_172ds2 import m3exam_datasets as m3exam_vi_ppl2
    from .datasets.m3exam.m3exam_vi_ppl3_172ds2 import m3exam_datasets as m3exam_vi_ppl3
    from .datasets.m3exam.m3exam_vi_ppl4_172ds2 import m3exam_datasets as m3exam_vi_ppl4
    #------------------------------------------------------
    # EXAM - GEN
    from .datasets.m3exam.m3exam_jv_4fs13f import m3exam_datasets as m3exam_jv
    from .datasets.m3exam.m3exam_th_481ea1 import m3exam_datasets as m3exam_th
    from .datasets.m3exam.m3exam_vi_172ds2 import m3exam_datasets as m3exam_vi
    #------------------------------------------------------
    # MRC
    from .datasets.belebele.belebele_th_ppl_23f2d2 import belebele_datasets as belebele_th_ppl
    from .datasets.belebele.belebele_id_ppl_23f2d2 import belebele_datasets as belebele_id_ppl
    from .datasets.belebele.belebele_vi_ppl_23f2d2 import belebele_datasets as belebele_vi_ppl
    #------------------------------------------------------
    # NLI
    from .datasets.xnli.xnli_vi_ppl_121b02 import xnli_datasets as xnli_vi_ppl
    from .datasets.xnli.xnli_th_ppl_121b02 import xnli_datasets as xnli_th_ppl
    from .datasets.indonli.indonli_id_ppl_384732 import indonli_datasets as indonli_id_ppl
    #------------------------------------------------------
    # CR
    from .datasets.xcopa.xcopa_th_ppl_49je23 import xcopa_datasets as xcopa_th_ppl
    from .datasets.xcopa.xcopa_id_ppl_49je23 import xcopa_datasets as xcopa_id_ppl
    from .datasets.xcopa.xcopa_vi_ppl_49je23 import xcopa_datasets as xcopa_vi_ppl  
    #------------------------------------------------------
    # SA
    from .datasets.wisesight_senti.wisesenti_th_ppl_3ir202 import wisesenti_datasets as wisesenti_th_ppl
    from .datasets.indolem_senti.indolem_id_bc2fc2 import indolem_datasets as indolem_id_ppl
    from .datasets.vsmec.vsmec_vi_3i7d12 import vsmec_datasets as vsmec_vi_ppl
    #------------------------------------------------------
    
# QA
# datasets = [*xquad_th, *tydiqa_id, *xquad_vi]

#------------------
# MT
# datasets = [*flores_en_th, *flores_en_id, *flores_en_vi]
# datasets += [*flores_th_en, *flores_id_en, *flores_vi_en]

#------------------
# TS
# datasets = [*thaisum_th, *indosum_id, *xlsum_vi]

#------------------
# EXAM - PPL
# datasets = [*m3exam_th_ppl2, *m3exam_th_ppl4, *m3exam_th_ppl5]
# datasets += [*m3exam_jv_ppl]
# datasets += [*m3exam_vi_ppl2, *m3exam_vi_ppl3, *m3exam_vi_ppl4]

#------------------
# EXAM - GEN
datasets = [*m3exam_jv, *m3exam_th, *m3exam_vi]
#------------------
# MRC
# datasets = [*belebele_th_ppl, *belebele_id_ppl, *belebele_vi_ppl]

#-----------------------------------------------
# NLI
# datasets = [*xnli_th_ppl, *indonli_id_ppl, *xnli_vi_ppl]
#-----------------------------------------------
# CR
# datasets = [*xcopa_th_ppl, *xcopa_id_ppl, *xcopa_vi_ppl]
#-----------------------------------------------
# SA
# datasets = [*wisesenti_th_ppl, *indolem_id_ppl, *vsmec_vi_ppl]



# models = sailor_7b + seallm_7b_hybrid + sealion_7b
# models = typhoon_8b_v1_5 + vinallama_7b + bloom_7b1
# models = llama3_8b + mistral_7b + gemma_7b
# models = qwen1_5_7b + llama2_7b + falcon_7b
models = qwen2_5_0_5b
