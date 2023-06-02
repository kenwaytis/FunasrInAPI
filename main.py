from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from modelscope.utils.logger import get_logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from loguru import logger as log
import base64

app = FastAPI()
executor = ThreadPoolExecutor()
loaded_model = None
log.add("serving_{time}.log",level="ERROR",rotation="5 MB",retention=2)

loaded_model = {"model_type": None, "model": None}
hotword_parm = {"hotword": None}

class Audio(BaseModel):
    file: str
    format: str = "wav"
    audio_fs: int = 16000
    hotword: str = None
    model_type: str = "normal"

def initialize_model(model_type, hotword):
    model = None
    if model_type != "hotword" and hotword != None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    elif model_type == "normal":
        log.debug("lodding model: normal")
        loaded_model["model_type"] = "normal"
        model = pipeline(
            task = Tasks.auto_speech_recognition,
            vad_model = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            model = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            # lm_model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch',
            punc_model = "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
        )# 默认Pytorch版
    elif model_type == "long":
        log.debug("lodding model: long")
        loaded_model["model_type"] = "long"
        model = pipeline(
            task = Tasks.auto_speech_recognition,
            model = "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        )
    elif model_type == "UniASR":
        log.debug("lodding model: UniASR")
        loaded_model["model_type"] = "UniASR"
        model = pipeline(
            task = Tasks.auto_speech_recognition,
            model = "damo/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline"
        )
    elif (model_type == "hotword" and hotword_parm["hotword"] != hotword) or hotword == None:
        log.debug("lodding model: hotword")
        loaded_model["model_type"] = "hotword"
        hotword_parm["hotword"] = hotword
        model = pipeline(
            task = Tasks.auto_speech_recognition,
            model = "damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
            param_dict = hotword_parm
        )
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    return model


def load_model(model_type, hotword):
    if loaded_model["model_type"] is None or loaded_model["model_type"] != model_type or (loaded_model["model_type"]=="hotword" and hotword_parm["hotword"] != hotword):
            loaded_model["model"] = initialize_model(model_type, hotword)

@app.post("/asr",tags = ["ASR"],summary = "DAMO ASR Model using paraformer larger asr 16k vocab8404 pytorch")
async def predict(items:Audio):
    """

    - **file**: Base64 encode audio file.
    - **format**: Default is wav. Choosing between wav or pcm. If choosing pcm, should also post param: audio_fs. 
    - **audio_fs**: Default is 16000. Only useful while format is pcm.
    - **hotword**: Default is None. Only use while model_type is hotword.
    - **model_type**: Default is normal. Choosing from normal/long/hotword.

    """
    decoded_data = base64.b64decode(items.file)
    load_model(model_type=items.model_type, hotword = items.hotword)
    rec_result = loaded_model["model"](audio_in = decoded_data)
    print(rec_result)
    return rec_result

