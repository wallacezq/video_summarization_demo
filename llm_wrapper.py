import torch

from langchain_core.outputs import GenerationChunk
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from ipex_llm.transformers import AutoModelForCausalLM
from decord import VideoReader, cpu 
from pydantic import Field
from typing import List, Optional, Sequence, Any, Iterator, Dict
from PIL import Image

MAX_NUM_FRAMES=8

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

class MiniCPMV26Wrapper(LLM):
    model: object 
    tokenizer: object
    max_token_length: int = Field(default=128)

    @property
    def _llm_type(self) -> str:
        return "Custom MiniCPM-V-2_6"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:

        # Set decode params for video. TO DO. Make class attributes
        params={}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution >  448*448

        # Format prompt for "model.chat()"
        splitted_prompt = prompt.split('$$$$')
        is_video = False
        
        frames = []
        
        if len(splitted_prompt) == 2:
           video_fh, question = splitted_prompt
           frames = encode_video(video_fh)
        else:
           question = splitted_prompt 
           
        msgs = [{'role': 'user', 'content': frames + [question]},]
        
        # Chat
        response = self.model.chat(
            image=None,
            msgs=msgs,
            max_length=self.max_token_length,
            tokenizer=self.tokenizer,
            **params
        )
        
        if torch.xpu.is_available():
             torch.xpu.synchronize()
        
        return response
        
class Llama32Wrapper(LLM):
    model: object 
    tokenizer: object
    #max_token_length: int = Field(default=2048)
    #prompt_length: int = Field(default=0)
    max_token_length:   Optional[int]   = 2048
    top_p:              Optional[float] = 0.1
    top_k:              Optional[int]   = 50
    temperature:        Optional[float] = 0.6
    do_sample:          Optional[bool]  = True
    prompt_length:      Optional[int]   = 0

    @property
    def _llm_type(self) -> str:
        return "Custom Llama3.2 Wrapper"
        
    @property
    def _get_model_default_parameters(self):
      self.do_sample = True if self.temperature != 0.0 else False
      return {
        "max_length": self.max_token_length,
        "top_k": self.top_k,
        "top_p": self.top_p,
        "temperature": self.temperature,
        "do_sample": self.do_sample,      
      }        

    def _call(
        self,
        prompt: List[object],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,        
    ) -> str:

        # Set decode params for video. TO DO. Make class attributes
        params={}
       
        llama32_formatted_template = []
        
        prompt_str = prompt
        
        input_ids = self.tokenizer.encode(prompt_str, return_tensors="pt").to('xpu')
        output = self.model.generate(input_ids, **self._get_model_default_parameters)
                                
        self.prompt_length = input_ids.shape[1]                        
        
        if torch.xpu.is_available():
             torch.xpu.synchronize()
        
        output = output.cpu()     
        # Chat
        response = self.tokenizer.decode(output[0][self.prompt_length:], skip_special_tokens=True)     
        
        return response
        

