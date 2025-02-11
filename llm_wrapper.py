import torch
from langchain_core.outputs import GenerationChunk
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from ipex_llm.transformers import AutoModelForCausalLM
from decord import VideoReader, cpu 
from pydantic import Field
from typing import List, Optional, Sequence, Any, Iterator, Dict
from PIL import Image
import cv2
import numpy as np

MAX_NUM_FRAMES=8

def encode_video_vllava(
        video_path,
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8):
    
    cv2_vr = cv2.VideoCapture(video_path)
    duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
    frames=[]
    
    #print(f"frame_id_list: {frame_id_list}")

    video_data = []
    for frame_idx in frame_id_list:
        cv2_vr.set(1, frame_idx)
        _, frame = cv2_vr.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cv2_vr.release()
    return np.stack(frames)

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
    max_token_length: Optional[int] = 128

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
        
        
class VideoLLaVAWrapper(LLM):
    model: Optional[object] = None 
    processor: Optional[object] = None
    device: Optional[str] = 'cpu'
    max_token_length: Optional[int] = 128
    prompt_length:  Optional[int]   = 0

    @property
    def _llm_type(self) -> str:
        return "Custom Video Llava LVM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,       
    ) -> str:

        # Set decode params for video. TO DO. Make class attributes
        params={}
        inputs=None

        # Format prompt for "model.chat()"
        splitted_prompt = prompt.split('$$$$')
        
        frames = None
        
        if len(splitted_prompt) == 2:
           video_fh, question = splitted_prompt
           frames = encode_video_vllava(video_fh)
        else:
           question = splitted_prompt 
        
        # format the prompt for vllava
        if len(splitted_prompt) == 2:
          question = f"<video>\n{question}"
        # TODO: Add image support
        #if image is not None:
        #   prompt_text = "<image>\n" + prompt_text       
        question = f"USER: {question} ASSISTANT:"        
        
        
        with torch.inference_mode():
          inputs = self.processor(text=question, videos=frames, padding=True, return_tensors="pt")
          
          self.prompt_length=inputs['input_ids'].shape[1]
          
          #print(f"prompt_length: {self.prompt_length}")
          
          # Move inputs to the XPU
          inputs = {k: v.to(self.device) for k, v in inputs.items()}    
          
          # Chat
          generate_ids = self.model.generate(
              **inputs,
              max_length=self.max_token_length,
          )
        
          if 'xpu' in self.device and torch.xpu.is_available():
             torch.xpu.synchronize()
             
          #print(f"generate_ids: {generate_ids}\nlen(generate_ids): {len(generate_ids)}")
          #generate_ids = generate_ids.cpu()
          
          result = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
          #result = self.processor.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return result        
        
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
        

