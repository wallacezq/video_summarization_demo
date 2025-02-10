#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, APIRouter
import tempfile
import uvicorn
from starlette.background import BackgroundTasks
import os
import time
import argparse
import requests
import torch
from PIL import Image
from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoProcessor
from pathlib import Path
import cv2
import numpy as np
from decord import VideoReader, cpu 

from dataclasses import dataclass
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import (
    BaseMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig

from langchain.llms.base import LLM

from pydantic import Field
from typing import List, Optional, Sequence

from VideoChunkLoader import VideoChunkLoader

MAX_NUM_FRAMES=8

#logger = logging.getlogger('uvicorn.error')



#@asynccontextmanager
#async def lifespan(app: FastAPI):
#   global model

def output_handler(text: str,
                   filename: str = None,
                   mode: str = None,
                   verbose: bool = True):
    
    # Print to terminal
    if verbose:
        print(text)

    # Write to file, if requested
    if filename:
        with open(filename, mode) as FH:
            print(text, file=FH)


def load_video(
       video_path,
       clip_start_sec=0.0,
       clip_end_sec=None,
       num_frames=8):
    
    cv2_vr = cv2.VideoCapture(video_path)
    duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
    frames=[]

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
    
class CustomChatPromptValue(ChatPromptValue):
    tokenizer: object = Field(default=None)
    #messages: Sequence[BaseMessage]    
    
    def __init__(self, tokenizer, messages, **kwargs):
        super().__init__(messages=messages, **kwargs)    
        self.tokenizer = tokenizer
        
    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.
        This is used to determine the namespace of the object when serializing.
        Defaults to ["langchain", "prompts", "chat"].
        """
        return ["langchain", "prompts", "chat"]        

    def to_string(self) -> str:
        llama32_formatted_template = []
        for entry in self.messages:
           #print(f"\nentry: {entry}, type: {entry.type}\n")
           llama32_formatted_template.append({ 'role': entry.type if entry.type not in ["human"] else "user", 'content':  entry.content})
         
        #print(f"formatted: {llama32_formatted_template}")
        
        # Format prompt for "model.chat()"
        prompt_str = self.tokenizer.apply_chat_template(llama32_formatted_template, add_generation_prompt=True, tokenize=False)        
        
        #print(f"\nPROMP_STR: {prompt_str}")        
        return prompt_str

class CustomChatPromptTemplate(ChatPromptTemplate):
    tokenizer: object = Field(default=None)
    def __init__(self, template: str, tokenizer: object=None):
        super().__init__(messages=template)
        self.tokenizer = tokenizer

    
    def invoke(self, input: dict, config: Optional[RunnableConfig] = None) -> CustomChatPromptValue:
        # Custom formatting logic
        formatted_messages = super().format_messages(**input)
        return CustomChatPromptValue(tokenizer=self.tokenizer, messages=formatted_messages)    

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
    max_token_length: int = Field(default=2048)

    @property
    def _llm_type(self) -> str:
        return "Custom Llama3.2 Wrapper"

    def _call(
        self,
        prompt: List[object],
        stop: Optional[List[str]] = None,
    ) -> str:

        # Set decode params for video. TO DO. Make class attributes
        params={}
       
        llama32_formatted_template = []
        
        prompt_str = prompt
        
        input_ids = self.tokenizer.encode(prompt_str, return_tensors="pt").to('xpu')
        output = self.model.generate(input_ids,
                                max_length=self.max_token_length)
        
        if torch.xpu.is_available():
             torch.xpu.synchronize()
        
        output = output.cpu()     
        # Chat
        response = self.tokenizer.decode(output[0], skip_special_tokens=False)     
        
        return response        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `chat()` API for openbmb/MiniCPM-V-2_6 model')
    parser.add_argument('--repo-id-or-model-path', type=str,
                        help='The Hugging Face or ModelScope repo id for the MiniCPM-V-2_6 model to be downloaded'
                             ', or the path to the checkpoint folder')
    parser.add_argument("--lowbit-path", type=str,
        default="",
        help="The path to the saved model folder with IPEX-LLM low-bit optimization. "
             "Leave it blank if you want to load from the original model. "
             "If the path does not exist, model with low-bit optimization will be saved there."
             "Otherwise, model with low-bit optimization will be loaded from the path.",
    )
    parser.add_argument('--image-url-or-path', type=str,
                        default='http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg',
                        help='The URL or path to the image to infer')
    parser.add_argument('--prompt', type=str, default="What is in the image?",
                        help='Prompt to infer')
    parser.add_argument('--stream', action='store_true',
                        help='Whether to chat in streaming mode')
    parser.add_argument('--modelscope', action="store_true", default=False, 
                        help="Use models from modelscope")

    args = parser.parse_args()

    if args.modelscope:
        from modelscope import AutoTokenizer
        model_hub = 'modelscope'
    else:
        from transformers import AutoTokenizer
        model_hub = 'huggingface'
    
    model_path = args.repo_id_or_model_path if args.repo_id_or_model_path else \
        ("OpenBMB/MiniCPM-V-2_6" if args.modelscope else "openbmb/MiniCPM-V-2_6")
    image_path = args.image_url_or_path

    lowbit_path = args.lowbit_path
    
    if not lowbit_path or not os.path.exists(lowbit_path):
        # Load model in 4 bit,
        # which convert the relevant layers in the model into INT4 format
        # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
        # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
        model = AutoModel.from_pretrained(model_path, 
                                        load_in_low_bit="sym_int4",
                                        optimize_model=True,
                                        trust_remote_code=True,
                                        use_cache=True,
                                        modules_to_not_convert=["vpm", "resampler"],
                                        model_hub=model_hub)

        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
    else:
        model = AutoModel.load_low_bit(lowbit_path, 
                                       optimize_model=True,
                                       trust_remote_code=True,
                                       use_cache=True,
                                       modules_to_not_convert=["vpm", "resampler"])
        tokenizer = AutoTokenizer.from_pretrained(lowbit_path,
                                                  trust_remote_code=True)
    
    model.eval()

    if lowbit_path and not os.path.exists(lowbit_path):
        processor = AutoProcessor.from_pretrained(model_path,
                                                trust_remote_code=True)
        model.save_low_bit(lowbit_path)
        tokenizer.save_pretrained(lowbit_path)
        processor.save_pretrained(lowbit_path)

    model = model.half().to('xpu')
    
    minicpm_wrapper = MiniCPMV26Wrapper(model=model,
                                        tokenizer=tokenizer,
                                        max_token_length=512)
                                        

    # create template for input
    prompt = PromptTemplate(
          input_variables=["video", "question"],
          template="{video}$$$${question}"  # we use $$$$ as delimeter to separate video and question
          )

    prompt_textonly = PromptTemplate(input_variables=["question"],
                                     template="{question}")

    
    # Create pipeline and invoke
    minicpm_chain =  prompt | minicpm_wrapper
    
    minicpm_chain_textonly = prompt_textonly | minicpm_wrapper  # for testing if minicpm is good for making summary
    
    # load a Llama3.2-2B model
    llama32_path = "meta-llama/Llama-3.2-3B-Instruct"
    model_llama32 = AutoModelForCausalLM.from_pretrained(llama32_path,
                                                 load_in_4bit=True,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True)
                                                 
    model_llama32 = model_llama32.half().to('xpu')
    tokenizer_llama32 = AutoTokenizer.from_pretrained(llama32_path, trust_remote_code=True)
    
    llama32_template = CustomChatPromptTemplate(template=[("system", "You are helpful AI bot."), ("human", "{input}"),], tokenizer=tokenizer_llama32)    
    
    
    """    
    inputs = {"video": image_path, "question": args.prompt}
    
    st = time.time()
    output = chain.invoke(inputs)
    end = time.time()
    
    print(f'Inference time: {end-st} s')
    
    print(output)
    """
    
    tot_st_time = time.time()

    loader = VideoChunkLoader(
        video_path=image_path,
        chunking_mechanism="sliding_window",
        #specific_intervals=[
        #    {"start": 0, "duration": 7},
        #    {"start": 3, "duration": 10},
        #],
        chunk_duration=6,
        chunk_overlap=2
        
        )
        
        
    chunk_summaries = []
    
    for doc in loader.lazy_load():
        # Log metadata
        output_handler(str(f"Chunk Metadata: {doc.metadata}"),
                       filename=None)
        output_handler(str(f"Chunk Content: {doc.page_content}"),
                       filename=None)
        
        # Generate sumarries
        chunk_st_time = time.time()
        inputs = {"video": doc.metadata['chunk_path'], "question": args.prompt}        
        output = minicpm_chain.invoke(inputs)

        # Log output
        output_handler(output)
        chunk_summaries.append(f"Start time: {doc.metadata['start_time']} End time: {doc.metadata['end_time']}\n" + output)
        print("\nChunk Inference time: {} sec\n".format(time.time() - chunk_st_time))
                    

    # Summarize the full video, using the subsections summaries from each chunk
    
    overall_summ_st_time = time.time()
    full_summ_prompt = 'The following are summaries of subsections of a video. Each subsection summary is separated by the delimiter ">|<". Each subsection summary will start with the start and end timestamps of the subsection relative to the full video. Please create a summary of the overall video, highlighting all important information, including timestamps:\n\n{}'    
    inputs = full_summ_prompt.format("\n>|<\n".join(chunk_summaries))
    
    llama32_wrapper = Llama32Wrapper(model=model_llama32,
                                       tokenizer=tokenizer_llama32,
                                       max_token_length=2048)
    
    llama32_chain = llama32_template | llama32_wrapper
    
    # Use llama3.2-2B to generate summary
    output = llama32_chain.invoke({"input": inputs})   
    
    # Use minicpm to generate summary
    #output = minicpm_chain_textonly.invoke({"question": inputs})
    

    print("\nOverall video summary inference time: {} sec\n".format(time.time() - overall_summ_st_time))
    
    print("\nTotal Inference time: {} sec\n".format(time.time() - tot_st_time))
    
    output_handler(output)

"""                                       
    images = None
    image = None

    query = args.prompt
    if os.path.exists(image_path):
       if image_path.endswith((".mp4", ".avi")):
           images = encode_video(image_path)
       else:
           image = Image.open(image_path).convert('RGB')
    else:
       image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')

    # Generate predicted tokens
    # here the prompt tuning refers to https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/main/README.md
    
    if images is not None:
        msgs = [{'role': 'user', 'content': images + [args.prompt]}, ]
    else:  
        msgs = [{'role': 'user', 'content': [image, args.prompt]}]

    # ipex_llm model needs a warmup, then inference time can be accurate
    model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
    )


    params = {}
    params["use_image_id"] = False
    params["max_slice_number"] = 2 

    if args.stream:
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            stream=True,
            **params
        )

        print('-'*20, 'Input Image', '-'*20)
        print(image_path)
        print('-'*20, 'Input Prompt', '-'*20)
        print(args.prompt)
        print('-'*20, 'Stream Chat Output', '-'*20)
        for new_text in res:
            print(new_text, flush=True, end='')
    else:
        st = time.time()
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )
        torch.xpu.synchronize()
        end = time.time()

        print(f'Inference time: {end-st} s')
        print('-'*20, 'Input Image', '-'*20)
        print(image_path)
        print('-'*20, 'Input Prompt', '-'*20)
        print(args.prompt)
        print('-'*20, 'Chat Output', '-'*20)
        print(res)
"""
