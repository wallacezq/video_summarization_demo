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

from transformers import AutoTokenizer, AutoProcessor, VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
from ipex_llm import optimize_model
from ipex_llm.optimize import low_memory_init, load_low_bit

from pathlib import Path
import cv2
import numpy as np

#from langchain.llms.base import LLM
#from pydantic import Field
#from typing import List, Optional, Sequence, Any, Iterator, Dict
from langchain.prompts import PromptTemplate
from llm_wrapper import MiniCPMV26Wrapper, Llama32Wrapper, VideoLLaVAWrapper
from custom_prompt import CustomChatPromptValue, CustomChatPromptTemplate
from VideoChunkLoader import VideoChunkLoader


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


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate` API for LanguageBind/Video-LLaVA-7B-hf')
    parser.add_argument('--repo-id-or-model-path', type=str,
                        help='The Hugging Face or ModelScope repo id for the LanguageBind/Video-LLaVA-7B-hf model to be downloaded'
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
    #parser.add_argument('--stream', action='store_true',
    #                    help='Whether to chat in streaming mode')
    # parser.add_argument('--modelscope', action="store_true", default=False, 
    #                    help="Use models from modelscope")

    args = parser.parse_args()

    #if args.modelscope:
    #    from modelscope import AutoTokenizer
    #    model_hub = 'modelscope'
    #else:
    model_hub = 'huggingface'
    
    model_path = args.repo_id_or_model_path if args.repo_id_or_model_path else "LanguageBind/Video-LLaVA-7B-hf"
    image_path = args.image_url_or_path

    lowbit_path = args.lowbit_path
    
    """   
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
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", trust_remote_code=True) #, torch_dtype=torch.float16)
        processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", trust_remote_code=True)
        model = optimize_model(model, low_bit="sym_int4").to(device)
    """   
    model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", trust_remote_code=True) #, torch_dtype=torch.float16)
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", trust_remote_code=True)
    model = optimize_model(model, low_bit="sym_int4").to('xpu')
        
    if lowbit_path and not os.path.exists(lowbit_path):
        model.save_low_bit(lowbit_path)

    
    vllava_wrapper =  VideoLLaVAWrapper(model=model,
                                        device='xpu',
                                        processor=processor,
                                        max_token_length=256)
                                        

    # create template for input
    prompt = PromptTemplate(
          input_variables=["video", "question"],
          template="{video}$$$${question}"  # we use $$$$ as delimeter to separate video and question
          )
          
    # this prompt is to test if minicpm is good summary writer
    prompt_textonly = PromptTemplate(input_variables=["question"],
                                     template="{question}")

    
    # Create vllava pipeline and invoke
    vllava_chain =  prompt | vllava_wrapper
    
    vllava_chain_textonly = prompt_textonly | vllava_wrapper  # for testing if minicpm is good for making summary
    
    # load a Llama3.2-3B-Instruct model
    llama32_path = "meta-llama/Llama-3.2-3B-Instruct"
    model_llama32 = AutoModelForCausalLM.from_pretrained(llama32_path,
                                                 load_in_4bit=True,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True)
                                                 
    model_llama32 = model_llama32.half().to('xpu')
    tokenizer_llama32 = AutoTokenizer.from_pretrained(llama32_path, trust_remote_code=True)
    
    llama32_template = CustomChatPromptTemplate(template=[("system", "You are helpful AI bot."), ("human", "{input}"),], tokenizer=tokenizer_llama32)    
    
    
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
        output = vllava_chain.invoke(inputs)

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
                                       max_token_length=2048, temperature=0.6, top_p=0.9, top_k=50)
    
    llama32_chain = llama32_template | llama32_wrapper
    
    # Use llama3.2-2B to generate summary
    output = llama32_chain.invoke({"input": inputs})   
    
    # Use minicpm to generate summary
    #output = minicpm_chain_textonly.invoke({"question": inputs})
    

    print("\nOverall video summary inference time: {} sec\n".format(time.time() - overall_summ_st_time))
    
    print("\nTotal Inference time: {} sec\n".format(time.time() - tot_st_time))
    
    output_handler(output)

    # python test.py --image-url-or-path ../../vllava/examples/rag_videos/op_1_0320241830.mp4 --prompt "describe the video and note the humam behavior and their interaction with store merchandise"

