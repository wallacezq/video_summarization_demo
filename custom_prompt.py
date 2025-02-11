from langchain_core.runnables import RunnableConfig
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import (
    BaseMessage,
    get_buffer_string,
)


#from pydantic import Field
from typing import List, Optional, Sequence, Any, Iterator, Dict

class CustomChatPromptValue(ChatPromptValue):
    tokenizer: Optional[object] = None

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
    #tokenizer: object = Field(default=None)
    tokenizer: Optional[object] = None
    def __init__(self, template: str, tokenizer: object=None):
        super().__init__(messages=template)
        self.tokenizer = tokenizer

    
    def invoke(self, input: dict, config: Optional[RunnableConfig] = None) -> CustomChatPromptValue:
        # Custom formatting logic
        formatted_messages = super().format_messages(**input)
        return CustomChatPromptValue(tokenizer=self.tokenizer, messages=formatted_messages)    

