from langchain_community.chat_models import ChatLlamaCpp
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser,JsonOutputKeyToolsParser
from langchain_postgres import PostgresChatMessageHistory,PGEngine
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RouterInput,RunnableWithMessageHistory,RunnableParallel,Runnable,RouterRunnable,RunnableBranch,RunnablePick
from  langchain_core.prompts import ChatPromptTemplate,PromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import os
import gc
import ctypes
import os
from App.core.LoggingInit import get_core_logger
from pathlib import Path
from uuid import uuid4
logger=get_core_logger(__name__)

class Lc_Connector:
    def __init__(self,model_path:Path):
        self.modelPath=model_path
        self.current_active_model={None}

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc, tb):
        pass
    def __del__(self):
        del self.current_active_model
        gc.collect()
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)  
 
    def init_llm(self, **kwargs)->ChatLlamaCpp:
        """
        Initializes the LlamaCpp language model with optimized parameters.

        This method configures the ChatLlamaCpp instance, specifically targeting 
        stable performance on local hardware (e.g., AMD RX 5600 XT).

        Args:
            **kwargs: Arbitrary keyword arguments for model configuration.
                temperature (float): Controls randomness. Defaults to 0.65.
                top_p (float): Nucleus sampling probability. Defaults to 0.9.
                top_k (int): Top-k sampling limit. Defaults to 30.
                streaming (bool): Whether to stream tokens. Defaults to False.
                repeat_penalty (float): Penalty for repeated tokens. Defaults to 1.15.
                max_token (int): Maximum tokens to generate. Defaults to 1024.
                n_batch (int): Batch size for prompt processing. Defaults to 512.
                n_ctx (int): Context window size. Defaults to 4096.
                n_threads (int): Number of CPU threads to use. Defaults to 6.
                gpu_layer (int): Number of layers to offload to GPU. -1 for all. Defaults to -1.
                stop (list[str]): Tokens that trigger generation stop.

        Returns:
            ChatLlamaCpp: A configured instance of the LLM.

        Raises:
            Exception: If the model file is missing or VRAM allocation fails.
        """
        try:
            llm = ChatLlamaCpp(
                model_path=str(self.model_path), # Ensure it's a string for the loader
                temperature=kwargs.get("temperature", 0.65),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 30),
                streaming=kwargs.get("streaming", False),
                repeat_penalty=kwargs.get("repeat_penalty", 1.15),
                max_tokens=kwargs.get("max_token", 1024),
                n_batch=kwargs.get("n_batch", 512),
                n_ctx=kwargs.get("n_ctx", 4096),
                n_threads=kwargs.get("n_threads", 6),
                n_gpu_layers=kwargs.get("gpu_layer", -1), # Fixed 'gou' typo
                verbose=False,
                stop=kwargs.get("stop", ["<|endoftext|>", "<|im_end|>", "<|object_ref|>", "User:", "Human:"]),
            )
            return llm
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise   
        # finally:
        #     del self.current_active_model
        #     libc = ctypes.CDLL("libc.so.6")
        #     libc.malloc_trim(0) # Tells the OS to reclaim unused memory from the process
    
    def create_llm(self,name:str):
        try:
            
            pass
        except Exception as e:
            logger.error(f"Error create llm due to {e}")
            raise