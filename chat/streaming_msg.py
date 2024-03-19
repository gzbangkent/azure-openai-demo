
import logging
import os
from openai import AzureOpenAI, AsyncAzureOpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai import Stream

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ModelAccessInfo

log=logging.getLogger(__name__)

# modelAccessInfo=ModelAccessInfo.find_model_access_info("gpt-4")[0]
modelAccessInfo=ModelAccessInfo.find_model_access_info("gpt-35-turbo", model_version='1106')[0]

client = AzureOpenAI(
  azure_endpoint = modelAccessInfo.endpoint,
  api_key=modelAccessInfo.api_key,  
  api_version="2024-02-15-preview"
)


message_text = [
  {"role":"system","content":"你是一个中国导游, 友好且有耐心的回答游客咨询的问题"},
  {'role': 'user', 'content': "广州有什么地方值得外地人游玩?"}
]

stream:Stream[ChatCompletionChunk]=client.chat.completions.create(
  model=modelAccessInfo.deployment_name, # model = "deployment_name"
  messages = message_text,
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None,
  stream=True
)

for chunk in stream:
  log.debug(chunk)
  if chunk.choices:
    print(chunk.choices[0].delta.content,end='')