#############################
# 场景: 数据分析助手
# 技术要点: Assistant-API的'代码解释器'功能
# 流程: 
# 1. 配置助手使用'代码解释器'功能
# 2. 上传需要分析的股票csv数据文件
# 3. 用户提问, 要求进行多维度分析
# 4. 助手执行分析, 根据用户要求生成相关代码, 用沙箱环境执行, 综合执行结果内容进行回复
#############################
import os
import logging
import time
from typing import List
from openai import AzureOpenAI, AsyncAzureOpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.beta.thread import Thread
from openai.types.beta.threads import Run, ThreadMessage
from openai.types.beta.threads.runs import RunStep

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env import ModelAccessInfo

log=logging.getLogger(__name__)

def downloadFile(client:AzureOpenAI, file_id:str, outputFp:str):
  log.info(f'下载文件{file_id}...')
  f = client.files.content(file_id)
  fBytes = f.read()

  with open(outputFp, "wb") as file:
      file.write(fBytes)

def downloadFileIfNotExists(client:AzureOpenAI, file_id:str, outputFp:str, fileType:str=None):
  suffix=''
  if fileType=='image_file':
    suffix='.png'
  
  outputFp=f'{outputFp}{suffix}'
  if not os.path.exists(outputFp):
    downloadFile(client, file_id, outputFp)
  return outputFp


__MSG_MAP__={}
def getMsgById(client:AzureOpenAI, threadId:str, msgId:str):
  k=f'{threadId}_{msgId}'
  msg=__MSG_MAP__.get(k)
  if msg:
    return msg
  
  msg=client.beta.threads.messages.retrieve(msgId,thread_id=threadId)
  __MSG_MAP__[k]=msg
  return msg

def appendMsgAndRun(client:AzureOpenAI, thread:Thread, msgContent:str, fileIds:List[str]=None):
  # 添加提问到会话 并 执行回复
  
  message = client.beta.threads.messages.create(
      thread_id=thread.id,
      role="user",
      content=msgContent,
      file_ids=fileIds if fileIds else []
  )
  log.debug(message.json(indent=2,ensure_ascii=False))
  log.info(f'已添加提问到会话: {message.id}')
  
  # 执行回复
  run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
  )
  return run

def waitForRunComplete(client:AzureOpenAI, thread:Thread, run:Run):
  # 等待助手回复完成
  
  # 查询目前状态
  run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
  )
  status = run.status

  stepPage=client.beta.threads.runs.steps.list(run_id=run.id, thread_id=thread.id, limit=100, order='asc')
  for stepNo,step in enumerate(stepPage):
    log.info(f'********步骤{stepNo+1}: {step.status}********')
    # log.debug(step.json(indent=2,ensure_ascii=False))
    dispRunStep(client,thread, step)

  # Wait till the assistant has responded
  while status not in ["completed", "cancelled", "expired", "failed"]:
    log.info(f'等待助手回复...{status}')
    time.sleep(5)
    run = client.beta.threads.runs.retrieve(thread_id=thread.id,run_id=run.id)
    status = run.status
    
    stepPage=client.beta.threads.runs.steps.list(run_id=run.id, thread_id=thread.id, limit=100, order='asc')
    for stepNo,step in enumerate(stepPage):
      log.info(f'步骤{stepNo+1}: {step.status}')
      # log.debug(step.json(indent=2,ensure_ascii=False))
      dispRunStep(client, thread, step)

  log.info(f'助手回复完成: {status}')

def dispMsg(client:AzureOpenAI, msg:ThreadMessage):
  log.info(f'--------[{msg.role}]消息: {msg.id} {msg.file_ids}')
  if msg.file_ids and msg.role!='user':
    for fileId in msg.file_ids:
      log.info(f'下载文件: {fileId}')
      downloadFileIfNotExists(client, fileId, f'{fileId}')
  
  for content in msg.content:
    if content.type=='text':
      log.info(f'文本内容: {content.text.value}')
    elif content.type=='image_file':
      fileId=content.image_file.file_id
      fp=downloadFileIfNotExists(client, fileId, fileId, 'image_file')
      log.info(f'图片路径: {fp}')
    else:
      log.info(f'其他内容: {content.json(indent=2,ensure_ascii=False)}')
  log.info(f'----------------')

def dispRunStep(client:AzureOpenAI, thread:Thread, runStep:RunStep):
  log.info(f'--------步骤: {runStep.id} {runStep.status} {runStep.step_details.type}')
  if runStep.step_details.type == "tool_calls":
    for toolCall in runStep.step_details.tool_calls:
      if toolCall.type == "code_interpreter":
        log.info(f'执行代码: \n{toolCall.code_interpreter.input}')
        log.info(f'执行输出:')
        if toolCall.code_interpreter.outputs:
          for output in toolCall.code_interpreter.outputs:
            if output.type=='logs':
              log.info(f'日志输出: \n{output.logs}')
            elif output.type=='image':
              log.info(f'图片输出: {output.image.file_id}')
            else:
              log.info(f'未知输出: {output}')
      else:
        log.info(f'其他工具调用: {toolCall.type}')
  elif runStep.step_details.type == "message_creation":
    if runStep.status != "completed":
      log.info('执行消息生成中...')
    else:
      log.info('执行消息生成完成')
      stepMsg=getMsgById(client, thread.id, runStep.step_details.message_creation.message_id)
      dispMsg(client, stepMsg)
  else:
    log.info(f'其他类型步骤: {runStep.step_details.type}')
    log.info(runStep.step_details)
  log.info(f'----------------')
    

modelAccessInfo=ModelAccessInfo.find_model_access_info("gpt-4", model_version='1106-Preview',region='australiaeast')[0]
# modelAccessInfo=ModelAccessInfo.find_model_access_info("gpt-35-turbo", model_version='1106',region='australiaeast')[0]

client = AzureOpenAI(
  azure_endpoint = modelAccessInfo.endpoint,
  api_key=modelAccessInfo.api_key,  
  api_version="2024-02-15-preview",#"2024-02-01",
)

# # 上传文件
# stockFile = client.files.create(
#   file=open("stock.csv", "rb"),
#   purpose='assistants'
# )
# 获取文件信息
stockFile=client.files.retrieve('assistant-UIKyu937T9Dn7s1A2TP92SZc')
log.debug(stockFile.json(indent=2,ensure_ascii=False))
fontFile=client.files.retrieve('assistant-75YxsEC69LXDmWDvoPGZfe2Y')
log.debug(fontFile.json(indent=2,ensure_ascii=False))

# # 创建助手
# assistant = client.beta.assistants.create(
#     name="股票分析助手",
#     instructions="你是一个专业的股票分析助手，可以帮助用户多维度分析股票数据。",
#     tools=[{"type": "code_interpreter"}],
#     model=modelAccessInfo.deployment_name
# )

# 获取助手信息
# asstId="asst_7FXWn1qShZJTNoQBm4HcgA1u"#3.5
asstId="asst_zUtcJuQ2KHzePOW9jVX2Az7f"#4
assistant = client.beta.assistants.retrieve(asstId)
log.debug(assistant.json(indent=2,ensure_ascii=False))
log.info(f'助手: {assistant.id}')

# 创建会话
thread = client.beta.threads.create()
# 获取会话信息
# thread = client.beta.threads.retrieve('thread_kHw4LqG8NbayIC0K2qPMmHN7')
log.debug(thread.json(indent=2,ensure_ascii=False))
log.info(f'会话: {thread.id}')

# #添加提问到会话
run = appendMsgAndRun(client, thread, f'''csv文件是上周A股云计算行业的涨跌幅数据。请从多维度进行详细的分析, 包含哪些个股有哪些突出的表现。

绘制图表时, 按以下步骤应用中文字体:
1. 将我上传的zip文件(/mnt/data/{fontFile.id})解压后保存到 /mnt/data/fonts/msyh.ttc
2. 设置我上传的字体为全局字体

全局设定自定义字体示例代码如下:

import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
import zipfile

# 1. 解压中文字体压缩包
zip_path = '/mnt/data/{fontFile.id}'
extract_path = '/mnt/data/fonts'

if not os.path.exists(extract_path):
    os.makedirs(extract_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# 2. 全局设置使用中文字体
font_manager.fontManager.addfont("/mnt/data/fonts/msyh.ttc")
plt.rcParams['font.family']=['Microsoft YaHei']
''', [stockFile.id,fontFile.id])
# 获取执行信息
# run =  client.beta.threads.runs.retrieve('run_46yIQDNZwbjWucEyx47YM26D', thread_id=thread.id)
log.debug(run.json(indent=2,ensure_ascii=False))
log.info(f'助手开始执行...: {run.id}')

while True:  
  waitForRunComplete(client, thread, run)

  messages = client.beta.threads.messages.list(
    thread_id=thread.id,
    order="asc",
  )

  # log.debug(messages.json(indent=2,ensure_ascii=False))
  for msgNo,msg in enumerate(messages):
    log.info(f'********消息{msgNo+1}********')
    dispMsg(client, msg)
        
  newMsgContent=input('请继续提问: ')
  if not newMsgContent:
    break
  
  run = appendMsgAndRun(client, thread, newMsgContent)
  
