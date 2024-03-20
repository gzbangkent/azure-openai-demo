#############################
# 场景: 运营AI助手
# 技术要点: function-call(1106)
# 流程: 
# 1. 注册业务函数
# 2. 用户提问
# 3. AI助手根据上下文, 调用相应业务函数
# 4. 函数执行, 返回结果
# 5. AI助手根据函数返回内容综合回复用户
#############################
from datetime import date, datetime
import json
import os
import logging
from openai import AzureOpenAI, AsyncAzureOpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai import Stream

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ModelAccessInfo

log=logging.getLogger(__name__)

# modelAccessInfo=ModelAccessInfo.find_model_access_info("gpt-4", model_version='1106-Preview')[0]
modelAccessInfo=ModelAccessInfo.find_model_access_info("gpt-35-turbo", model_version='1106')[0]

client = AzureOpenAI(
  azure_endpoint = modelAccessInfo.endpoint,
  api_key=modelAccessInfo.api_key,  
  api_version="2024-02-15-preview"#'2023-12-01-preview'
)

# 目前支持的模型及版本
# gpt-4-1106-preview
# gpt-35-turbo-1106

def list_salesman_performance(startDate:str, endDate:str):
  log.info(f'执行查询销售业绩{startDate}到{endDate}...')
  return {
    '张一': 11111,
    '张二': 22222,
    '张三': 33333,
    '张四': 44444,
  }

def list_user():
  log.info(f'执行查询员工信息...')
  return [
    {'name':'张一', 'deptName':'销售A部', 'role':'销售员', 'age':38, 'sex':'男', 'active':True},
    {'name':'张二', 'deptName':'销售A部', 'role':'销售员', 'age':28, 'sex':'女', 'active':True},
    {'name':'张三', 'deptName':'销售B部', 'role':'销售员', 'age':23, 'sex':'男', 'active':False},
    {'name':'张四', 'deptName':'销售B部', 'role':'销售员', 'age':25, 'sex':'女', 'active':True},
    {'name':'张五', 'deptName':'运营部', 'role':'运营', 'age':32, 'sex':'男', 'active':True},
    {'name':'张六', 'deptName':'技术部', 'role':'技术', 'age':37, 'sex':'男', 'active':True},
  ]

def send_mail(toUserMail:str, subject:str, content:str):
  log.info(f'发送邮件给{toUserMail} 主题:{subject} 内容:{content}...')
  return True

functions={
  'list_salesman_performance':{
    'python_function': list_salesman_performance,
    'description': "查询指定日期范围内所有销售员工各自的业绩",
    "parameters": {
        "type": "object",
        "properties": {
            "startDate": {
                "type": "string",
                "description": "开始日期字符串, 格式: YYYY-MM-DD",
            },
            "endDate": {
                "type": "string",
                "description": "结束日期字符串, 含结束日期当天. 格式: YYYY-MM-DD",
            },
        },
        "required": ["startDate", "endDate"],
    },
  },
  'list_user':{
    'python_function': list_user,
    'description': "查询所有员工的信息. name:姓名, deptName:部门, role:角色, age:年龄, sex:性别, active:是否在职",
    "parameters": {
        "type": "object",
        "properties": {
        },
        "required": [],
    },
  },
  'send_mail':{
    'python_function': send_mail,
    'description': "发送邮件给指定用户",
    "parameters": {
        "type": "object",
        "properties": {
            "toUserMail": {
                "type": "string",
                "description": "发送邮件的目标邮箱地址",
            },
            "subject": {
                "type": "string",
                "description": "邮件主题",
            },
            "content": {
                "type": "string",
                "description": "邮件内容",
            },
        },
        "required": ["toUserMail","subject", "content"],
    },
  },
}

def genFuncTools():
  tools=[]
  for func_name, func_info in functions.items():
    tools.append({
        'type':'function',
        'function':{
          'name': func_name,
          'description': func_info['description'],
          'parameters': func_info['parameters']
        }
      })
  return tools

tools=genFuncTools()

sysMsg=f'''你是我的智能助手, 严谨且精确的结合当前工具与你自身的知识解答我的问题. 以下为上下文相关信息:

当前时间为:{datetime.now().strftime("%Y年%m月%d日 %H:%M:%S 星期%w")}
我的资料是:{{"姓名":"迪迦", "年龄":38, "性别":"男", "角色":"CEO", "邮箱":"dijia@bangkeyun.cn"}}'''

questionMsg=f'上个自然周各销售部门的业绩情况如何? 总体销售额是多少? (注: 只统计在职的销售员, 离职的不统计). 将结果以表格形式发送到我的邮箱'

promptMsgs=[
  {"role": "system", "content": sysMsg},
  {"role": "user", "content": questionMsg}
]
response = client.chat.completions.create(
  model=modelAccessInfo.deployment_name,
  temperature=0,
  messages=promptMsgs,
  tools=tools,
  tool_choice='auto'
)
log.debug(f'chat.completions.create: {response}')

rspMsg=response.choices[0].message
while rspMsg.tool_calls:
  promptMsgs.append(rspMsg)
  for tool_call in rspMsg.tool_calls:
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    function_to_call = functions[function_name]['python_function']
    
    log.info(f'调用工具: {function_name}({function_args})...')
    
    function_response = function_to_call(**function_args)
    log.info(f'工具调用结果: {function_response}')
    
    promptMsgs.append({
      "tool_call_id": tool_call.id,
      "role": "tool",
      "name": function_name,
      "content": str(function_response),
    })
    
  response = client.chat.completions.create(
    model=modelAccessInfo.deployment_name,
    messages=promptMsgs,
    temperature=0,
    tools=tools,
    tool_choice='auto'
  )
  log.debug(f'chat.completions.create: {response}')
  rspMsg=response.choices[0].message
  
log.info(f'''===输出==================
{response.choices[0].message.content}
----------------------------''')