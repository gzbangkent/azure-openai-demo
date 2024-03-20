#############################
# 场景: 客服沟通内容审计打标
# 技术要点: JSON输出(1106), PROMPT工程
# 流程: 
# 1. 设置JSON输出格式
# 2. 设定系统消息, 限定JSON内容为标签名和标签值
# 3. 将沟通内容放入对话中提问
# 4. AI按要求进行打标, 输出标签列表
#############################
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
  api_version="2024-02-01"
)

labels=['客户有负面情绪','售前问题', '价格或优惠问题', '产品参数咨询', '产品使用咨询','产品质量问题', '投诉', '建议']

system_msg=f'''你是一位标签打标员, 从日常客服与客户的沟通内容中, 进行标签打标.
打标签时, 请遵守以下规则:
- 允许的标签名包括:{labels}. 不允许自行添加标签
- 标签值只允许为: 是, 否
- 你的输出为JSON格式, 标签名为key, 标签值为value
- 所有的标签名都必须出现在输出中, 不能遗漏'''

# sales_converstion = '''客户: 你好, 我想咨询一下你们的产品, 有什么优惠吗?
# 客服: 你好, 我们的产品现在正在进行促销, 买一送一, 你可以看一下我们的官网, 或者我帮你发一份产品手册给你, 你可以看一下.
# 客户: 好的, 那我看一下手册, 有问题我再问你.
# 客服: 好的, 你看一下, 有问题随时问我.
# 客户: 建议你们的产品手册能不能加上产品的参数, 这样更方便我们了解产品.'''
sales_converstion = '''客户: 你好, 我想投诉一下你们的VM产品, 总是出现问题, 你们的产品太差了. 赶紧给我解决.
客服: 对不起, 请问您是遇到哪些具体问题? 你可以告诉我一下你的问题, 我帮你解决.
客户: 我的VM总是断线, 有时候还会出现蓝屏, 你们的产品太差了.
客服: 您看这样, 我帮您提交一下问题, 我们的技术人员会尽快联系您, 请您耐心等待一下, 谢谢您的反馈.
客户: 好的, 你们尽快解决问题, 我等不了太久了.'''


log.info(f'''===AI设定===================
{system_msg}
----------------------------''')
log.info(f'''===客服客户对话==============
{sales_converstion}
----------------------------''')

response = client.chat.completions.create(
  model=modelAccessInfo.deployment_name,
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": system_msg},
    {"role": "user", "content": f'请对以下沟通内容,按要求进行打标. \n{sales_converstion}'}
  ]
)

log.info(f'''===打标输出==================
{response.choices[0].message.content}
----------------------------''')