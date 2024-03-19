import os
from typing import List
import coloredlogs, logging

coloredlogs.DEFAULT_LEVEL_STYLES = {'critical': {'bold': True, 'color': 'red'}, 'debug': {'color': 'white', 'faint':True}, 'error': {'color': 'red'}, 'info': {'bold': True,'color': 'green'}, 'notice': {'color': 'magenta'}, 'spam': {'color': 'green', 'faint': True}, 'success': {'bold': True, 'color': 'green'}, 'verbose': {'color': 'blue'}, 'warning': {'color': 'yellow'}}
coloredlogs.install(level='DEBUG', fmt='%(asctime)s %(name)s %(levelname)s %(message)s')

#配置代理
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1087"

#AOAI实例访问信息配置
AOAI_INSTS={
    "??AOAI实例名??":{
        "endpoint":"https://??AOAI实例名??.openai.azure.com/",
        "api_key":"??KEY??",
        "region":"??地区??",
        "depolyments":{
            "??模型部署名??":{
                "model":"gpt-4",
                "version":"1106-Preview"
            },
            "??模型部署名??":{
                "model":"gpt-35-turbo",
                "version":"1106"
            },
        }
    },
}

#模型访问信息
class ModelAccessInfo:
    def __init__(self, endpoint:str, deployment_name:str, api_key:str, region:str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.region = region

    def __str__(self):
        return f"ModelAccessInfo({self.model_name}, {self.model_version}, {self.endpoint}, {self.api_key}, {self.region})"
 
    #从AOAI_INSTS配置中查找指定的模型访问信息
    @staticmethod
    def find_model_access_info(model_name, model_version:str=None, region:str=None)->List['ModelAccessInfo']:
        models=[]
        for inst_name, inst_info in AOAI_INSTS.items():
            for dep_name, dep_info in inst_info["depolyments"].items():
                if dep_info["model"] == model_name:
                    if model_version is None or model_version == dep_info["version"]:
                        if region is None or region == inst_info["region"]:
                            models.append(ModelAccessInfo(inst_info["endpoint"], dep_name, inst_info["api_key"], inst_info["region"]))
        return models