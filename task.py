import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)
from conversation import Conversation,get_conv_template
#local_env = os.environ.copy()
#local_env["PATH"]="/root/miniconda3/envs/myenv/bin:" + local_env["PATH"]
#os.environ.update(local_env)
import glob
import logging
from copy import copy
from typing import List, Optional, Union
from pydantic import BaseModel, conlist, constr
import json
from ws4py.client.threadedclient import WebSocketClient
import GPUtil
from threading import Thread
import time
import traceback

avg_time = 0;
all_time = 0;
stop_task = False
model_directory = "/workspace/Emerhyst-20B-4bpw-h8-exl2"
draft_directory = "/workspace/TinyLlama-4.0bpw"

def load_model():
    global generator, default_settings,tokenizer
    if not generator:
        model_config = ExLlamaV2Config()
        model_config.model_dir = model_directory
        model_config.prepare()
        model = ExLlamaV2(model_config)
        #model_cache = ExLlamaV2Cache(model, lazy = True)
        model_cache = ExLlamaV2Cache_8bit(model, lazy = not model.loaded)
        model.load_autosplit(model_cache)
        tokenizer = ExLlamaV2Tokenizer(model_config)
        #generator = ExLlamaV2StreamingGenerator(model, model_cache, tokenizer, draft, draft_cache, 5)
        generator = ExLlamaV2StreamingGenerator(model, model_cache, tokenizer)
        generator.warmup()
    return generator

class MyWebSocketClient(WebSocketClient):
 
    def opened(self):
        self.send_uid()
        #self.get_task()
    def closed(self, code, reason=None):
        print("WebSocket closed. Code: {}, Reason: {}".format(code, reason))
        #self.reconnect()
        
    def received_message(self, resp):
        global complete_num,stop_task,avg_time,all_time
        #print(resp)
        resp_dict = json.loads(str(resp))
        if resp_dict.get('type') == 'ping':
            data = {
                "event": "pong"
            }
            self.send(json.dumps(data))
        elif resp_dict.get('type') == 'task':
            #print(resp)
            try:
                print("已接收到新任务")
                start_time = time.time()
                params = Item.model_validate(resp_dict)
                msg_dict = json.loads(str(params.messages))
                params.messages = msg_dict
                text,tokens = generate(params)
                #tokens = 249
                #tokens = cached_tokenize(text).shape[-1]
                complete_num = complete_num+1
                end_time = time.time()
                elapsed_time = end_time - start_time
                data = {
                    "event": "complete",
                    "text": text,
                    "tokens": tokens,
                    "is_translate": params.is_translate,
                    "history_id": params.history_id,
                    "chat_id": params.chat_id,
                    "user_id": params.user_id,
                    "robot_id": params.robot_id,
                    "lang_code": params.lang_code,
                    "complete_num": complete_num,
                    "run_time_ms": elapsed_time*1000,
                    "current_id": params.current_id,
                    "model_id": 1,
                }
                self.send(json.dumps(data))
                elapsed_time = round(elapsed_time*1000, 2)
                each_token = round(elapsed_time/tokens, 2)
                all_time = all_time + elapsed_time
                avg_time = round(all_time/complete_num, 2)
                
                print(f"任务完成 tokens：{tokens} 执行时间：{elapsed_time}ms {each_token} ms/token avg: {avg_time}ms")
                
                #print(f"{params.prompt}{text}")
            except Exception as e:
                data = {
                "event": "republish",
                "current_id": resp_dict.get('current_id')
                }
                print("任务生成失败")
                print("An error occurred:", str(e))
                print(resp_dict)
                self.send(json.dumps(data))
            self.get_task()

        elif resp_dict.get('type') == 'no_task':
            #print('没有新任务')
            time.sleep(0.4)
            self.get_task()
        elif resp_dict.get('type') == 'stop':
            print('结束任务')
            stop_task = True
        elif resp_dict.get('type') == 'init':
            global model_directory,alpha_value,max_seq_len
            print("init model. . .")
            #model_directory=resp_dict.get('model_directory')
            alpha_value=int(resp_dict.get('alpha_value'))
            max_seq_len=int(resp_dict.get('max_seq_len'))
            load_model()
            print("start request task. . .")
            self.get_task()
        else:
            print('不能解析的参数')
        
    def send_uid(self):
        # 定义一个 Python 对象
        data = {
            "event": "subscribe",
            "uid": uuid,
            "gpus": gpu_array,
            "cpu_count": cpu_count,
            "memory_total": 0,
            "memory_used": 0,
            "complete_num": complete_num
        }
        self.send(json.dumps(data))
    #获取一个任务        
    def get_task(self):
        if not stop_task:
            data = {
                "event": "get",
                "model": 1
            }
            self.send(json.dumps(data))
        else:
            sys.exit()
            os.system("sudo shutdown now")
            
def start_websocket_client():
    global timeout,ws
    try:
        ws = MyWebSocketClient('ws://43.135.148.3:8282')
        #ws.daemon = True
        ws.connect()
        ws.run_forever()
        timeout = 0
        ws.send_uid()
    except KeyboardInterrupt:
        ws.close()
    except:
        timeout = timeout + 1
        print("Timing out for %i seconds. . ." % timeout)
        time.sleep(timeout)
        print("Attempting reconnect. . .")
        if not stop_task:
            start_websocket_client()
        else:
            sleep(30)

        
class Item(BaseModel):
    text: str
    messages: str
    system: str
    bot_name: str = "bot_name"
    user_name: str = "user_name"
    role1: str
    role2: str
    chat_id: int
    user_id: int
    robot_id: int = 0
    lang_code: str = "en"
    current_id: str
    is_translate: int = 0
    history_id: int = 0
    repetition_penalty: float = 1.15
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 100
    typical: float = 1.0
    min_p: float = 0.00
    repetition_penalty_sustain: int = 256
    beams: int = 1
    beam_length: int = 1
    max_new_tokens: int = 128


def generate(params: Item):
    global tokenizer
    if not params:
        raise ValueError("No params provided")
    try:
        conv = Conversation(name="alpaca", system_message=params.system, messages=params.messages, roles = (params.role2,params.role1))
        #conv = get_conv_template("alpaca")
        conv.append_message(params.role2, params.text)
        conv.append_message(params.role1, None)
        #print(conv.get_prompt())
        prompt = conv.get_prompt(tokenizer,3000)
    except Exception as e:
        traceback.print_exc()
    try:
        # Prompt
        gen_settings = ExLlamaV2Sampler.Settings()
        gen_settings.temperature = params.temperature
        gen_settings.top_k = params.top_k
        gen_settings.top_p = params.top_p
        gen_settings.token_repetition_penalty = params.repetition_penalty
        gen_settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
        
        input_ids = tokenizer.encode(prompt)
        prompt_tokens = input_ids.shape[-1]
        generator.set_stop_conditions(["<|", "\n#", "###", "####", "\n\n\n", "\n[", "\n(", f"\n{params.bot_name}",f"\n{params.user_name}"])
        generator.begin_stream(input_ids, gen_settings)
        generated_tokens = 0
        text = '';
        while True:
            chunk, eos, _ = generator.stream()
            generated_tokens += 1
            text = text + chunk
            #print (chunk, end = "")
            #sys.stdout.flush()
            if eos or generated_tokens == params.max_new_tokens: break
    except Exception as e:
        traceback.print_exc()
        
    
    #print(f"Statistics: {generator.stats()} Chat Id: {params.chat_id}")
    return text,generated_tokens


if __name__ == '__main__':
    ws = None
    # 获取所有可用的 GPU 设备
    gpus = GPUtil.getGPUs()
    gpu_array = []
    for gpu in gpus:
        gpu_info = {"name": gpu.name, "memoryTotal": gpu.memoryTotal}
        gpu_array.append(gpu_info)
    cpu_count = 24
    uuid = os.getenv("AutoDLContainerUUID", "")
    memory_info = {}
    complete_num = 0
    generator = None
    default_settings = None
    alpha_value = 1
    max_seq_len = 4096
    timeout = 0
    
    try:
        t = Thread(target=start_websocket_client)
        t.daemon = True
        t.start()
        t.join()
    # 断开客户端连接
    except KeyboardInterrupt:
        ws.close()
    except:
        timeout = timeout + 1
        print("Timing out for %i seconds. . ." % timeout)
        time.sleep(timeout)
        print("Attempting reconnect. . .")
        ws.connect()