import os
from openai import OpenAI
import json
import random
from tqdm import tqdm

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

## 生成数据
def alpaca_data(num=10):
    from modelscope.msdatasets import MsDataset
    ds =  MsDataset.load('AI-ModelScope/alpaca-gpt4-data-zh', subset_name='default', split='train')
    # 保存到jsonl文件里，随机抽取10条数据
    data=[]
    for ds_data in ds:
        data.append(ds_data)
    # 假设 data 已经通过前面代码加载
    sample = random.sample(data, num)   # 随机选 10 条
    return sample


## 读取jsonl文件并处理提示词
def jsonl(jsonl_path):
    if os.path.exists(jsonl_path):
        data = [json.loads(line) for line in open(jsonl_path, 'r', encoding='utf-8')]
    else:
        # 需要生成
        data = alpaca_data(num=10)
    return data


## 生成回答
def generate_reasoning_content():
    # 默认保存在data里
    data_path="./data"
    os.makedirs(data_path, exist_ok=True)
    all_data = jsonl(os.path.join(data_path, "sample.jsonl"))

    output_file = os.path.join(data_path, "inference.jsonl")
    with open(output_file, 'a', encoding='utf-8') as f:  
        for data in tqdm(all_data,total=len(all_data),desc="生成推理数据"):
            prompt = data["instruction"]+"\n"+data["input"]
            completion = client.chat.completions.create(
                # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                model="qwen3-235b-a22b-thinking-2507",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
                # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
                # extra_body={"enable_thinking": False},
                max_tokens=2048,
                extra_body={"enable_thinking": True,"thinking_budget":1024}
            )
            result = {
                "input":prompt,
                "reasoning_content":completion.choices[0].message.reasoning_content,
                "content":completion.choices[0].message.content
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')  
    print(f"成功保存到 {output_file}（JSONL 格式）")


if __name__ == "__main__":
    generate_reasoning_content()

