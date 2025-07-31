import os
import utils
import time
import random
from glob import glob
import pandas as pd
from tqdm import tqdm
import json

## 批量生成数据
def generation_pretrain_text_data(
        output_dir="./data",  # 最终的结果保存地址
        model_name="/home/lixinyu/weights/Qwen2.5-3B", # 本地模型地址
        data_dir="/home/lixinyu/data/cosmopedia-50k/data", # 保存到本地的数据集
        request_batch_size=2,  # batch
        temperature=1.0,
        top_p=1.0,
):
    ### 确保输出目录存在，并初始化输出文件路径（关键修复：提前定义output_file）
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pretrain.jsonl")  # 固定输出文件路径
    pretrain_data = []  # 用于存储所有数据的列表

    ### 加载已存在的数据（如果有）
    if os.path.exists(output_file):
        try:
            pretrain_data = utils.jload(output_file)
            # 确保加载的数据是列表格式
            if not isinstance(pretrain_data, list):
                pretrain_data = []
            print(f"Loaded {len(pretrain_data)} existing pretrain data")
        except Exception as e:
            print(f"Warning: Failed to load existing data, starting fresh. Error: {e}")
            pretrain_data = []

    ### 读取parquet文件并提取prompt
    parquet_files = glob(os.path.join(data_dir, "*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    all_prompts = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        if "prompt" in df.columns:
            all_prompts.extend(df["prompt"].dropna().tolist())  # 过滤空值
        else:
            print(f"Warning: 'prompt' column not found in {file}")
    
    if not all_prompts:
        raise ValueError("No valid prompts found in dataset files")

    ### 选择20条新的prompt（避免重复）
    existing_prompts = {item["prompt"] for item in pretrain_data} if pretrain_data else set()
    new_prompts = [p for p in all_prompts if p not in existing_prompts]
    num_samples = min(20, len(new_prompts))
    
    if num_samples == 0:
        print("No new prompts to generate. Exiting.")
        return
    
    selected_prompts = random.sample(new_prompts, num_samples)
    print(f"Selected {num_samples} new prompts for generation")

    ### 调用模型生成数据
    decoding_args = utils.OpenAIDecodingArguments(
        temperature=temperature,
        max_tokens=3072,
        top_p=top_p,
        stop=["\n20", "20.", "20."],
    )

    request_start = time.time()
    batch_size = min(request_batch_size, num_samples)
    results = []
    
    for i in tqdm(range(0, num_samples, batch_size), desc="Generating text"):
        batch = selected_prompts[i:i+batch_size]
        batch_results = utils.openai_completion(
            prompts=batch,
            model_name=model_name,
            batch_size=batch_size,
            decoding_args=decoding_args,
        )
        results.extend(batch_results)
    
    print(f"Generation completed in {time.time() - request_start:.2f} seconds")

    ### 保存生成的数据（确保一一对应）
    ### 保存为 JSONL 格式（核心修改部分）
    new_entries_count = 0
    with open(output_file, 'a', encoding='utf-8') as f:  
        for prompt, result in tqdm(zip(selected_prompts, results), 
                                    total=num_samples, 
                                    desc="保存结果到 JSONL"):
            entry = {
                "prompt": prompt,
                "generated_text": result.text,
                "timestamp": time.time()
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')  
            new_entries_count += 1
    print(f"成功保存 {new_entries_count} 条新数据到 {output_file}（JSONL 格式）")


if __name__ == "__main__":
    generation_pretrain_text_data()