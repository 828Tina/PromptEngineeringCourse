# 提示词工程教程

提示词工程（Prompt Engingering），也被称为上下文提示（In-Context Prompting），指的是通过结构化文本的方法来完善提示词（Prompt），从而引导大模型生成更符合预期的输出结果的技术。简单点说就是“用更聪明的提问方式，让AI更好地理解并完成任务”。通过提示词工程可以在不通过训练更新模型权重的情况下，让大模型完成不同类型的任务。提示词工程的效果在不同的模型中可能会有很大的差异，因此需要大量的实验和探索。掌握了提示工程相关技能将有助于用户更好地了解大型语言模型的能力和局限性。

## 💡前言

这种无需微调就能解决问题的特性，让提示词在实际应用中变得越来越重要。要知道，不是所有场景都有足够的标注数据来做微调，也不是所有团队都有能力承担微调的成本。这时候，一句精准的提示词就能让大模型在零样本或少样本的情况下完成任务，大大降低了使用门槛。但我们也要清楚，提示词的效果和模型本身的能力紧密相关。只有当模型参数量达到一定规模，涌现出足够强的理解和推理能力时，提示词才能发挥最大作用。要是模型本身 “底子薄”，哪怕提示词设计得再精巧，也很难得到理想的结果。

## 🚀环境安装与平台准备

### ⚙️环境安装

1. python>=3.9
2. pytorch：在官网安装最新版本即可，需要注意的是2.5版本bug较多，不建议使用
3. 其他的python库：transformers、modelscope、vllm等

```python
# 创建环境
conda create -n test python=3.10

# pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他
pip install -U transformers modelscope
```

### 🔍平台选择

模型可以保存到本地运行，也可以使用云平台，调用API使用大模型，下面我们简单举了几个例子，方便大家快速上手大模型的使用。

> HuggingFace和vLLM的方法都需要本地保存模型，API的方式不用。

**HuggingFace**

如果将模型保存在***本地***，可以使用Huggingface提供的推理代码，在后续的举例中，如果没有特殊说明，我们都会使用Qwen系列模型作为基线模型，关于推理代码，我们使用Qwen官网提供的[推理代码](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)

<details>
  <summary>完整推理代码</summary>
  
```python
### 加载模型
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = 'Qwen/Qwen2.5-3B'  # 替换为你下载的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,device_map='auto', torch_dtype='auto')

### 提示词
prompt = "Hello, Who are you?"

### 推理代码
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

</details>


**vLLM**

`vLLM`是伯克利大学LMSYS组织开源的大语言模型高速推理框架，旨在极大地提升实时场景下的语言模型服务的吞吐与内存使用效率。vLLM是一个快速且易于使用的库，用于 LLM 推理和服务，可以和HuggingFace 无缝集成。vLLM利用了全新的注意力算法「PagedAttention」，有效地管理注意力键和值。

`vllm在吞吐量方面，vLLM的性能比HuggingFace Transformers(HF)高出 24 倍，文本生成推理（TGI）高出3.5倍。`

简单点说就是vllm框架的推理速度很快，但是显存占用较高，同样的3B模型，本地推理可能只需要15GB左右，而vllm框架则需要37GB，因此如果硬件资源不足，vllm并不是一个很好的选择。

<details>
  <summary>完整流程</summary>

---

1. 在环境里安装vllm库

   
```python
pip install -U vllm
```

2. 开启一个终端页面，运行下面的代码

```python
vllm serve /your/path/of/model
```

3. 开启新的终端页面运行各个代码

在新的终端页面，我们就可以跑我们对应的服务了，需要注意的是，我们使用大模型来进行推理的时候可以使用[openai的prompt的API的接口](https://openai.apifox.cn/api-55352401)，输入接口参考给出的文档即可，讲起来比较抽象，我们看下代码例子：

*模型生成回答的代码 ：*

```python
results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            # logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )

```



*对应的工具utils的 openai_completion函数：*

```python
def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
):
……
……
    completion_batch = client.completions.create(
                prompt=prompt_batch, **shared_kwargs
            )
    choices = completion_batch.choices
    
……
……

```


`client.completions.create`其实使用的就是openai库中的推理问答的函数。

经过这些步骤后，我们在运行推理服务的时候就能充分利用GPU资源，高效完成各项推理任务。


</details>

**API**

大模型的 API 调用，简单来说，是指开发者或用户通过应用程序编程接口（API） 与大模型进行交互，从而利用大模型的能力完成特定任务的进程，这些大模型无需部署到本地，你使用的资源其实是这些厂商提供的服务集群，不仅推理速度快，而且并不占用显存，不过每个模型会有费用的消耗，不一定能免费使用。

本次教程我们参考了两个云平台，链接如下：

1. [硅基流动](https://cloud.siliconflow.cn/sft-d1n0sv33jrms738gmgpg/models)
2. [阿里云百炼](https://bailian.console.aliyun.com/?spm=5176.12818093_47.resourceCenter.1.223c2cc96V9eQn&tab=api#/api/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2712576.html)



## 📚课程目录


| 教程章节   | 核心内容 |  
|:--------|:------|
| [1.前置准备](./1.preparatory_work_in_advance)   | 模型安装、平台选择、模型选择(base,instruct,reasoning)   |
| [2.提示词撰写技巧](./2.tips_for_prompt)   | 提示词结构、提示词要素、思维链等   |
| [3.常见任务示例](./3.common_task_examples)   | 文本摘要、数学计算、多轮对话等   | 
| [4.多模态提示词](./4.multimodal_prompt)   | 多模态提示词使用示例   | 
| [5.合成数据](./5.synthetic_data)   | 预训练、微调、推理数据集合成代码实践   | 
| [6.RAG](./6.RAG)   | RAG实践   | 
| [7.Agent](./7.Agent)   | 函数调用、MCP实践、Agents原理   | 
| [8.提示词与上下文工程](xxx)   | 提示词工程与上下文工程区别  | 


---

### 🌟核心贡献

- [李馨雨](https://github.com/828Tina) （情感机器（北京）科技有限公司技术员-西安电子科技大学）







