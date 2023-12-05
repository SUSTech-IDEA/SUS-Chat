# 🐷SUS-Chat: Instruction tuning done right

<p align="left">
中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp
</p>

<br><br>

<div align="center">

<p align="center">
<img width="200px" src="https://github.com/SUSTech-IDEA/SUS-Chat/raw/main/assets/sustech.svg?sanitize=true">
</p>

<div style="display: inline-block;">

<a rel="noopener nofollow" href="https://github.com/SUSTech-IDEA/SUS-Chat/issues">
<img src="https://img.shields.io/github/issues/SUSTech-IDEA/SUS-Chat?logo=github" style="margin: 0 0;">
</a>

</div>

<div style="display: inline-block;">

<a href="https://huggingface.co/SUSTech">
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SUSTech-blue" style="margin: 0 0;">
</a>

</div>

<div style="display: inline-block;">

<a rel="noopener nofollow" href="https://www.modelscope.cn/organization/sustc/">
<img src="https://img.shields.io/badge/ModelScope-sustec-blue" style="margin: 0 0;">
</a>

</div>

<div style="display: inline-block;">

<a rel="noopener nofollow" href="https://github.com/SUSTech-IDEA/SUS-Chat/blob/main/LICENSE">
<img src="https://img.shields.io/badge/Code_License-Apache_2.0-lightblue" style="margin: 0 0;">
</a>

</div>

<div style="display: inline-block;">

<a rel="noopener nofollow" href="https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt">
<img src="https://img.shields.io/badge/Model_License-Model_Agreement-lightblue" style="margin: 0 0;">
</a>

</div>

<div style="display: inline-block;">

<a rel="noopener nofollow" href="mailto:oss@data.sustech.edu.cn">
<img src="https://img.shields.io/badge/✉️-data@sustech.edu.cn-FFE01B" style="margin: 0 0;">
</a>

</div>

</div>

# 新闻

- 2023-12-05: SUS-Chat在[Open LLM
  leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)排名第二，并在所有小于70B的模型中排名第一。

- 2023-12-01: SUS-Chat-34B现已在HuggingFace🤗上可用。

# 介绍

![](https://hackmd.io/_uploads/HJlDtzhBa.png)

**SUS-Chat**
是一个34B的中英文对话模型，由南方科技大学和粤港澳大湾区数字经济研究院联合发布。SUS-Chat-34B模型在数百万高质、多语言的指令数据上进行了微调，在保持基础模型强大的语言能力的同时，SUS-Chat-34B模型通过高质量指令微调改善了模型对人类指令的响应方式并擅长通过思维链的方式模仿人类思考过程,并在长文本间引入指令间注意力共享，将窗口长度从4K扩展到8K,
显著提升了多轮对话的可用性。

它在几乎所有基准测试中超过了所有同尺寸的模型，而且能够更好地满足了复杂多语言任务的实际需求，相比于更大的模型，SUS-Chat-34B仍具有相当竞争力，在我们的综合评测中取得了最先进的表现。

SUS-Chat有力地证明了通过正确的指令微调，学术机构可以在不增加模型参数的情况下，通过开源的数据集和模型，获得更好的性能,
这弥合了学术界和工业界的在大语言模型上的差距，为学术界和工业界的合作提供了新的可能性。

# 性能

为了更好地评估SUS-Chat-34B模型的性能，我们在多个基准测试中进行了评估，并开源了评估框架[TLEM](https://huggingface.co/spaces/SUSTech/tlem)，以便于其他研究人员进行复现和比较。

在TLEM中，我们使用了多个基准测试，包括：MMLU, CMMLU, C-Eval, BBH,
GSM-8K, MATH,
专注于衡量模型的知识和思维能力，在这些指标中SUS-Chat-34B模型取得了最先进的表现，我们还额外引入了[lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)测试了SUS-Chat和同类模型在winogrande,
hellaswag, arc, truthful-qa的表现, 衡量模型的常识性推理能力和幻觉。

综合上看，SUS-Chat-34B模型显著领先于同规模的模型，并取得了最先进的综合性能。

| model             | mmlu-chat | cmmlu-chat | ceval-chat | gsm8k |   BBH |  MATH | winogrande |   arc | hellaswag | truthfulqa | average |
|:------------------|----------:|-----------:|-----------:|------:|------:|------:|-----------:|------:|----------:|-----------:|--------:|
| GPT-4             |        83 |         71 |       69.9 |  91.4 |  86.7 |  45.8 |       87.5 |  94.5 |      91.4 |        nan | 80.1333 |
| SUS-Chat-34B      |     77.35 |      78.68 |      82.42 | 80.06 | 67.62 |  28.8 |      81.22 | 81.54 |     83.79 |      57.47 |  71.895 |
| Qwen-72B-Chat     |     74.52 |      77.02 |      77.22 | 76.57 | 72.63 |  35.9 |      80.58 | 81.29 |     87.02 |      50.64 |  71.339 |
| DeepSeek-67B-Chat |     69.43 |      48.51 |       59.7 | 74.45 | 69.73 | 29.56 |      76.09 |  82.1 |     86.06 |      56.37 |    65.2 |
| OrionStar-34B     |     68.51 |      66.88 |      65.13 | 54.36 | 62.88 |  12.8 |      77.27 | 80.19 |     84.54 |      53.24 |   62.58 |
| Yi-34B-Chat       |     66.96 |      55.16 |      77.16 | 63.76 | 61.54 | 10.02 |      76.64 | 70.66 |     82.29 |      54.57 |  61.876 |

![](assets/radar.png)

# 用法

SUS-Chat-34B是标准的LLaMA模型，应该可以无缝地与LLaMA生态系统兼任，我们提供下面的例子来展示如何使用它进行多轮对话

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer


def chat_template(messages):
    history = ""
    for message in messages:
        match message:
            case {"role": "human", "content": message}:
                history += f"### Human: {message}\n\n### Assistant: "
            case {"role": "assistant", "content": message}:
                history += message
    return history


model_path = "SUSTech/SUS-Chat-34B"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype="auto"
).eval()

messages = [{"role": "user", "content": "hi"}]

input_ids = tokenizer.encode(chat_template(messages), return_tensors="pt").to("cuda")
output_ids = model.generate(input_ids.to("cuda"))
response = tokenizer.decode(
    output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
)

messages.append({"role": "assistant", "content": response})

# Second round

messages.append({"role": "user", "content": "What is the capital of China?"})

input_ids = tokenizer.encode(chat_template(messages), return_tensors="pt").to("cuda")
output_ids = model.generate(input_ids.to("cuda"))
response = tokenizer.decode(
    output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
)

messages.append({"role": "assistant", "content": response})
```

# 限制

SUS-Chat只进行了监督微调，尚未进行人类偏好学习，因此在一些情况下可能会产生不合理的回复，并放大某些语言模型现有的问题,
包括幻觉、非确定性和累积误差,
为了实现更有利于下游任务的性能，我们建议相应地调整生成是配置参数。

# 免责声明

我们在训练过程中使用数据合规检查算法，尽力确保训练模型的合规性。由于数据复杂且语言模型使用场景多样，我们无法保证模型在所有情况下生成正确和合理的输出。请注意，模型仍然存在产生问题输出的风险。对于因滥用、误导、非法使用和相关错误信息以及相关数据安全问题而导致的任何风险和问题，我们将不承担责任。

# 许可

该模型完全开发供学术研究和免费商业使用，但需要遵守来自零一万物的[许可](https://github.com/SUSTech-IDEA/SUS-Chat/blob/main/MODEL_LICENSE_AGREEMENT.txt)