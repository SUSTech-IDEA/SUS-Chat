# 🐷SUS-Chat: Instruction tuning done right

<p align="left">
<a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish&nbsp
</p>

<br><br>

<div align="center">

<p align="center">
<img src="https://github.com/SUSTech-IDEA/SUS-Chat/raw/main/assets/sustech.svg?sanitize=true" width="200px">
<img src="https://github.com/SUSTech-IDEA/SUS-Chat/raw/main/assets/ccnl.png?sanitize=true" width="200px">
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
<img src="https://img.shields.io/badge/ModelScope-sustc-blue" style="margin: 0 0;">
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

# News

- 2023-12-05: SUS-Chat is ranked 2nd in [Open LLM
  leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
  and surpassed all models under 70B.

- 2023-12-01: SUS-Chat-34B is now avaliable on HuggingFace🤗.

# Introduction

<img src="https://hackmd.io/_uploads/HJlDtzhBa.png" id="fig-sus"
alt="Figure 1: DALL·E 2023-12-01 11.03.28 - An imposing, majestic wild boar combined with elements of a futuristic transformer robot. The boar itself should be intricately blended with these tra" />

**SUS-Chat** is a 34B bilingual Chinese-English dialogue model, jointly
released by the **Southern University of Science and Technology** and
**Cognitive Computing and Natural Language Center of International
Digital Economy Academy (IDEA-CCNL)**. This model is based on
`01-ai/Yi-34B` and has been fine-tuned on millions of high-quality,
multilingual instruction data. While maintaining the strong language
capabilities of the base model, the SUS-Chat-34B model has improved the
model’s response to human instructions through high-quality instruction
fine-tuning and excels at imitating human thought processes through
chains of thought. It introduces inter-instruction attention sharing in
long texts, expanding the window size from 4K to 8K, significantly
enhancing the usability of multi-round dialogues.

It has surpassed all models of the same size in almost all benchmark
tests and is better suited to meet the practical needs of complex
multilingual tasks. Compared to larger models, SUS-Chat-34B remains
highly competitive and achieved state-of-the-art performance in our
comprehensive evaluations.

SUS-Chat powerfully demonstrates that through the right instruction
fine-tuning, academic institutions can achieve better performance
without increasing model parameters, using open-source datasets and
models. This bridges the gap between academia and industry in large
language models and opens new possibilities for collaboration between
academic and industrial sectors.

# Performance

To better evaluate the performance of the SUS-Chat-34B model, we
conducted assessments across multiple benchmark tests and have
open-sourced the evaluation framework
[TLEM](https://huggingface.co/spaces/SUSTech/tlem) to facilitate
replication and comparison by other researchers.

In TLEM, we utilized various benchmark tests including MMLU, CMMLU,
C-Eval, BBH, GSM-8K, and MATH, focusing on measuring the model’s
knowledge and thinking capabilities. In these metrics, the SUS-Chat-34B
model achieved state-of-the-art performance. Additionally, we
incorporated
[lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) to test
SUS-Chat and similar models on winogrande, hellaswag, arc, and
truthful-qa, assessing the model’s common-sense reasoning ability and
susceptibility to illusions.

Overall, the SUS-Chat-34B model significantly outperformed models of
similar scale and achieved the most advanced comprehensive performance.

| model             | mmlu-chat | cmmlu-chat | ceval-chat | gsm8k |   BBH |  MATH | winogrande |   arc | hellaswag | truthfulqa | average |
|:------------------|----------:|-----------:|-----------:|------:|------:|------:|-----------:|------:|----------:|-----------:|--------:|
| GPT-4             |        83 |         71 |       69.9 |  91.4 |  86.7 |  45.8 |       87.5 |  94.5 |      91.4 |        nan | 80.1333 |
| SUS-Chat-34B      |     77.35 |      78.68 |      82.42 | 80.06 | 67.62 |  28.8 |      81.22 | 81.54 |     83.79 |      57.47 |  71.895 |
| Qwen-72B-Chat     |     74.52 |      77.02 |      77.22 | 76.57 | 72.63 |  35.9 |      80.58 | 81.29 |     87.02 |      50.64 |  71.339 |
| DeepSeek-67B-Chat |     69.43 |      48.51 |       59.7 | 74.45 | 69.73 | 29.56 |      76.09 |  82.1 |     86.06 |      56.37 |    65.2 |
| OrionStar-34B     |     68.51 |      66.88 |      65.13 | 54.36 | 62.88 |  12.8 |      77.27 | 80.19 |     84.54 |      53.24 |   62.58 |
| Yi-34B-Chat       |     66.96 |      55.16 |      77.16 | 63.76 | 61.54 | 10.02 |      76.64 | 70.66 |     82.29 |      54.57 |  61.876 |

<img
src="https://github.com/SUSTech-IDEA/SUS-Chat/raw/main/assets/radar.png"
id="fig-bench" alt="Figure 2: Benchmark" />

# Usage

SUS-Chat-34B is a standard LLaMA model and should be seamlessly
compatible with the LLaMA ecosystem. We provide the following example to
demonstrate how it can be used for multi-turn dialogues.

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer


def chat_template(messages):
    history = ""
    for message in messages:
        match message:
            case {"role": "user", "content": message}:
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

input_ids = tokenizer.encode(
    chat_template(messages), return_tensors="pt", add_special_tokens=False
).to("cuda")
output_ids = model.generate(input_ids.to("cuda"), max_length=256)
response = tokenizer.decode(
    output_ids[0][input_ids.shape[1] :], skip_special_tokens=False
)

messages.append({"role": "assistant", "content": response})

# Second round

messages.append({"role": "user", "content": "What is the capital of China?"})

input_ids = tokenizer.encode(
    chat_template(messages), return_tensors="pt", add_special_tokens=False
).to("cuda")
output_ids = model.generate(input_ids.to("cuda"), max_length=256)
response = tokenizer.decode(
    output_ids[0][input_ids.shape[1] :], skip_special_tokens=False
)

messages.append({"role": "assistant", "content": response})
```

# Limitations

SUS-Chat has only undergone supervised fine-tuning and has not yet been
trained on human preference learning. As a result, it may produce
unreasonable responses in some situations and exacerbate existing issues
in language models, including hallucinations, non-determinism, and
cumulative errors. To achieve better performance for downstream tasks,
we recommend adjusting the generation configuration parameters
accordingly.

# Disclaimer

During the training process, we used data compliance check algorithms to
ensure the compliance of the training model as much as possible. Due to
the complexity of the data and the diverse use cases of language models,
we cannot guarantee that the model will produce correct and reasonable
outputs in all scenarios. Please be aware that there is still a risk of
the model generating problematic outputs. We will not be responsible for
any risks or issues arising from misuse, misguidance, illegal use, and
related misinformation, as well as data security issues related to the
model.

# License

This model is developed entirely for academic research and free
commercial use, but it must adhere to the
[license](https://github.com/SUSTech-IDEA/SUS-Chat/blob/main/MODEL_LICENSE_AGREEMENT.txt)
from 01-ai.
