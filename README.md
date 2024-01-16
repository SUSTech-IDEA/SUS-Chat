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
<img src="https://img.shields.io/badge/🤖ModelScope-sustc-blue" style="margin: 0 0;">
</a>

</div>

<a href="https://wisemodel.cn/organization/SUSTech">
<img src="https://img.shields.io/badge/WiseModel-SUSTech-blue"> </a>

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

- 2024-1-04: 🔥 `cloudyu` created a series of top ranked
  [MOE](https://huggingface.co/cloudyu/Yi-34Bx2-MoE-60B) based on our
  model

- 2023-12-09: 🔥 `Tigerbot` variant has been
  [deleted](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/438),
  `SUS-Chat-34B` is now the the top-ranked LLaMA model and the
  top-ranked chat model.

- 2023-12-07: SUS-Chat-34B is now available on
  [WiseModel🧠](https://wisemodel.cn/model/SUSTech/SUS-Chat-34B).

- 2023-12-06: Try [SUS-Chat-34B
  chat-ui](https://huggingface.co/spaces/SUSTech/SUS-Chat-34B).

- 2023-12-05: SUS-Chat-34B is now available on
  [ModelScope🤖](https://www.modelscope.cn/models/SUSTC/SUS-Chat-34B/summary)

- 2023-12-05: SUS-Chat-34B is ranked 2nd in [Open LLM
  leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
  and surpassed all models under 70B.

- 2023-12-01: SUS-Chat-34B is now available on
  [HuggingFace🤗](https://huggingface.co/SUSTech/SUS-Chat-34B).

# Introduction

<img src="https://hackmd.io/_uploads/HJlDtzhBa.png" id="fig-sus"
alt="Figure 1: DALL·E 2023-12-01 11.03.28 - An imposing, majestic wild boar combined with elements of a futuristic transformer robot. The boar itself should be intricately blended with these tra" />

**SUS-Chat-34B** is a 34B bilingual Chinese-English dialogue model,
jointly released by the **[Southern University of Science and
Technology](https://huggingface.co/SUSTech)** and
**[IDEA-CCNL](https://huggingface.co/IDEA-CCNL)**. This model is based
on [`01-ai/Yi-34B`](https://huggingface.co/01-ai/Yi-34B) and has been
fine-tuned on millions of high-quality, multilingual instruction data.
While maintaining the strong language capabilities of the base model,
the SUS-Chat-34B model has improved the model’s response to human
instructions through high-quality instruction fine-tuning and excels at
imitating human thought processes through chains of thought. It
introduces inter-instruction attention sharing in long texts, expanding
the window size from 4K to 8K, significantly enhancing the usability of
multi-turn dialogues.

It has surpassed all models of the same size in almost all benchmark
tests and is better suited to meet the practical needs of complex
multilingual tasks. Compared to larger models, SUS-Chat-34B remains
highly competitive and has achieved state-of-the-art performance in our
comprehensive evaluations.

SUS-Chat-34B model has the following highlights:

1.  Large-scale complex instruction following data: Trained with 1.4
    billion tokens of high-quality complex instruction data, covering
    Chinese and English, multi-turn dialogues, mathematics, reasoning,
    and various other types of instruction data;
2.  Strong performance in general tasks: The SUS-Chat-34B model excels
    in numerous mainstream Chinese and English tasks, surpassing other
    open-source instruction fine-tuned models of the same parameter
    scale. It also competes well against models with larger parameter
    scales;
3.  Longer context window and excellent multi-turn dialogue
    capabilities: Currently, SUS-Chat-34B supports an 8K context window,
    and is trained with a large amount of multi-turn instruction and
    single-multi-turn mixed data, demonstrating remarkable capabilities
    in long-text dialogue information focus and instruction follow-up.

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
C-Eval, BBH, GSM-8K, and MATH, to measure the model’s knowledge and
thinking capabilities. In these metrics, the SUS-Chat-34B model achieved
state-of-the-art performance. Additionally, we incorporated
[lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) to test
SUS-Chat and similar models on winogrande, hellaswag, arc, and
truthful-qa, assessing the model’s common-sense reasoning ability and
susceptibility to illusions.

Overall, the SUS-Chat-34B model significantly outperformed models of
similar scale and achieved the most advanced comprehensive performance.

<img
src="https://github.com/SUSTech-IDEA/SUS-Chat/raw/main/assets/radar.png"
id="fig-bench" alt="Figure 2: Benchmark" />

<div>

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: center;"><div width="50.0%"
data-layout-align="center">
<h2 id="english-understanding">English Understanding</h2>
<table>
<thead>
<tr class="header">
<th style="text-align: right;">Model</th>
<th style="text-align: center;">mmlu (0-shot)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: right;">GPT-4</td>
<td style="text-align: center;">83</td>
</tr>
<tr class="even">
<td style="text-align: right;">SUS-Chat-34B</td>
<td style="text-align: center;"><u>74.35</u></td>
</tr>
<tr class="odd">
<td style="text-align: right;">Qwen-72b-Chat</td>
<td style="text-align: center;"><strong>74.52</strong></td>
</tr>
<tr class="even">
<td style="text-align: right;">Deepseek-68b-Chat</td>
<td style="text-align: center;">69.43</td>
</tr>
<tr class="odd">
<td style="text-align: right;">OrionStar-Yi-34B-Chat</td>
<td style="text-align: center;">68.51</td>
</tr>
<tr class="even">
<td style="text-align: right;">Yi-34B-Chat</td>
<td style="text-align: center;">66.96</td>
</tr>
</tbody>
</table>
</div></td>
<td style="text-align: center;"><div width="50.0%"
data-layout-align="center">
<h2 id="chinese-capabilities">Chinese Capabilities</h2>
<table>
<colgroup>
<col style="width: 34%" />
<col style="width: 32%" />
<col style="width: 32%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: right;">Model</th>
<th style="text-align: center;">cmmlu (0-shot)</th>
<th style="text-align: center;">C-Eval (0-shot)<a href="#fn1"
class="footnote-ref" id="fnref1"
role="doc-noteref"><sup>1</sup></a></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: right;">GPT-4</td>
<td style="text-align: center;">71</td>
<td style="text-align: center;">69.9</td>
</tr>
<tr class="even">
<td style="text-align: right;">SUS-Chat-34B</td>
<td style="text-align: center;"><strong>78.68</strong></td>
<td style="text-align: center;"><strong>82.42</strong></td>
</tr>
<tr class="odd">
<td style="text-align: right;">Qwen-72b-Chat</td>
<td style="text-align: center;"><u>77.02</u></td>
<td style="text-align: center;"><u>77.22</u></td>
</tr>
<tr class="even">
<td style="text-align: right;">Deepseek-68b-Chat</td>
<td style="text-align: center;">48.51</td>
<td style="text-align: center;">59.7</td>
</tr>
<tr class="odd">
<td style="text-align: right;">OrionStar-Yi-34B-Chat</td>
<td style="text-align: center;">66.88</td>
<td style="text-align: center;">65.13</td>
</tr>
<tr class="even">
<td style="text-align: right;">Yi-34B-Chat</td>
<td style="text-align: center;">55.16</td>
<td style="text-align: center;">77.16</td>
</tr>
</tbody>
</table>
</div></td>
</tr>
</tbody>
</table>
<section id="footnotes" class="footnotes footnotes-end-of-document"
role="doc-endnotes">
<hr />
<ol>
<li id="fn1"><p>C-Eval results are evaluated on the validation
datasets<a href="#fnref1" class="footnote-back"
role="doc-backlink">↩︎</a></p></li>
</ol>
</section>

</div>

## Math & Reasoning

|                 Model | gsm8k (0-shot) | MATH (0-shot) | BBH (0-shot) |
|----------------------:|:--------------:|:-------------:|:------------:|
|                 GPT-4 |      91.4      |     45.8      |     86.7     |
|          SUS-Chat-34B |   **80.06**    |     28.7      |    67.62     |
|         Qwen-72b-Chat |  <u>76.57</u>  |   **35.9**    |  **72.63**   |
|     Deepseek-68b-Chat |     74.45      | <u>29.56</u>  | <u>69.73</u> |
| OrionStar-Yi-34B-Chat |     54.36      |     12.8      |    62.88     |
|           Yi-34B-Chat |     63.76      |     10.02     |    61.54     |

## More Tasks

|                 Model | winogrande (5-shot) | arc (25-shot) | hellaswag (10-shot) | TruthfulQA mc1 (0-shot) | TruthfulQA mc2 (0-shot) |
|----------------------:|:-------------------:|:-------------:|:-------------------:|:-----------------------:|:-----------------------:|
|                 GPT-4 |          —          |     94.5      |        91.4         |          59.00          |            —            |
|          SUS-Chat-34B |      **81.22**      | <u>81.54</u>  |        83.79        |        **40.64**        |        **57.47**        |
|         Qwen-72b-Chat |        76.09        |   **82.10**   |    <u>86.06</u>     |          39.17          |      <u>56.37</u>       |
|     Deepseek-68b-Chat |    <u>80.58</u>     |     81.29     |      **87.02**      |      <u>40.02</u>       |          50.64          |
| OrionStar-Yi-34B-Chat |        77.27        |     80.19     |        84.54        |          36.47          |          53.24          |
|           Yi-34B-Chat |        76.64        |     70.66     |        82.29        |          38.19          |          54.57          |

## Overall

|                 Model |  Average  |
|----------------------:|:---------:|
|          SUS-Chat-34B | **69.05** |
|         Qwen-72b-Chat |   68.41   |
|     Deepseek-68b-Chat |   62.91   |
| OrionStar-Yi-34B-Chat |   60.21   |
|           Yi-34B-Chat |   59.72   |

To reproduce the results, please start a corresponding vllm server and
refer to
[here](https://sustech-tlem.static.hf.space/index.html#start-evaluating-your-model-in-3-line).

# Usage

SUS-Chat-34B is a standard LLaMA model and should be seamlessly
compatible with the LLaMA ecosystem. We provide the following example to
demonstrate how it can be used for multi-turn dialogues.

Feel free to [open an
issue](https://github.com/SUSTech-IDEA/SUS-Chat/issues) if you have any
questions.

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer # 🤗 Transformers, or 
# from modelscope import AutoModelForCausalLM, AutoTokenizer # 🤖 ModelScope

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
# model_path = "SUSTC/SUS-Chat-34B" # ModelScope

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
[license](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt)
from [01-ai](https://huggingface.co/01-ai).
