---
license: apache-2.0
tags:
- codet5
datasets:
- conala-mined
- Python3.7-API
- MBPP
inference: false
---

# Fine-tuned-CodeT5 (fine-tune the CodeT5-base model) 

Fine-tuned CodeT5 model. The code for fine-tuning is released in [this repository](https://github.com/gangqinxiao13/CodeT5-code-generation.git)

The pre-trained model was introduced in the paper [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models
for Code Understanding and Generation](https://arxiv.org/abs/2109.00859) by Yue Wang, Weishi Wang, Shafiq Joty, Steven C.H. Hoi and first released in [this repository](https://github.com/salesforce/CodeT5). 


## Model description


This model is fine-tuned on CodeT5-base to accomplish the transfer-learning downstream task-Python code generation. CodeT5 is a unified pre-trained encoder-decoder Transformer model that better leverages the code semantics conveyed from the developer-assigned identifiers. Comprehensive experiments show that CodeT5 significantly outperforms prior methods on understanding tasks such as code defect detection and clone detection, and generation tasks across various directions including PL-NL, NL-PL, and PL-PL. 


### How to use

Here is how to use this model:

```python
from transformers import RobertaTokenizer, T5ForConditionalGeneration

tokenizer = RobertaTokenizer.from_pretrained('gangqinxiao13/fine-tuned-codet5')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/fine-tuned-codet5')

text = "print hello"
task_prefix = "Generate Python code from natural language:"
input = tokenizer(
        [task_prefix + input],
        return_tensors="pt",
        padding='max_length',
        max_length=512,
        truncation=True
    )
input_ids = tokenizer(text, return_tensors="pt").input_ids

outputs = model.generate(
    input_ids=input['input_ids'],
    attention_mask=input["attention_mask"],
    do_sample=False,
    max_new_tokens=512
    )
generated_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(generated_code[0]) 
```

## Pre-training data

The CodeT5 model was pretrained on CodeSearchNet [Husain et al., 2019](https://arxiv.org/abs/1909.09436). Additionally, the authors collected two datasets of C/CSharp from [BigQuery1](https://console.cloud.google.com/marketplace/details/github/github-repos) to ensure that all downstream tasks have overlapped programming languages with the pre-training data. In total, around 8.35 million instances are used for pretraining. 

## Fine-tuning data

The fine-tuned model was trained on conala-mined [Yin et al., 2019](https://arxiv.org/abs/1805.08949), Re-sampled Python3.7 API Knowledge [Xu et al., 2020](https://arxiv.org/abs/2004.09015) and MBPP (Mostly Basic Python Programming) [Austin et al., 2021](https://arxiv.org/abs/2108.07732)