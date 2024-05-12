### Infini-attention

20240425, https://github.com/dingo-actual/infini-transformer/tree/main

### infini-mini-transformer

20240501，https://github.com/jiahe7ay/infini-mini-transformer/tree/main
error: Detected kernel version 4.18.0, which is below the recommended minimum of 5.5.0;
solution: uname -r.系统内核版本太低！
LLM模型需要较大的显卡内存，我这破机器上跑不起来，但我感觉这个项目还是比较简单易上手的。
GemmaForCausalLM model <- GemmaPreTrainedModel,

- self.model: GemmaModel
- inputs:
  - input_ids, 将输入词汇索引input_ids转换为对应的word embedding.
  - attention mask, 掩码
  - position_ids
  - past_key_value
- outputs:
  - linear logits <- nn.Linear(outputs[0]), 词汇表上的概率分布logits
  - loss,
  - past_key_value
  - attentions, attention 权重

### LLMs-from-scratch

20240506，LLMs教程：https://github.com/rasbt/LLMs-from-scratch

- 第一部分：关于LLM的基础。包含了各种transformer，self-attention的计算。  
- 第二部分：大模型本身。LLM本身相关的内容，如何pretrained，inference，evaluation。  
- 第三部分：fine-tune LLM。这是对绝大部分人有用的一部分，如何让LLM应用在各大领域。包括：paper tips, 多模态VLM.
  - ch03, causal multi-head attention mechanism
  - ch04, implement a GPT-like LLM architecture
  - ch05, pre-trained LLM
  - ch06, finetuning LLM based on classification-finetuning
