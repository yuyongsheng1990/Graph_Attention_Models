

### Infini-attention

20240425, https://github.com/dingo-actual/infini-transformer/tree/main


### infini-mini-transformer

https://github.com/jiahe7ay/infini-mini-transformer/tree/main
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
