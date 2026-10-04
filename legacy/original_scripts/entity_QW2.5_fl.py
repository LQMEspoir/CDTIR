# =============================================================================
# 文件：legacy/original_scripts/entity_QW2.5_fl.py
# 模块职责：原始 Qwen2.5 + Focal Loss 实体抽取脚本，保留自定义模型包装与焦点损失实现。
# 主要数据流：NER token -> Qwen hidden/logits -> Focal Loss 训练 -> generate 预测。
# 阅读建议：先看本文件的公开函数/类，再沿 import 跟踪到 data_utils、modeling、evaluation 等公共模块。
# 维护说明：本版本仅新增解释性注释，不修改原有表达式、控制流、参数默认值或函数调用关系。
# =============================================================================

#project="Qwen2-NER-fintune
#64批次

import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

# -----------------------------------------------------------------------------
# 【类说明】CustomQwenNERModel
# - 职责：旧版 Qwen NER 包装模型，在基础 CausalLM 之上自定义 loss 计算并保持外部属性兼容。
# - 继承：nn.Module。
# - 使用方式：实例化后由训练/推理主流程调用其公开方法；内部状态由 __init__ 与保存的配置共同决定。
# -----------------------------------------------------------------------------
class CustomQwenNERModel(nn.Module):
    # -----------------------------------------------------------------------------
    # 【函数说明】CustomQwenNERModel.__init__
    # - 职责：初始化对象状态，保存后续训练、路由、损失或回调阶段需要的配置与依赖。
    # - 主要参数：base_model。
    # - 返回：无显式返回值；主要通过对象状态、文件、日志或外部训练流程产生效果。
    # - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
    # -----------------------------------------------------------------------------
    def __init__(self, base_model):
        super(CustomQwenNERModel, self).__init__()
        self.model = base_model
        self.focal_loss = FocalLoss(gamma=2.0)

    # -----------------------------------------------------------------------------
    # 【函数说明】CustomQwenNERModel.forward
    # - 职责：定义 PyTorch 模块的前向计算路径；输入张量经过当前模块计算后返回 logits、loss、hidden state 或封装输出。
    # - 主要参数：input_ids、attention_mask=None、labels=None。
    # - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
    # - 关键过程：保持标准 PyTorch/Transformers 调用约定；具体分支根据类的职责计算路由、adapter logits、任务损失或测试输出。
    # - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
    # -----------------------------------------------------------------------------
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        logits = outputs.logits

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.focal_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

    # -----------------------------------------------------------------------------
    # 【函数说明】CustomQwenNERModel.__getattr__
    # - 职责：当包装模型自身找不到属性时，将访问转发给内部基础模型，以兼容 Transformers/Trainer 对模型属性的读取。
    # - 主要参数：name。
    # - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
    # - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
    # -----------------------------------------------------------------------------
    def __getattr__(self, name):
        # 将未定义的方法都传递给 self.model
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

# -----------------------------------------------------------------------------
# 【类说明】FocalLoss
# - 职责：多分类 Focal Loss 实现，通过 (1-p_t)^gamma 降低易分类 token 的损失权重、聚焦困难样本。
# - 继承：nn.Module。
# - 使用方式：实例化后由训练/推理主流程调用其公开方法；内部状态由 __init__ 与保存的配置共同决定。
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    # -----------------------------------------------------------------------------
    # 【函数说明】FocalLoss.__init__
    # - 职责：初始化对象状态，保存后续训练、路由、损失或回调阶段需要的配置与依赖。
    # - 主要参数：alpha=0.25、gamma=2.0、reduction='mean'。
    # - 返回：无显式返回值；主要通过对象状态、文件、日志或外部训练流程产生效果。
    # - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
    # -----------------------------------------------------------------------------
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    # -----------------------------------------------------------------------------
    # 【函数说明】FocalLoss.forward
    # - 职责：定义 PyTorch 模块的前向计算路径；输入张量经过当前模块计算后返回 logits、loss、hidden state 或封装输出。
    # - 主要参数：inputs、targets。
    # - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
    # - 关键过程：保持标准 PyTorch/Transformers 调用约定；具体分支根据类的职责计算路由、adapter logits、任务损失或测试输出。
    # - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
    # -----------------------------------------------------------------------------
    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算 Focal Loss 的权重
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

            
new_path = "entity_output.jsonl"  # 新生成的文件路径

# -----------------------------------------------------------------------------
# 【函数说明】process_func
# - 职责：把原始样本转换成模型训练所需的 token、attention mask 与 labels；通常同时应用 chat template 和最大长度截断。
# - 主要参数：example。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：模型推理/张量路径需保持 device 与 dtype 一致。
# -----------------------------------------------------------------------------
def process_func(example):
    """
    将数据集进行预处理
    """

    MAX_LENGTH = 256 
    input_ids, attention_mask, labels = [], [], []
    system_prompt = """
    你是一个文本实体识别领域的专家，你需要从给定的句子中提取 目的地; 出发地; 返程地; 餐饮; 住宿; 产品; 交通; 预算; 时长; 时间; 人群; 强度; 人气; 人数; 天气. 以 json 格式输出, 如 {\"entity_text\": \"北京\", \"entity_label\": \"目的地\"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出\"没有找到任何实体\".
    """
    
    
    instruction = tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


# -----------------------------------------------------------------------------
# 【函数说明】predict
# - 职责：统一预测流程：加载模型/adapter 或 Router 包，构造输入 prompt，执行生成，解析结构化结果，并按需要保存到文件。
# - 主要参数：messages、model、tokenizer。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 关键过程：区分 Router 包、普通 adapter 和基础模型三条路径；对输入构造统一 prompt；生成后用 parsing 模块转成结构化实体/关系。
# - 注意：模型推理/张量路径需保持 device 与 dtype 一致；会输出日志或告警信息。
# -----------------------------------------------------------------------------
def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(response)
     
    return response

# 确保模型目录已下载
model_dir = "LLM/qwen/Qwen2.5-7B-Instruct"
model_id = "Qwen2.5-7B-Instruct"



# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, 
    use_fast=False, 
    trust_remote_code=True, 
    use_cache=False
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_dir, device_map="auto", torch_dtype=torch.bfloat16
)
base_model.enable_input_require_grads()
#禁用缓存
base_model.config.use_cache = False

# 训练集
train_df = pd.read_json("entity_train.jsonl", lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

peft_model = get_peft_model(base_model, config)
model = CustomQwenNERModel(peft_model)


args = TrainingArguments(
    output_dir="output/Qwen2.5-NER",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=5,
    save_steps=100,
    learning_rate=3e-5,  #学习率
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen2.5-NER",
    experiment_name="Qwen2.5-7B-Instruct_fl",
    description="使用通义千问Qwen2.5-7B-Instruct模型在NER数据集上微调，实现关键实体识别任务。",
    config={
        "model": model_id,
        "model_dir": model_dir,
        "dataset": "entity_output.jsonl",
    },
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()

#测试集
test_df = pd.read_json("entity_test.jsonl", lines=True)

test_text_list = []
predictions = []  # 用于保存所有的预测结果


for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    output_value = row['output']
    
    # 准备输入消息
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    # 获取预测结果
    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})
    
    # 将预测结果和输入一起添加到 test_text_list 中
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    test_text_list.append(swanlab.Text(result_text, caption=response))
    
    # 保存预测值到 predictions 列表
    predictions.append({
        "instruction": instruction,
        "input": input_value,
        "predicted_output": response,
        "output":output_value
    })

# 打印预测结果
for prediction in predictions:
    print(f"Input: {prediction['input']}")
    print(f"Original Output: {prediction['output']}")  # 打印原始输出
    print(f"Predicted Output: {prediction['predicted_output']}")
    print("-" * 50)

# 将预测结果保存到 JSON 文件
output_json_file = "Qwen2.5_predicted_data.json"
with open(output_json_file, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=4)

# 使用 swanlab 记录预测结果
swanlab.log({"Prediction": test_text_list})
swanlab.finish()

print(f"Predictions have been saved to {output_json_file}")

