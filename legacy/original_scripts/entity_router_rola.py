# =============================================================================
# 文件：legacy/original_scripts/entity_router_rola.py
# 模块职责：原始 Router + 多 LoRA 实体抽取实验脚本。当前正式 router_multilora 实现以该文件的算法语义为兼容参考。
# 主要数据流：输入文本 -> 基座 hidden state -> Router 类别权重 -> 多个 LoRA 分支 logits -> 加权融合 -> CE + Router BCE 联合训练。
# 阅读建议：先看本文件的公开函数/类，再沿 import 跟踪到 data_utils、modeling、evaluation 等公共模块。
# 维护说明：本版本仅新增解释性注释，不修改原有表达式、控制流、参数默认值或函数调用关系。
# =============================================================================

# ✅ 修复 Qwen2.5 多LoRA NER 微调无法使用 GPU 的问题
# 修复点：Router 和所有 LoRA 模块强制迁移到 GPU，确保模型落地在 CUDA 上运行

import os
import gc
import json
import torch
import pandas as pd
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from swanlab.integration.transformers import SwanLabCallback  # ✅ 使用新路径
import swanlab

# ========== 设备设置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备: {device}")
torch.cuda.empty_cache()
gc.collect()

# ========== Router 模块 ==========
# -----------------------------------------------------------------------------
# 【类说明】SimpleCategoryRouter
# - 职责：轻量神经路由器，将 pooled hidden state 映射到每个实体/关系类别的独立 Router logit。
# - 继承：nn.Module。
# - 使用方式：实例化后由训练/推理主流程调用其公开方法；内部状态由 __init__ 与保存的配置共同决定。
# -----------------------------------------------------------------------------
class SimpleCategoryRouter(nn.Module):
    # -----------------------------------------------------------------------------
    # 【函数说明】SimpleCategoryRouter.__init__
    # - 职责：初始化对象状态，保存后续训练、路由、损失或回调阶段需要的配置与依赖。
    # - 主要参数：hidden_size、num_categories。
    # - 返回：无显式返回值；主要通过对象状态、文件、日志或外部训练流程产生效果。
    # - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
    # -----------------------------------------------------------------------------
    def __init__(self, hidden_size, num_categories):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_categories)
        )

    # -----------------------------------------------------------------------------
    # 【函数说明】SimpleCategoryRouter.forward
    # - 职责：定义 PyTorch 模块的前向计算路径；输入张量经过当前模块计算后返回 logits、loss、hidden state 或封装输出。
    # - 主要参数：hidden。
    # - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
    # - 关键过程：保持标准 PyTorch/Transformers 调用约定；具体分支根据类的职责计算路由、adapter logits、任务损失或测试输出。
    # - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
    # -----------------------------------------------------------------------------
    def forward(self, hidden):
        pooled = hidden.mean(dim=1)
        return self.linear(pooled)

# ========== 多LoRA模型结构 ==========
# -----------------------------------------------------------------------------
# 【类说明】MultiLoRAModel
# - 职责：原始实验版多 LoRA 包装模型：管理类别 LoRA 分支、Router 权重与联合训练损失。
# - 继承：nn.Module。
# - 使用方式：实例化后由训练/推理主流程调用其公开方法；内部状态由 __init__ 与保存的配置共同决定。
# -----------------------------------------------------------------------------
class MultiLoRAModel(nn.Module):
    # -----------------------------------------------------------------------------
    # 【函数说明】MultiLoRAModel.__init__
    # - 职责：初始化对象状态，保存后续训练、路由、损失或回调阶段需要的配置与依赖。
    # - 主要参数：base_model、lora_config、category_labels。
    # - 返回：无显式返回值；主要通过对象状态、文件、日志或外部训练流程产生效果。
    # - 注意：模型推理/张量路径需保持 device 与 dtype 一致。
    # -----------------------------------------------------------------------------
    def __init__(self, base_model, lora_config, category_labels):
        super().__init__()
        self.categories = category_labels
        self.num_categories = len(category_labels)

        # Router
        self.router = SimpleCategoryRouter(base_model.config.hidden_size, self.num_categories)
        param_dtype = next(base_model.parameters()).dtype
        param_device = next(base_model.parameters()).device
        self.router.to(dtype=param_dtype, device=param_device)

        # 构建每个分类对应的 LoRA 模型并迁移到 GPU
        self.lora_modules = nn.ModuleDict()
        for cat in category_labels:
            peft_model = get_peft_model(base_model, lora_config)
            peft_model = peft_model.to(dtype=param_dtype, device=param_device)
            self.lora_modules[cat] = peft_model

    # -----------------------------------------------------------------------------
    # 【函数说明】MultiLoRAModel.forward
    # - 职责：定义 PyTorch 模块的前向计算路径；输入张量经过当前模块计算后返回 logits、loss、hidden state 或封装输出。
    # - 主要参数：input_ids、attention_mask、labels=None、router_targets=None。
    # - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
    # - 关键过程：保持标准 PyTorch/Transformers 调用约定；具体分支根据类的职责计算路由、adapter logits、任务损失或测试输出。
    # - 注意：会输出日志或告警信息。
    # -----------------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, labels=None, router_targets=None):
        base_hidden = self.lora_modules[self.categories[0]].base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]

        router_logits = self.router(base_hidden)
        router_weights = torch.sigmoid(router_logits)

        outputs = []
        for i, cat in enumerate(self.categories):
            out = self.lora_modules[cat](
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            ).logits
            weighted_out = out * router_weights[:, i].unsqueeze(-1).unsqueeze(-1)
            outputs.append(weighted_out)

        final_logits = sum(outputs)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if router_targets is not None:
                router_loss = nn.BCEWithLogitsLoss()(router_logits, router_targets.float())
                loss = loss + 0.5 * router_loss

        return final_logits if loss is None else {"loss": loss, "logits": final_logits}

    # -----------------------------------------------------------------------------
    # 【函数说明】MultiLoRAModel.gradient_checkpointing_enable
    # - 职责：把梯度检查点开启操作透传到底层模型，用计算换显存，降低训练峰值显存占用。
    # - 主要参数：**kwargs。
    # - 返回：无显式返回值；主要通过对象状态、文件、日志或外部训练流程产生效果。
    # - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
    # -----------------------------------------------------------------------------
    def gradient_checkpointing_enable(self, **kwargs):
        for module in self.lora_modules.values():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable(**kwargs)

    # -----------------------------------------------------------------------------
    # 【函数说明】MultiLoRAModel.gradient_checkpointing_disable
    # - 职责：关闭底层模型的梯度检查点功能，恢复普通前向/反向计算路径。
    # - 主要参数：无显式业务参数（仅可能包含 self/cls）。
    # - 返回：无显式返回值；主要通过对象状态、文件、日志或外部训练流程产生效果。
    # - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
    # -----------------------------------------------------------------------------
    def gradient_checkpointing_disable(self):
        for module in self.lora_modules.values():
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()

# ========== 数据预处理 ==========
entity_labels = ["目的地", "出发地", "返程地", "餐饮", "住宿", "产品", "交通", "预算", "时长", "时间", "人群", "强度", "人气", "人数", "天气"]

# -----------------------------------------------------------------------------
# 【函数说明】process_func
# - 职责：把原始样本转换成模型训练所需的 token、attention mask 与 labels；通常同时应用 chat template 和最大长度截断。
# - 主要参数：example。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：模型推理/张量路径需保持 device 与 dtype 一致。
# -----------------------------------------------------------------------------
def process_func(example):
    MAX_LENGTH = 256
    system_prompt = """
    你是一个文本实体识别领域的专家，你需要从给定的句子中提取 目的地; 出发地; 返程地; 餐饮; 住宿; 产品; 交通; 预算; 时长; 时间; 人群; 强度; 人气; 人数; 天气. 以 json 格式输出, 如 {\"entity_text\": \"北京\", \"entity_label\": \"目的地\"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出\"没有找到任何实体\".
    """
    instruction = tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    label_vector = [0] * len(entity_labels)
    for ent in entity_labels:
        if f'"entity_label": "{ent}"' in example['output']:
            label_vector[entity_labels.index(ent)] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "router_targets": label_vector
    }

# ========== 模型与训练 ==========
tokenizer = AutoTokenizer.from_pretrained(
    "LLM/qwen/Qwen2.5-7B-Instruct",
    use_fast=False, trust_remote_code=True, use_cache=False
)
base_model = AutoModelForCausalLM.from_pretrained(
    "LLM/qwen/Qwen2.5-7B-Instruct",
    device_map="auto", torch_dtype=torch.bfloat16
)
base_model.enable_input_require_grads()
base_model.config.use_cache = False

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = MultiLoRAModel(base_model=base_model, lora_config=config, category_labels=entity_labels)
model = model.to(device)  # ✅ 强制放入 GPU

train_df = pd.read_json("entity_train.jsonl", lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

args = TrainingArguments(
    output_dir="output/Qwen2.5-NER",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=5,
    save_steps=100,
    learning_rate=3e-5,
    gradient_checkpointing=True,
    save_on_each_node=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen2.5-NER",
    experiment_name="Qwen2.5-7B-Instruct",
    description="使用Qwen2.5-7B-Instruct进行NER多LoRA微调",
    config={"model": "Qwen2.5-7B-Instruct", "dataset": "entity_output.jsonl"},
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

# ✅ 强制 Trainer 使用 GPU 模型
del trainer.model  # 删除原始引用，避免 CPU fallback
trainer.model = model.to(device)

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

