# =============================================================================
# 文件：legacy/original_scripts/relation.py
# 模块职责：原始关系抽取训练、生成、BLEU/ROUGE/三元组指标与可视化脚本，用作 legacy 对照。
# 主要数据流：关系数据 -> 预处理/tokenize -> 模型训练/生成 -> 三元组解析 -> 文本与结构化指标 -> 曲线/图表。
# 阅读建议：先看本文件的公开函数/类，再沿 import 跟踪到 data_utils、modeling、evaluation 等公共模块。
# 维护说明：本版本仅新增解释性注释，不修改原有表达式、控制流、参数默认值或函数调用关系。
# =============================================================================

import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import evaluate
import sacrebleu
import re
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType

# =================== 加载数据并预处理 =================== #
file_path = "text_entity_relation_extraction.json"
with open(file_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# -----------------------------------------------------------------------------
# 【函数说明】preprocess_data
# - 职责：把原始关系抽取样本转换为训练脚本期望的输入/目标格式。
# - 主要参数：entry。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
# -----------------------------------------------------------------------------
def preprocess_data(entry):
    text = entry["text"]
    spo_list = entry["spo_list"]
    triples = []
    for spo in spo_list:
        subject = text[spo["subject"][0]:spo["subject"][1]]
        predicate = spo["predicate"]
        object_ = text[spo["object"][0]:spo["object"][1]]
        triples.append(f"<{subject}, {predicate}, {object_}>")
    prompt_text = (
        "你是一个故障诊断和文本实体关系抽取领域的专家。\n"
        "现在有以下实体类型：{轴承类型, 故障模式, 故障特征, 故障原因, 故障位置, 解决对策, 预防措施, 严重程度, 事故后果}，\n"
        "predicate 类型：{导致, 建议, 位于, 预防, 表征, 发生于, 属于, 引发}。\n"
        "请从下面的文本中提取所有三元组，输出格式要求如下：每一行为一个三元组，形式为 <主语, 谓语, 宾语>。\n"
        "请不要输出其他解释性语言，仅输出提取出的三元组：\n\n"
        f"{text}"
    )
    return {
        "input_text": prompt_text,
        "target_text": "\n".join(triples)
    }

processed_data = [preprocess_data(entry) for entry in raw_data]
raw_dataset = Dataset.from_list(processed_data)

# =================== 分词器 + 数据编码 =================== #
model_id = "qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
tokenizer.padding_side = "left"

# -----------------------------------------------------------------------------
# 【函数说明】tokenize_function
# - 职责：使用 tokenizer 对预处理后的文本进行编码、截断，并构造训练需要的 token 字段。
# - 主要参数：examples。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：模型推理/张量路径需保持 device 与 dtype 一致。
# -----------------------------------------------------------------------------
def tokenize_function(examples):
    inputs = tokenizer(
        examples["input_text"], padding="max_length", truncation=True, max_length=512
    )
    targets = tokenizer(
        examples["target_text"], padding="max_length", truncation=True, max_length=512
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_datasets = raw_dataset.map(
    tokenize_function, batched=True, remove_columns=["input_text", "target_text"]
)

train_size = int(len(tokenized_datasets) * 0.9)
train_dataset = tokenized_datasets.select(range(train_size))
eval_dataset = tokenized_datasets.select(range(train_size, len(tokenized_datasets)))
eval_raw_dataset = raw_dataset.select(range(train_size, len(raw_dataset)))

# =================== 加载模型并设置QLoRA =================== #
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16, use_cache=False
)
model.enable_input_require_grads()
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
    inference_mode=False, r=16, lora_alpha=64, lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, config)

training_args = TrainingArguments(
    output_dir="./output/Qwen2-Relation-QLoRA",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=1,
    num_train_epochs=5,
    save_steps=100,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
)

trainer.train()

# =================== 推理 + 清洗 =================== #
# -----------------------------------------------------------------------------
# 【函数说明】extract_triples_from_text
# - 职责：从模型生成字符串中解析 <subject, predicate, object> 或等价格式的关系三元组。
# - 主要参数：text。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
# -----------------------------------------------------------------------------
def extract_triples_from_text(text):
    pattern = re.compile(r"<(.+?),\s*(.+?),\s*(.+?)>")
    return set(pattern.findall(text))

# -----------------------------------------------------------------------------
# 【函数说明】generate_predictions
# - 职责：在验证/测试样本上批量或逐条调用 generate，并收集解码后的预测文本。
# - 主要参数：model、dataset、tokenizer、batch_size=4。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：模型推理/张量路径需保持 device 与 dtype 一致。
# -----------------------------------------------------------------------------
def generate_predictions(model, dataset, tokenizer, batch_size=4):
    model.eval()
    predictions, references, inputs_list = [], [], []
    raw_data = dataset.to_dict()
    input_texts = raw_data["input_text"]
    target_texts = raw_data["target_text"]

    for i in tqdm(range(0, len(input_texts), batch_size)):
        inputs = tokenizer(
            input_texts[i:i+batch_size], return_tensors="pt",
            truncation=True, padding="max_length", max_length=512
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
        references.extend(target_texts[i:i+batch_size])
        inputs_list.extend(input_texts[i:i+batch_size])

    return inputs_list, predictions, references

inputs_list, predictions, references = generate_predictions(model, eval_raw_dataset, tokenizer)

# =================== 保存预测结果 =================== #
results = []
for input_text, pred, ref in zip(inputs_list, predictions, references):
    pred_triples = extract_triples_from_text(pred)
    ref_triples = extract_triples_from_text(ref)
    correct = pred_triples & ref_triples
    errors = pred_triples - ref_triples
    missed = ref_triples - pred_triples
    results.append({
        "input": input_text,
        "prediction": pred,
        "reference": ref,
        "correct_triples": list(correct),
        "error_triples": list(errors),
        "missed_triples": list(missed)
    })

with open("prediction_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# =================== BLEU + ROUGE =================== #
# -----------------------------------------------------------------------------
# 【函数说明】compute_bleu
# - 职责：根据参考文本和预测文本计算 BLEU 类生成质量指标。
# - 主要参数：preds、refs。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
# -----------------------------------------------------------------------------
def compute_bleu(preds, refs):
    return sacrebleu.corpus_bleu(preds, [[r] for r in refs], smooth_method="exp").score

# -----------------------------------------------------------------------------
# 【函数说明】compute_rouge
# - 职责：计算预测与参考文本之间的 ROUGE 指标，用于兼容原关系抽取脚本的文本评价。
# - 主要参数：preds、refs。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
# -----------------------------------------------------------------------------
def compute_rouge(preds, refs):
    rouge = evaluate.load("metrics/rouge")
    score = rouge.compute(predictions=preds, references=refs)
    return score.get("rougeL", 0.0)

bleu_score = compute_bleu(predictions, references)
rouge_l_score = compute_rouge(predictions, references)
print(f"\n📊 BLEU-4 Score: {bleu_score:.2f}%")
print(f"📊 ROUGE-L Score: {rouge_l_score:.2f}%")

# =================== 三元组评估（基于清洗后） =================== #
# -----------------------------------------------------------------------------
# 【函数说明】compute_triple_metrics
# - 职责：对解析后的三元组进行 exact match，计算结构化关系抽取 Precision/Recall/F1。
# - 主要参数：preds、refs。
# - 返回：返回处理后的对象/指标/张量/路径等结果；实际结构由各 return 分支决定。
# - 注意：本函数主要做内存计算；除被调用对象自身行为外无额外持久化副作用。
# -----------------------------------------------------------------------------
def compute_triple_metrics(preds, refs):
    tp, fp, fn = 0, 0, 0
    for pred_text, ref_text in zip(preds, refs):
        pred_set = extract_triples_from_text(pred_text)
        ref_set = extract_triples_from_text(ref_text)
        tp += len(pred_set & ref_set)
        fp += len(pred_set - ref_set)
        fn += len(ref_set - pred_set)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {
        "triple_precision": round(prec * 100, 2),
        "triple_recall": round(rec * 100, 2),
        "triple_f1": round(f1 * 100, 2)
    }

triple_metrics = compute_triple_metrics(predictions, references)
print(f"\n📊 三元组评估：Precision={triple_metrics['triple_precision']}% | Recall={triple_metrics['triple_recall']}% | F1={triple_metrics['triple_f1']}%")

# =================== 可视化 =================== #
# -----------------------------------------------------------------------------
# 【函数说明】plot_all_metrics
# - 职责：把训练曲线和评测指标绘制/保存为图像，便于复现实验报告中的可视化结果。
# - 主要参数：bleu、rouge、triple。
# - 返回：无显式返回值；主要通过对象状态、文件、日志或外部训练流程产生效果。
# - 注意：可能读写磁盘文件/模型权重，请保证输出目录可写。
# -----------------------------------------------------------------------------
def plot_all_metrics(bleu, rouge, triple):
    metrics = ["BLEU-4", "ROUGE-L", "Triple Precision", "Triple Recall", "Triple F1"]
    scores = [bleu, rouge, triple["triple_precision"], triple["triple_recall"], triple["triple_f1"]]
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 5))
    bars = plt.bar(metrics, scores)
    plt.ylim(0, 100)
    plt.title("模型评估指标")
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y + 1, f"{y:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig("all_metrics.png")
    plt.show()

plot_all_metrics(bleu_score, rouge_l_score, triple_metrics)
