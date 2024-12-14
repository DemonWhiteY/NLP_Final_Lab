from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
from transformers import Trainer, TrainingArguments
import json
from datasets import Dataset
import torch
# 加载 BART 模型和分词器

def train():
    model_name = 'fnlp/bart-base-chinese'  # 适用于文本摘要的预训练BART模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    def load_jsonl(file_path,size):
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = []
            summarys=[]
            index=0    
            for line in f:
                index+=1
                linedata=json.loads(line.strip())
                article=" ".join(linedata['article'])
                summary=linedata['summary']
                articles.append(article)
                summarys.append(summary)
                if index>size:
                    break
            data={"article":articles,"summary":summarys}
            return data

    data=load_jsonl("Data/train.simple.label.jsonl",10000)

    dataset=Dataset.from_dict(data)

    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    def preprocess_function(examples):
        inputs = tokenizer(examples["article"], max_length=1024, truncation=True, padding="max_length")
        targets = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length")
        inputs["labels"] = targets["input_ids"]
        inputs = {key: torch.tensor(val).to(device) for key, val in inputs.items()}
        return inputs

    # 对训练集和验证集进行预处理
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # 设置数据格式，返回torch tensor
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir="./results",            # 输出目录
        eval_strategy="epoch",       # 每个epoch后进行评估
        learning_rate=5e-5,                # 学习率
        per_device_train_batch_size=4,     # 每个设备上的训练批次大小
        per_device_eval_batch_size=8,      # 每个设备上的评估批次大小
        num_train_epochs=3,                # 训练轮数
        weight_decay=0.01,                 # 权重衰减
        save_strategy="epoch",             # 每个epoch保存一次模型
        logging_dir="./logs",              # 日志目录
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    trainer.evaluate()

    model.save_pretrained("./fine_tuned_bart_chinese")
    tokenizer.save_pretrained("./fine_tuned_bart_chinese")
    # 输入文本


def summarize(text):
    model_name = 'fnlp/bart-base-chinese'  # 适用于文本摘要的预训练BART模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    # 确保模型在正确的设备上（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 使用加载的模型和分词器进行推理或其他任务

    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # 生成摘要
    summary_ids = model.generate(inputs["input_ids"], max_length=800, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    # 解码生成的摘要
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


def summarize_tuned(text):
    model = BartForConditionalGeneration.from_pretrained("./fine_tuned_bart_chinese")
    tokenizer = BertTokenizer.from_pretrained("./fine_tuned_bart_chinese")

    # 确保模型在正确的设备上（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 使用加载的模型和分词器进行推理或其他任务

    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # 生成摘要
    summary_ids = model.generate(inputs["input_ids"], max_length=800, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    # 解码生成的摘要
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary