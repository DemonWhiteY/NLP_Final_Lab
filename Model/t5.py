from transformers import T5Tokenizer, T5ForConditionalGeneration
import sentencepiece
def getT5Model(input):
    # 加载模型和分词器
    model_name = "t5-small"  # 或使用更大的模型如 t5-base/t5-large
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    # 添加任务前缀
    input_text = f"summarize: {input}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

    # 生成摘要
    outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("摘要：", summary)
    return summary
