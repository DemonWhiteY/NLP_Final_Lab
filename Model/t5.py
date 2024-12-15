from transformers import BertTokenizer, T5ForConditionalGeneration, Text2TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/t5-small-chinese-cluecorpussmall")
model = T5ForConditionalGeneration.from_pretrained("uer/t5-small-chinese-cluecorpussmall")