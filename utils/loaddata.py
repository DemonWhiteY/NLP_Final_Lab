import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import jieba

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)  # 解包批次数据

    # 使用 pad_sequence 对 src 和 trg 序列进行填充
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)  # 用 0 填充 src 序列
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=0)  # 用 0 填充 trg 序列

    return src_batch, trg_batch

class TextDataset(Dataset):
    def __init__(self, src_data, trg_data, src_vocab, trg_vocab):
        self.src_data = src_data
        self.trg_data = trg_data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return data

    def __getitem__(self, index):
        src_text = self.src_data[index]
        trg_text = self.trg_data[index]
        
        src_indices = [self.src_vocab.get(word, self.src_vocab['<unk>']) for word in src_text.split()]
        trg_indices = [self.trg_vocab.get(word, self.trg_vocab['<unk>']) for word in trg_text.split()]

        return torch.tensor(src_indices), torch.tensor(trg_indices)

    def __len__(self):
        return len(self.src_data)


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
        return articles,summarys

def create_vocab(articals,size):
    all_words = []
    for artical in articals:
        all_words.extend(artical.split())
        # 3. 使用 Counter 统计词频
    word_counts = Counter(all_words)

    # 4. 按照频率排序，构建词汇表
    # 默认的特殊标记：<pad>, <unk>, <sos>, <eos>
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    vocab = special_tokens + [word for word, count in word_counts.most_common(size)]

    # 5. 创建词汇表映射
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    # 打印词汇表
    print("Vocabularysize:", vocab.__len__())

    return word2idx

def load_dataset(file_path,data_size,src_size,trg_size):

    train_articles,train_summarys = load_jsonl(file_path,data_size)   
    
    print("训练集文章数量:", len(train_articles))
    print("训练集摘要数量:", len(train_summarys))
    article_vocab=create_vocab(train_articles,src_size)
    summary_vocab=create_vocab(train_summarys,trg_size)
    dataset = TextDataset(train_articles, train_summarys, article_vocab, summary_vocab)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True,collate_fn=collate_fn)
    return dataloader,article_vocab

def tokenize(text):
    seg_list = jieba.cut(text)
    print("分词：", "/ ".join(seg_list))
    return seg_list
    



        
