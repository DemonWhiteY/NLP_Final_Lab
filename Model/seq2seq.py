import torch
import torch.nn as nn
import torch.optim as optim
import random
from utils.loaddata import load_dataset
from utils.loaddata import tokenize
from tqdm import tqdm




class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        # 定义 LSTM 层
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)  # 将输入的词索引转换为向量
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)  # 转换成词向量
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化细胞状态
        
        out, (hn, cn) = self.lstm(embedded, (h0, c0))  # LSTM 输出
        return hn, cn  # 返回最后的隐藏状态和细胞状态
    
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)  # 将目标词索引转换为向量
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 最后一层输出词汇大小的预测
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)  # 将目标词转换为向量
        out, (hn, cn) = self.lstm(embedded, (hidden, cell))  # LSTM 层输出
        out = self.fc(out)  # 通过全连接层得到最终预测
        out = self.softmax(out)  # 应用 softmax 获取每个词的概率
        return out, hn, cn  # 返回预测结果和新的 hidden、cell 状态


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        
        # Encoder
        hidden, cell = self.encoder(src)
        
        # Decoder 初始输入是 [batch_size, 1]
        input = trg[:, 0].unsqueeze(1)  # 第一列是目标序列的开始标记
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output.squeeze(1)
            
            # 使用教师强制（Teacher Forcing）策略
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2)  # 选择概率最高的词
            
            input = trg[:, t].unsqueeze(1) if teacher_force else top1
            
        return outputs


# 定义超参数
input_size = 15005  # 输入词汇表大小
output_size = 10005  # 输出词汇表大小
data_size=10000
hidden_size = 256  # 隐层大小
num_layers = 2  # LSTM 层数
print(torch.__version__)  # 查看当前 PyTorch 版本
print(torch.cuda.is_available())  # 查看是否有可用的 CUDA 设备
print(torch.version.cuda)  # 查看 PyTorch 支持的 CUDA 版本

if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available!")
else:
    device = torch.device("cpu")   # 使用 CPU
    print("CUDA is not available, using CPU.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 Encoder 和 Decoder
encoder = Encoder(input_size, hidden_size, num_layers).to(device)
decoder = Decoder(output_size, hidden_size, num_layers).to(device)

# 初始化 Seq2Seq 模型
model = Seq2Seq(encoder, decoder, device).to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 假设我们有训练数据
# src, trg 是已经经过分词的张量
# 训练循环
num_epochs = 10  # 设置为你想要的训练轮数
train,article_vocab=load_dataset("Data/train.simple.label.jsonl",data_size,input_size-5,output_size-5);


for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    for batch_idx, (src, trg) in tqdm(enumerate(train), total=len(train), desc=f'Epoch {epoch+1}/{num_epochs}', ncols=100):
        # src: 输入序列， trg: 目标序列
        src, trg = src.to(device), trg.to(device)

        output = model(src, trg)
        # 计算损失
        loss = criterion(output.view(-1, output_size), trg.view(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    if (epoch + 1) % 5 == 0:  # 每5个epoch保存一次
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        print(f"Model saved after epoch {epoch+1}")

def preview(words):
    model = Seq2Seq(encoder, decoder, device).to(device)

    # 加载模型参数
    model.load_state_dict(torch.load('model_state_dict.pth'))
    input = [article_vocab.get(word, article_vocab['<unk>']) for word in tokenize(words)]
    
    # 设置模型为评估模式
    model.eval()