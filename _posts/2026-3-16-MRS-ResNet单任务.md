---
layout: post
title: MRS残差网络
description: 残差网络ResNet
categories: MRS
tags: [MRS]
---

## ResNet

[ResNet从理论到实践（一）|ResNet原理 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/378037292)

![image](/images/posts/2026-3-16-MRS-ResNet单任务/Resnet.png)



- 1）直接使用两个单任务模型分别预测主音和模式；

- 2）使用两个单任务模型分别预测体系和主音，然后间接计算这两个结果的模式；

```
公式：System同宫系统+Tonic主音=Pattern模式
```

- 3）使用一个多任务模型识别系统、主音和模式，其中模式既可以直接也可以间接导出。

**类型识别Type**，由于其数据分布不均、识别难度大且与其他三项关系不大，我们直接使用单一模型进行预测，而不将其加入多任务模型。



### ResNet18单任务网络架构

![image](/images/posts/2026-3-16-MRS-ResNet单任务/ResNet-Single-task.png)



1. **初始卷积层（Conv）**：
   - 卷积核大小（k）为 7x7，步长（s）为 1，输出通道数（c）为 64。
   - 这一层用于初步提取特征。
2. **最大池化层（Max Pooling）**：
   - 池化核大小为 3x3，步长为 2。
   - 这一层用于减少特征维度和提高模型的空间不变性。
3. **残差块（Residual Block）**：
   - 包括两个 3x3 卷积层，每层后面跟着批归一化（Batch Normalization）和 ReLU 激活函数。
   - 每个卷积层的输出通道数（c）为 ci，ci 是可变的，取决于具体块中的设置。
   - 步长（s）在第一卷积层为 1 或 2，第二卷积层始终为 1，步长为 2 用于降采样。
   - 每个块的最后通过相加操作（+）融合主路径和捷径（shortcut）的输出，然后再应用 ReLU 激活函数。
4. **全局平均池化层（AdaptiveAvgPool）**：
   - 这是全局平均池化层，将特征图缩减为 1x1，减少参数数量，同时保持特征。
5. **重复**：
   - 指示残差块重复的次数，这里是 8 次。



### 实现

```python
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_cqt_spectrogram(file_path, resample_rate=5, segment_duration=20):
    y, sr = librosa.load(file_path, sr=None)
    cqt = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1'), n_bins=168, bins_per_octave=24)
    cqt_amplitude = np.abs(cqt)#标准化
    cqt_resampled = librosa.resample(cqt_amplitude, orig_sr=sr, target_sr=resample_rate, axis=1)#下采样#计算一个片段中的样本数
    samples_per_segment = resample_rate * segment_duration#5*20=100个时间点
    total_segments = int(np.ceil(cqt_resampled.shape[1] / samples_per_segment))
    segments = []
    for i in range(total_segments):#如果末尾超过 CQT 的长度，则用零填充剩余部分，达到指定长度
        start = i * samples_per_segment
        end = start + samples_per_segment
        if end > cqt_resampled.shape[1]:
            padding_length = end - cqt_resampled.shape[1]
            padding = np.zeros((cqt_resampled.shape[0], padding_length))
            segment = np.hstack((cqt_resampled[:, start:cqt_resampled.shape[1]], padding))
        else:
            segment = cqt_resampled[:, start:end]
        segments.append(segment)
    segments = np.array(segments)  # 输出为数组格式
    print("segments shape:", segments.shape)
    return segments

def generate_cqt_spectrogram_Tonic(file_path, resample_rate=5, segment_duration=20):
    y, sr = librosa.load(file_path, sr=None)
    cqt = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1'), n_bins=168, bins_per_octave=24)
    cqt_amplitude = np.abs(cqt)
    cqt_resampled = librosa.resample(cqt_amplitude, orig_sr=sr, target_sr=resample_rate, axis=1)
    if cqt_resampled.shape[1] > 500:
        cqt_resampled = cqt_resampled[:, -500:]#取最后500帧分析主音
    else:
        pass
    samples_per_segment = resample_rate * segment_duration  # 5 * 20 = 100
    total_segments = int(np.ceil(cqt_resampled.shape[1] / samples_per_segment))
    segments_Tonic = []
    for i in range(total_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        if end > cqt_resampled.shape[1]:
            padding_length = end - cqt_resampled.shape[1]
            padding = np.zeros((cqt_resampled.shape[0], padding_length))
            segment = np.hstack((cqt_resampled[:, start:cqt_resampled.shape[1]], padding))
        else:
            segment = cqt_resampled[:, start:end]
        segments_Tonic.append(segment)
    segments_Tonic = np.array(segments_Tonic)
    print("segments_Tonic shape:", segments_Tonic.shape)
    return segments_Tonic

class AudioDataset(Dataset):
    def __init__(self, df, label_column, transform=None):
        self.df = df
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['audio']

        # 根据任务类型决定使用哪个函数生成频谱图
        if self.label_column == 'Tonic':
            spectrogram = generate_cqt_spectrogram_Tonic(audio_path)
        else:
            spectrogram = generate_cqt_spectrogram(audio_path)
        print(f"CQT Spectrogram shape for {self.label_column}:", spectrogram.shape)
        eps = 1e-10  # 避免除以零
        spectrogram = spectrogram / (np.max(spectrogram) + eps)  # 标准化到[0,1]
        spectrogram = np.expand_dims(spectrogram, axis=0)  # 增加通道维度
        label = self.df.iloc[idx][self.label_column]
        return torch.from_numpy(spectrogram).float(), label

def custom_collate_fn(batch):
    spectrograms, labels = zip(*batch)    # 分离频谱图和标签
    spectrograms = [s[0] for s in spectrograms]  # 移除不必要的维度
    spectrograms_padded = pad_sequence(spectrograms, batch_first=True, padding_value=0)# 填充频谱图使它们在时间维度上的长度相同
    print("Batch shape:", spectrograms_padded.shape)# 验证最终形状
    labels = torch.tensor(labels)# 将标签转换为Tensor
    print("Batch shape after padding:", spectrograms_padded.shape)
    return spectrograms_padded, labels

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SingletaskResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(SingletaskResNet, self).__init__()

#由两个大小为 3x3 的卷积层组成，步长（s）为 1 或 2。

        self.in_planes = 64# 修改输入层通道数为1，并移除降采样
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)#第一个卷积层self.conv1被设置为接受单通道输入#64卷积层的输出通道数;bias=False：指示该层不使用偏置参数（bias）;
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        #是stride=1还是stride=2？
        self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 修改步长为1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 是stride=1还是stride=2
        #构建多个残差层_make_layer
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 在网络的末尾使用了全局平均池化，以将特征图的尺寸从任意大小减少到1x1，进而为全连接层（self.fc）提供输入
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def initialize_model(num_classes, device, learning_rate=0.001):

    num_blocks = [2, 2, 2, 2]
    # num_blocks = [2, 2, 2, 2]，残差块2个一层，一共4层，8个残差块。
    model = SingletaskResNet(BasicBlock, num_blocks, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

# 训练模型的函数
def train_model(train_loader, model, criterion, optimizer, device, num_epochs=10):
    model.train()  # 确保模型处于训练模式，不修改epoch的值#进入训练模式，权重参数不可修改
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 确保inputs和labels都在同一个设备上
            # labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# 在验证集上评估模型
def evaluate_model(val_loader, model, device):
    model.eval()  # 设置模型为评估模式
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    return total_correct / total_samples

file_path = './label.csv'
df = pd.read_csv(file_path, encoding='gbk')
audio_dir = os.path.dirname(file_path)

df['audio'] = df['File_Name'].apply(lambda x: os.path.join(audio_dir, 'CNPM_audio', x))#x即为File_Name列中的元素，为文件名，audio_dir路径+'CNPM_audio_old'+x文件名=完整路径
df = df[['audio', 'System', 'Tonic', 'Pattern', 'Type']] # 包含所有列

# 转换过程
transform = transforms.Compose([transforms.ToTensor()])

# 假设train_df和val_df已经定义并包含正确的列
task_types = ['System', 'Tonic', 'Pattern', 'Type']
num_classes = {'System': 12, 'Tonic': 12, 'Pattern': 5, 'Type': 6}

dataloaders = {}
models = {}
criterions = {}
optimizers = {}
accuracies = {}

#单任务不同之处在于每次预测不同类的时候，处理数据之后需要各自再传入模型
# 划分训练集和验证集并创建相应的DataLoader， df 是包含音频路径和标签的 DataFrame
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)  # 以 80-20 的比例划分训练集和验证集，为训练集和验证集创建相应的 DataLoader

for task in task_types:#循环4次，task_types = ['System', 'Tonic', 'Pattern', 'Type']
    # 创建数据集实例
    train_dataset = AudioDataset(train_df, label_column=task, transform=transform)
    val_dataset = AudioDataset(val_df, label_column=task, transform=transform)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    dataloaders[task] = (train_loader, val_loader)

    # 初始化模型
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, criterion, optimizer = initialize_model(num_classes[task], device)

    models[task] = model
    criterions[task] = criterion
    optimizers[task] = optimizer

    epochs = 3
    # 训练和评估模型
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        # 使用已划分的数据集
        train_model(train_loader, model, criterion, optimizer, device, num_epochs=1)  # 为了示例，设置为1个epoch
        accuracies[task] = evaluate_model(val_loader, model, device)
        # accuracies[task] = accuracies[task] / len(val_loader.dataset)
    if task == "System":
        ACC1 = accuracies['System']
        print("ACC1:", ACC1)
    if task == "Tonic":
        ACC2 = accuracies['Tonic']
        print("ACC2:", ACC2)
    if task == "Pattern":
        ACC3 = accuracies['Pattern']
        print("ACC3:", ACC3)
    if task == "Type":
        ACC5 = accuracies['Type']
        print("ACC5:",ACC5)

ACC4 = (accuracies['Tonic'] + accuracies['Pattern']) / 2
print("ACC4:",ACC4)
ACC6 = (accuracies['Tonic'] + accuracies['Pattern'] + accuracies['Type']) / 3
print("ACC6:",ACC6)

print(f"ACC1(System Accuracy): {ACC1:.4f}")
print(f"ACC2(Tonic Accuracy): {ACC2:.4f}")
print(f"ACC3(Pattern Accuracy): {ACC3:.4f}")
print(f"ACC4(Average Tonic and Pattern Accuracy): {ACC4:.4f}")
print(f"ACC5(Type Accuracy): {ACC5:.4f}")
print(f"ACC6(Average of Tonic, Pattern, and Type Accuracy): {ACC6:.4f}")
print("Done!")
```

