# train.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import cnn_model  # 导入自定义的CNN模型

# 读取数据
data = pd.read_csv('train.csv')
labels = data.iloc[:, 0].values
features = data.iloc[:, 1:].values.reshape(-1, 1, 28, 28).astype('float32')

# 划分训练集和验证集
features_train, features_val, labels_train, labels_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# 创建 DataLoader
train_dataset = TensorDataset(torch.tensor(features_train), torch.tensor(labels_train))
train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True)

# 初始化模型、损失函数和优化器
model = cnn_model.CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs =1000
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0  # 初始化总损失
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()  # 累加每个批次的损失
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)  # 计算平均损失
    print(f'Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}')
    if epoch%40==0:
        torch.save(model.state_dict(), f'cnn_model{epoch}.pth')
