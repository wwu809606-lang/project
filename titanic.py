import pandas as pd
import numpy as np
import wuwenjun as wwj

from wuwenjun.tensor import Tensor

# ======== 1. 读取并预处理 Titanic 数据 ========
df = pd.read_csv('titanic.csv')

# 选择我们要用的四个特征
features = ['Pclass', 'Sex', 'Age', 'Fare']
label = 'Survived'

# 性别编码：female → 1, male → 0
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 年龄 Age 有缺失值，用平均值填补
df['Age'] = df['Age'].fillna(df['Age'].mean())

# 取出特征和标签
X = df[features].values.astype(float)
y = df[label].values.astype(float).reshape(-1, 1)

# 数据归一化（提升训练效果）
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 转换成 Tensor 数据集（你的框架需要逐样本输入）
dataset = [(Tensor(x), Tensor(target)) for x, target in zip(X, y)]
import random
random.shuffle(dataset)  # ← 这行写在这里！
print(f"数据集加载成功，共 {len(dataset)} 个样本，示例：")
print(dataset[0])


# ======== 2. 构建网络结构 ========
# 输入 4 维 → 隐藏层 3 维 → 输出 1 维
model = wwj.Sequential(
    wwj.Linear(4, 3),
    wwj.Sigmoid(),
    wwj.Linear(3, 1),
    wwj.Sigmoid()
)

# ======== 3. 定义损失函数与优化器 ========
loss_fn = wwj.mse_loss
optimizer = wwj.SGD(model.parameters(), lr=0.05)



# ======== 4. 划分训练集和测试集 ========
# 简单按 8:2 划分
split = int(0.8 * len(dataset))
train_data = dataset[:split]
test_data = dataset[split:]

print(f"训练样本数: {len(train_data)}, 测试样本数: {len(test_data)}")

# ======== 5. 开始训练 ========
epochs = 2000
for epoch in range(epochs):
    total_loss = 0.0
    for x, target in dataset:
        pred = model(x)             # forward
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()             # backward
        optimizer.step()            # update weights

        total_loss += loss.data

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss = {total_loss / len(dataset)}")


# ======== 6. 测试模型效果 ========
print("\n训练结束！测试前 10 个样本：\n")
for i in range(10):
    x, target = dataset[i]
    pred = model(x).data
    print(f"真实值 = {target.data}, 预测值 = {pred}")

# ========= 7.计算整体精度 =========
correct = 0
total = len(test_data)

for x, y_true in test_data:
    y_pred = model(x)         # 前向计算
    pred_label = 1 if y_pred.data > 0.5 else 0   # 概率阈值0.5分类
    true_label = int(y_true.data[0])

    if pred_label == true_label:
        correct += 1

accuracy = correct / total
print(f"\n模型准确率 = {accuracy * 100:.2f}%")
