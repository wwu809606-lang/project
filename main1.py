# main.py

import wuwenjun as wwj  # 自定义的深度学习框架
import numpy as np

# ========= 1. 构造训练数据 =========
# AND 逻辑表
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

Y = np.array([[0.0], [0.0], [0.0], [1.0]])  # 期望输出


# ========= 2. 定义 2-2-1 网络结构 =========
model = wwj.Sequential(
    wwj.Linear(2, 2),     # 输入 2 → 隐藏层 2
    wwj.Sigmoid(),        # 激活函数
    wwj.Linear(2, 1),     # 隐藏 2 → 输出 1
    wwj.Sigmoid()         # 输出激活
)

# ========= 3. 定义损失函数 =========
loss_fn = wwj.mse_loss

# ========= 4. 指定优化器 =========
optimizer = wwj.SGD(model.parameters(), lr=0.1)

# ========= 5. 训练循环 =========
EPOCHS = 2000

for epoch in range(EPOCHS):

    total_loss = 0

    for x, y in zip(X, Y):
        # 每次喂一个样本（手搓框架先支持单样本就够了）

        x = wwj.Tensor(x, requires_grad=False)  # 输入不需要梯度
        y = wwj.Tensor(y, requires_grad=False)

        # Forward
        y_pred = model(x)

        # Loss
        loss = loss_fn(y_pred, y)
        total_loss += loss.data

        # Backward
        loss.backward()

        # 更新参数
        optimizer.step()

        # 清空梯度，避免累积
        optimizer.zero_grad()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss = {total_loss}")

print("\n训练结束！")
print("测试模型：")

for x in X:
    x = wwj.Tensor(x)
    print(x.data, "→", model(x).data)
