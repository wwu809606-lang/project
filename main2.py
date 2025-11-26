import wuwenjun as wwj
from wuwenjun.tensor import Tensor

# ======== 构建一个 4 输入特征的一隐藏层神经网络 ========
# 网络结构：
# 输入层(4) → 隐藏层(3) → 输出层(1)
# 你可以自由改隐藏层维度
model = wwj.Sequential(
    wwj.Linear(4, 3),   # 输入 4 维 → 隐藏层 3 维
    wwj.Sigmoid(),
    wwj.Linear(3, 1),   # 隐藏层 3 维 → 输出 1 维
    wwj.Sigmoid()       # 输出概率
)

# ======== 定义训练数据 ========
# 每个样本：4 个特征 + 1 个标签（0 or 1）
data = [
    (Tensor([0, 0, 0, 0]), Tensor([0])),
    (Tensor([0, 1, 0, 1]), Tensor([0])),
    (Tensor([1, 0, 1, 0]), Tensor([1])),
    (Tensor([1, 1, 1, 1]), Tensor([1])),
]

# ======== 定义损失函数与优化器 ========
loss_fn = wwj.mse_loss
optimizer = wwj.SGD(model.parameters(), lr=0.1)

# ======== 开始训练 ========
epochs = 2000
for epoch in range(epochs):
    total_loss = 0.0
    for x, y in data:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss = {total_loss}")

print("\n训练结束！\n测试模型：")
for x, _ in data:
    print(f"{x.data} → {model(x).data}")
