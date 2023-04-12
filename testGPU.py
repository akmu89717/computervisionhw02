import numpy as np
from matplotlib import pyplot as plt
np.random.seed(7) # 用來確保每次生成的隨機資料是一樣的，否則訓練結果無法比較
x = np.random.rand(100, 1)
y = 2 + 5 * x + .2 * np.random.randn(100, 1) # randn 的 n 為 normal distribution
idx = np.arange(100)
np.random.shuffle(idx) # 打散索引
train_idx = idx[:80] # 取前 80 筆為訓練資料
val_idx = idx[80:] #取後 20 筆為驗證資料
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]
# plt.scatter(x_train, y_train)
# plt.show()
# # 隨機給定數值，初始化 a, b 的值
# np.random.seed(7)
# a = np.random.randn(1)
# b = np.random.randn(1)
# # 設定學習率
# lr = 1e-1
# # 設定 epochs
# epochs = 500
# for epoch in range(100): 
#     # 計算模型的預測 
#     yhat = a + b * x_train
#     # 用預測和標記來計算 error 
#     error = (y_train - yhat)
#     # 用 error 來計算 loss
#     loss = (error ** 2).mean()
    
#     # 計算兩個參數的梯度
#     a_grad = -2 * error.mean()
#     b_grad = -2 * (x_train * error).mean()
    
#     # 用梯度和學習率來更新參數
#     a = a - lr * a_grad
#     b = b - lr * b_grad

# print(a, b)
# from sklearn.linear_model import LinearRegression
# LR = LinearRegression()
# LR.fit(x_train, y_train)
# print(LR.intercept_, LR.coef_[0])
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_train_tensor = torch.from_numpy(x_train).to(device)
y_train_tensor = torch.from_numpy(y_train).to(device)
print(type(x_train), type(x_train_tensor), x_train_tensor.type())
# a = torch.randn(1, requires_grad=True)
# b = torch.randn(1, requires_grad=True)
a = torch.randn(1).to(device)
b = torch.randn(1).to(device)
a.requires_grad_()
b.requires_grad_()
lr = 1e-1
epochs = 500
torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, device=device)
b = torch.randn(1, requires_grad=True, device=device)
for epoch in range(epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()
    # 不用再自己計算梯度了 
    # a_grad = -2 * error.mean()
    # b_grad = -2 * (x_tensor * error).mean()
    
    # 從 loss 做 backward 來幫我們取得梯度
    loss.backward()
    
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
    
    # 清除梯度
    a.grad.zero_()
    b.grad.zero_()
    
print(a, b)
torch.manual_seed(7)
a = torch.randn(1, requires_grad=True, device=device)
b = torch.randn(1, requires_grad=True, device=device)
yhat = a + b * x_train_tensor
error = y_train_tensor - yhat
loss = (error ** 2).mean()
