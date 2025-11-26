import torch

x = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[3.0],[5.0],[7.0],[9.0]])

w = torch.rand(1,1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

lr = 0.01
epochs = 200

for epoch in range(epochs):
    y_pred = x @ w+b
    loss = ((y_pred - y)**2).mean()
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 20 == 0:
        print(f'Epoch: {epoch} loss: {loss}')

x_test = torch.tensor([[5.0]])
with torch.no_grad():
    print('Prediction for x = 5: ', x_test @ w + b)