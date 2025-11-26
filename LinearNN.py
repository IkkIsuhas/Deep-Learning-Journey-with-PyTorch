import torch
import torch.nn as nn

x = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y = torch.tensor([[3],[5],[7],[9]],dtype=torch.float32)

model = nn.Linear(in_features=1,out_features=1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

epochs = 500
for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 50 == 0:
        print(f"Epoch: {epoch} loss: {loss}") 
x_test = torch.tensor([[5.0]])
a = model(x_test)
print(f'prediction: {a}')