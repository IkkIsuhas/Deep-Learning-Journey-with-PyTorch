import torch
import torch.nn as nn

x = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[3.0],[5.0],[7.0],[9.0]])

class simpleNN(nn.Module):
    def __init__(self):
        super(simpleNN,self).__init__()
        self.layer1 = nn.Linear(1,4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4,1)

    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
    
model = simpleNN()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 500

for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 50 == 0:
        print(f'Epochs: {epoch} Loss: {loss}')

x_test = torch.tensor([[5.0]])

model.eval()
with torch.no_grad():
    predication = model(x_test)
    print(f'Predication: {predication.item():.3f}')