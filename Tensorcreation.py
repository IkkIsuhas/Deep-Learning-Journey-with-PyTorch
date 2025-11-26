import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.rand(2,2).to(device)
y = torch.ones(2,2).to(device)

print(x)
print(x*y)