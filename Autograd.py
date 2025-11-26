import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor([5.0], requires_grad=True)
y = 3*(x**3)+2*(x**2)+x
print(y)
print(y.backward())
print(x.grad)