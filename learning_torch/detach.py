import torch

a = torch.tensor([1, 2, 3.], requires_grad=True)
print(a.grad)
out = a.sigmoid()
print(out)

#######################
# 差在這裡
# 添加detach(),c的requires_grad为False
c = out.detach()
print(c)
#######################

out.sum().backward()
print(a.grad)
