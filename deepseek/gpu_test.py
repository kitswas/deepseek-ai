import torch

# Find why the GPU is not available
print(torch.cuda.is_available())

# Find the number of GPUs available
print(torch.cuda.device_count())

# Find the name of the GPU
print(torch.cuda.get_device_name(0))

x = torch.rand(5, 3)
x.cuda()
print(x)
