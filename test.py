import os
import torch
import sys

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'

print(torch.cuda.get_device_name())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print()

print(sys.path)
print()

print(os.listdir("/opt/rocm"))
print()

x = torch.tensor([1]).to(device)
print(x + x)