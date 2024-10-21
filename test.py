import os
import torch
import sys

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'

from torchvision.models import resnet152
from torch import nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
print()

print(torch.cuda.get_device_name())
print()

print(sys.path)
print()

print(os.listdir("/opt/rocm"))
print()

x = torch.tensor([1., 2.]).to(DEVICE)

print(x + x)
print()


model = resnet152(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE).eval()

inp = torch.randn((1, 3, 224, 224)).to(DEVICE)

with torch.no_grad():
    logits = model(inp)
    pred = torch.argmax(logits, dim=1).cpu().item()
    print(logits, pred)