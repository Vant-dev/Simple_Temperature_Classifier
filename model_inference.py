import torch
import torch.nn as nn
from model import TemperatureClassifier
from safetensors.torch import load_file

dic =  load_file('temperature_classifier.safetensors')

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = TemperatureClassifier(1, 100, 2).to(device)
model.load_state_dict(dic)

print(dic)


print("Введите число: ")
x = float(input())

x_tensor = torch.tensor([[x]], dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    logits = model(x_tensor)
    print(logits)
    out = torch.argmax(logits, dim=1)
    
print(out.item())