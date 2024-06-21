import torch
import pandas as pd
from nnmodel import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

inputWidth = 7
input_model_filename = 'models/cpu_model_xtodo00_2000_epochs_20.pt'

model = NeuralNetwork(inputWidth)
model.load_state_dict(torch.load(input_model_filename))

# x = torch.rand(1, 5, device=device)
# print(x)
x = torch.FloatTensor([[5,6,3,10,-4,26,24]])
# print(x)

logits = model(x)
probs = torch.nn.Softmax(dim=1)(logits)
pred = probs.argmax(1)

print(f"{logits=}")
print(f"{probs=}")
print(f"{pred=}")
print(f"Win probability: {probs[0][1]}")