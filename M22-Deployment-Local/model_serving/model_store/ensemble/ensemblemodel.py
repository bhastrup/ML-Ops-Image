
import json

import torch
from torch import nn
from torchvision import models
from torchvision import datasets, transforms as T
from PIL import Image

with open('index_to_name.json') as json_file:
    index_to_name = json.load(json_file)

backbones = ['resnet18', 'resnext101_32x8d']
model_list = nn.ModuleList([getattr(models, bb)(pretrained=True) for bb in backbones])  

class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbones = nn.ModuleList([getattr(models, bb)(pretrained=True) for bb in backbones])

    def forward(self, x: torch.Tensor):
        res = [bb(x) for bb in self.backbones]
        
        # todo: combine the output in res
        # final_pred = res[0] + res[1]
        return res

my_ens_model = EnsembleModel()
my_ens_model()

########################################################################
########################################################################


# Load the image
image = Image.open('blade_runner.jpg')
print(image.format)
print(image.mode)
print(image.size)
# image.show()


normalize = T.Normalize(mean=[2, 2, 2],
                         std=[2, 2, 2])
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

image_ready = transform(image).unsqueeze(0)
image_ready.shape


model_a = model_list[0]
model_b = model_list[1]

k=3
logits = model_b(image_ready)
softmax = nn.Softmax(dim=1)
probs = softmax(logits)
top3 = torch.topk(probs, k=k)

prob1 = float(top3.values.squeeze()[0])
name1 = index_to_name[str(int(top3.indices.squeeze()[0]))][1]

results = {}
for kk in range(0, k):
    prob = float(top3.values.squeeze()[kk])
    name = index_to_name[str(int(top3.indices.squeeze()[kk]))][1]
    results[kk] = {'name': name, 'prob': prob}

print(results)