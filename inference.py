import glob

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


from tqdm.notebook import tqdm

from model_vit import VisionTransformer
from datasets import image_dataset

class cfg():
    seed = 42
    weight_path = './vit_blog.pth'

device = torch.device('cuda:0')

model = VisionTransformer(weights_path=cfg.weight_path).to(device)

inference_list = glob.glob('.\img\*.jp*')

inference_trans = transforms.Compose(
    [
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
    ]
)

inference_dataset = image_dataset(inference_list, transform=inference_trans)
inference_loader = DataLoader(dataset = inference_dataset, batch_size=1, shuffle=False)

pred_list = []
with torch.no_grad():
    for data, label in tqdm(inference_loader):
        data = data.to(device)
        label = label.to(device)

        val_output = model(data).argmax(dim=1)
        pred_list.append(val_output)

print(pred_list)