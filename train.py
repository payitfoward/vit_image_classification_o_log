import glob

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


from tqdm.notebook import tqdm

from model import VisionTransformer
from datasets import image_dataset

class cfg():
    batch_size = 4
    epochs = 20
    lr = 3e-5
    gamma = 0.7
    seed = 42
    weight_path = './L_16_imagenet1k.pth'

device = torch.device('cuda:0')

train_list = glob.glob('.\img\*.jp*')
label_list = [path.split('\\')[-1].split('_')[0] for path in train_list]

label = list(set([path.split('\\')[-1].split('_')[0] for path in train_list]))

train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=label_list, random_state=cfg.seed)

train_trans = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.RandomResizedCrop(384),
            transforms.RandomAffine(
                degrees=15, translate=(0.2, 0.2),
                scale=(0.8, 1.2), shear=15, interpolation=Image.BILINEAR)
        ]),
        transforms.ToTensor(),
    ]
)

val_trans = transforms.Compose(
    [
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
    ]
)

train_dataset = image_dataset(train_list, transform=train_trans)
valid_dataset = image_dataset(valid_list, transform=val_trans)
train_loader = DataLoader(dataset = train_dataset, batch_size=cfg.batch_size, shuffle=True)
valid_loader = DataLoader(dataset = train_dataset, batch_size=cfg.batch_size, shuffle=True)

model = VisionTransformer(weights_path=cfg.weight_path).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=cfg.gamma)

max_value = 0
max_loss = 100
for epoch in range(cfg.epochs):
    train_loss = 0
    train_acc = 0
    
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        train_acc += acc / len(train_loader)
        train_loss += loss / len(train_loader)

    with torch.no_grad():
        val_acc = 0
        val_loss = 0
        for data, label in tqdm(valid_loader):
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            val_acc += acc / len(valid_loader)
            val_loss += val_loss / len(valid_loader)

    if train_acc > max_value and train_loss < max_loss:
        max_value = train_acc
        max_loss = train_loss
        torch.save(model.state_dict(), './vit_blog.pth')
    print(f"Epoch: {epoch+1} train_loss: {train_loss:.3f} train_acc: {train_acc:.3f} valid_loss: {val_loss:.3f} valid_acc: {val_acc:.3f} \n")