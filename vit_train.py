import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm

from datasets import image_dataset

from model_vit import VisionTransformer

class cfg():
    batch_size = 4
    epochs = 20
    lr = 3e-5
    gamma = 0.7
    seed = 42

device = torch.device('cuda:0')

train_list = glob.glob('.\img\*.jp*')
labels = list(set([path.split('\\')[-1].split('_')[0] for path in train_list]))

aug_train = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.RandomResizedCrop(384),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

train_data = image_dataset(train_list, transform=aug_train)
train_loader = DataLoader(dataset = train_data, batch_size=cfg.batch_size, shuffle=True)

model = VisionTransformer(weights_path='./imagenet1k.pth').to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=cfg.gamma)

max_value = 0
max_loss = 100
for epoch in range(cfg.epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    if epoch_accuracy > max_value and epoch_loss < max_loss:
        max_value = epoch_accuracy
        max_loss = epoch_loss
        torch.save(model.state_dict(), './vit_blog.pth')
    print(f"Epoch : {epoch+1} train_loss : {epoch_loss:.4f} train_acc: {epoch_accuracy:.4f}\n")