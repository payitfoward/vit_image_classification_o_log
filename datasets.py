from torch.utils.data import Dataset
from PIL import Image

class image_dataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.label2idx = {
            'grand': 0,
            'modern': 1,
            'antic': 2,
            'vintage': 3,
            'cozy': 4,
            'gorgeous': 5,
            'retro': 6,
            'scenery': 7,
            'classic': 8,
            'calm': 9
        }

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split('\\')[-1].split('_')[0]
        label = self.label2idx[label]

        return img_transformed, label