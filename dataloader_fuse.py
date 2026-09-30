import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

class TriplePairedDataset(Dataset):
    def __init__(self, rgb_dir, nir_up_dir, nir_gt_dir, transform=None, crop_size=None, training=True):
        self.rgb_dir = rgb_dir
        self.nir_up_dir = nir_up_dir
        self.nir_gt_dir = nir_gt_dir
        self.transform = transform
        self.crop_size = crop_size
        self.training = training

        self.rgb_images = sorted([f for f in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, f))])
        self.nir_up_images = sorted([f for f in os.listdir(nir_up_dir) if os.path.isfile(os.path.join(nir_up_dir, f))])
        self.nir_gt_images = sorted([f for f in os.listdir(nir_gt_dir) if os.path.isfile(os.path.join(nir_gt_dir, f))])

        assert len(self.rgb_images) == len(self.nir_up_images) == len(self.nir_gt_images), \
            "Mismatch in number of RGB, NIR_UP, and NIR_GT images"

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_images[idx])
        nir_up_path = os.path.join(self.nir_up_dir, self.nir_up_images[idx])
        nir_gt_path = os.path.join(self.nir_gt_dir, self.nir_gt_images[idx])

        rgb = Image.open(rgb_path).convert('RGB')
        nir_up = Image.open(nir_up_path).convert('L')
        nir_gt = Image.open(nir_gt_path).convert('L')

        if self.transform:
            rgb = self.transform(rgb)
            nir_up = self.transform(nir_up)
            nir_gt = self.transform(nir_gt)

        if self.training and self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=(self.crop_size, self.crop_size))
            rgb = transforms.functional.crop(rgb, i, j, h, w)
            nir_up = transforms.functional.crop(nir_up, i, j, h, w)
            nir_gt = transforms.functional.crop(nir_gt, i, j, h, w)

        return rgb, nir_up, nir_gt

def create_dataloaders(train_rgb, train_nir_up, train_nir_gt,
                       test_rgb, test_nir_up, test_nir_gt,
                       crop_size=256, batch_size=4):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_loader = None
    test_loader = None

    if train_rgb and train_nir_up and train_nir_gt:
        train_dataset = TriplePairedDataset(train_rgb, train_nir_up, train_nir_gt,
                                            transform=transform, crop_size=crop_size, training=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    if test_rgb and test_nir_up and test_nir_gt:
        test_dataset = TriplePairedDataset(test_rgb, test_nir_up, test_nir_gt,
                                           transform=transform, training=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, test_loader
