import os

from PIL import Image

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from torch.utils.data import Dataset

from preprocessing.transforms import get_image_transform


class BDD100KDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        if transform is None:
            self.transform = get_image_transform()
        else:
            self.transform = transform

        self.image_list = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        image_name = self.image_list[index]

        image_path = os.path.join(self.image_dir, image_name)

        mask_name = image_name.replace(".jpg", "_train_id.png")

        mask_path = os.path.join(self.mask_dir, mask_name)

        # Read image
        image = Image.open(image_path).convert("RGB")

        # Read mask
        mask = Image.open(mask_path)

        # Transform image
        image = self.transform(image)

        # Resize mask
        mask = TF.resize(
            mask,
            size=(256, 512),
            interpolation=InterpolationMode.NEAREST
        )

        # Convert mask to tensor
        mask = TF.pil_to_tensor(mask).squeeze(0).long()

        return image, mask