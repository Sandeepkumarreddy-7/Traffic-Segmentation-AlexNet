from torch.utils.data import DataLoader

from preprocessing.dataset import BDD100KDataset


def get_dataloader(image_dir, mask_dir, batch_size=16, shuffle=True):

    dataset = BDD100KDataset(
        image_dir=image_dir,
        mask_dir=mask_dir
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader