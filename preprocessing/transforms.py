from torchvision import transforms


def get_image_transform():
    return transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor()
    ])