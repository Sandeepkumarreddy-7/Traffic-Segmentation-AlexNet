from preprocessing.dataset import BDD100KDataset

dataset = BDD100KDataset(
    image_dir="dataset/10k/train",
    mask_dir="dataset/labels/train"
)

print("Total Images:", len(dataset))

# Load first sample
image, mask = dataset[0]

print(type(image))
print(type(mask))

print(image.size)
print(mask.size)