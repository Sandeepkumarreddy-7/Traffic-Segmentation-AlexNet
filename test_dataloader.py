from preprocessing.dataloader import get_dataloader

train_loader = get_dataloader(
    image_dir="dataset/10k/train",
    mask_dir="dataset/labels/train",
    batch_size=16,
    shuffle=True
)

print("Total Batches:", len(train_loader))

for images, masks in train_loader:

    print("Images Shape :", images.shape)
    print("Masks Shape  :", masks.shape)

    break