import torch


def calculate_miou(predictions, masks, num_classes, ignore_index=255):
    """
    Calculate Mean Intersection over Union (mIoU)
    """

    iou_list = []

    for cls in range(num_classes):

        valid_mask = masks != ignore_index

        pred = (predictions == cls) & valid_mask
        target = (masks == cls) & valid_mask

        intersection = (pred & target).sum().item()

        union = (pred | target).sum().item()

        if union == 0:
            continue

        iou = intersection / union

        iou_list.append(iou)

    if len(iou_list) == 0:
        return 0.0

    return sum(iou_list) / len(iou_list)