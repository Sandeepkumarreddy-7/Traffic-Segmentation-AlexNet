import numpy as np

# BDD100K / Cityscapes-like color palette
COLOR_MAP = np.array([
    [128, 64, 128],    # Road
    [244, 35, 232],    # Sidewalk
    [70, 70, 70],      # Building
    [102, 102, 156],   # Wall
    [190, 153, 153],   # Fence
    [153, 153, 153],   # Pole
    [250, 170, 30],    # Traffic Light
    [220, 220, 0],     # Traffic Sign
    [107, 142, 35],    # Vegetation
    [152, 251, 152],   # Terrain
    [70, 130, 180],    # Sky
    [220, 20, 60],     # Person
    [255, 0, 0],       # Rider
    [0, 0, 142],       # Car
    [0, 0, 70],        # Truck
    [0, 60, 100],      # Bus
    [0, 80, 100],      # Train
    [0, 0, 230],       # Motorcycle
    [119, 11, 32]      # Bicycle
], dtype=np.uint8)


def colorize_mask(mask):
    """
    Convert class IDs to RGB colors.
    """

    height, width = mask.shape

    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for cls in range(len(COLOR_MAP)):
        color_mask[mask == cls] = COLOR_MAP[cls]

    return color_mask