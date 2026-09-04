import rasterio
import numpy as np
from PIL import Image

INPUT = "data/raw/test_area.tif"
OUTPUT = "data/processed/rgb.png"

with rasterio.open(INPUT) as src:

    # Read first three bands
    image = src.read([1, 2, 3])

    # Rasterio gives:
    # (bands, height, width)
    #
    # We need:
    # (height, width, bands)

    image = np.transpose(image, (1, 2, 0))

    # Make sure values are suitable for PNG
    image = np.clip(image, 0, 255).astype(np.uint8)

    Image.fromarray(image).save(OUTPUT)

print("Saved:", OUTPUT)
print("Image shape:", image.shape)

