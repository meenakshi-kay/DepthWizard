import rasterio
import matplotlib.pyplot as plt
import numpy as np

with rasterio.open("data/raw/test_area.tif") as src:
    img = src.read()  # shape: (bands, height, width)
    img = np.transpose(img, (1, 2, 0))  # -> (height, width, bands) for plotting

plt.imshow(img)
plt.title("RGB Preview")
plt.axis("off")
plt.show()

