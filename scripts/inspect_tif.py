import rasterio

FILE = "data/raw/test_area.tif"

with rasterio.open(FILE) as src:

    print("========== IMAGE INFORMATION ==========")

    print("Width:", src.width)
    print("Height:", src.height)
    print("Number of bands:", src.count)

    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Transform:", src.transform)

    print("Data type:", src.dtypes)

    print("\nBand descriptions:")
    for i, description in enumerate(src.descriptions, start=1):
        print(f"Band {i}: {description}")

