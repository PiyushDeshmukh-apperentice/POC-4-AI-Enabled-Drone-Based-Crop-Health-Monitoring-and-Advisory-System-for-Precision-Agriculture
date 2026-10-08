import numpy as np
import rasterio

# 1. Define your input file paths
# Replace these with the actual paths to your single-band TIF files
red_path = r"D:\Projects\Plant Leaf Disease\WeedyRice-RGBMS-DB\Multispectral\DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m_R.TIF"
green_path = r"D:\Projects\Plant Leaf Disease\WeedyRice-RGBMS-DB\Multispectral\DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m_G.TIF"
nir_path = r"D:\Projects\Plant Leaf Disease\WeedyRice-RGBMS-DB\Multispectral\DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m_NIR.TIF"
output_path = "false_color_rgb.tif"


def percent_stretch(band, percent=2):
    """Applies a min-max contrast stretch to make the image visible."""
    # Exclude nodata values (usually 0) from contrast calculation
    valid_pixels = band[band > 0]
    if valid_pixels.size == 0:
        return np.zeros_like(band, dtype=np.uint8)

    low = np.percentile(valid_pixels, percent)
    high = np.percentile(valid_pixels, 100 - percent)

    # Clip and normalize to 0-255 (8-bit RGB)
    stretched = np.clip(band, low, high)
    stretched = ((stretched - low) / (high - low) * 255).astype(np.uint8)
    return stretched


# 2. Read the bands and their metadata
with rasterio.open(green_path) as src_g:
    green_band = src_g.read(1)
    meta = src_g.meta.copy()  # Save geospatial metadata (CRS, transform)

with rasterio.open(red_path) as src_r:
    red_band = src_r.read(1)

with rasterio.open(nir_path) as src_n:
    nir_band = src_n.read(1)

# 3. Apply contrast stretching to each channel
# For a standard False-Color Infrared (CIR) composite:
# Channel 1 (Red) = NIR Band
# Channel 2 (Green) = Red Band
# Channel 3 (Blue) = Green Band
r_channel = percent_stretch(nir_band)
g_channel = percent_stretch(red_band)
b_channel = percent_stretch(green_band)

# 4. Stack bands into a 3D array (3, height, width)
rgb_composite = np.stack([r_channel, g_channel, b_channel], axis=0)

# 5. Update metadata for the new 3-band 8-bit output image
meta.update(
    {
        "count": 3,  # 3 channels (RGB)
        "dtype": "uint8",  # Stretched to 8-bit
        "nodata": 0,
    }
)

# 6. Write the final RGB image to disk
with rasterio.open(output_path, "w", **meta) as dst:
    dst.write(rgb_composite)

print(f"Successfully created false-color RGB image at: {output_path}")
