import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
import numpy as np

def load_band(filepath):
    """
    Loads a single band (like the PAN band) at its native resolution.
    Returns the numpy array and the spatial profile metadata.
    """
    with rasterio.open(filepath) as src:
        data = src.read(1)  
        profile = src.profile
    return data, profile

def load_and_resample_band(filepath, target_shape):
    """
    Loads a MS band (30m) and resamples it to the target shape (15m PAN shape)
    on the fly to save memory. Uses bilinear interpolation.
    """
    with rasterio.open(filepath) as src:
        data = src.read(
            1,
            out_shape=target_shape,
            resampling=Resampling.bilinear
        )
    return data

def save_fused_image(filepath, r_matrix, g_matrix, b_matrix, base_profile):
    """
    Stacks the newly fused R, G, B matrices and writes them to a new .TIF file,
    preserving the geographic coordinates from the PAN band.
    """
    profile = base_profile.copy()
    profile.update(
        count=3,                  
        dtype=r_matrix.dtype,      
        photometric='RGB',
        nodata = 0          
    )

    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(r_matrix, 1)
        dst.write(g_matrix, 2)
        dst.write(b_matrix, 3)
        
    print(f"Successfully saved fused image to: {filepath}")

def load_band_from_bytes(uploaded_file):
    """Reads a band from an uploaded file-like object."""
    with MemoryFile(uploaded_file) as memfile:
        with memfile.open() as src:
            data = src.read(1)
            profile = src.profile
    return data, profile

def load_and_resample_band_from_bytes(uploaded_file, target_shape):
    """Reads and resamples a band from an uploaded file-like object."""
    with MemoryFile(uploaded_file) as memfile:
        with memfile.open() as src:
            data = src.read(1, out_shape=target_shape, resampling=Resampling.bilinear)
    return data