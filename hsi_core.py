import numpy as np

def rgb_to_hsi(r, g, b):
    """
    Converts Red, Green, and Blue matrices to Hue, Saturation, and Intensity.
    Assumes r, g, b are normalized arrays in the range [0.0, 1.0].
    """
    eps = 1e-8
    
    # 1. Calculate Intensity (I)
    i = (r + g + b) / 3.0
    
    # 2. Calculate Saturation (S)
    min_rgb = np.minimum(np.minimum(r, g), b)
    s = 1.0 - (3.0 / (r + g + b + eps)) * min_rgb
    
    # 3. Calculate Hue (H) in radians
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b) * (g - b)) + eps
    
    # Use np.clip to prevent arccos floating point domain errors (-1.0 to 1.0)
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    
    h = np.copy(theta)
    
    # If Blue is greater than Green, Hue is 360 degrees (2*pi) minus theta
    h_mask = b > g
    h[h_mask] = 2 * np.pi - h[h_mask]
    
    # Fill any NaNs with 0 (might occur in absolute black pixels despite eps)
    h = np.nan_to_num(h)
    s = np.nan_to_num(s)
    
    return h, s, i

def histogram_match(source, reference):
    """
    Adjusts the source matrix (PAN band) to match the mean and 
    standard deviation of the reference matrix (Intensity band).
    """
    src_mean = np.mean(source)
    src_std = np.std(source)
    
    ref_mean = np.mean(reference)
    ref_std = np.std(reference)
    
    matched = (source - src_mean) * (ref_std / (src_std + 1e-8)) + ref_mean
    
    return np.clip(matched, 0.0, 1.0)

def hsi_to_rgb(h, s, i):
    """
    Converts Hue, Saturation, and Intensity back to Red, Green, and Blue.
    Expects Hue in radians.
    """
    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)
    
    idx1 = (h >= 0) & (h < 2 * np.pi / 3)
    b[idx1] = i[idx1] * (1 - s[idx1])
    r[idx1] = i[idx1] * (1 + (s[idx1] * np.cos(h[idx1])) / np.cos(np.pi / 3 - h[idx1]))
    g[idx1] = 3 * i[idx1] - (r[idx1] + b[idx1])
    
    idx2 = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
    h_mod2 = h[idx2] - (2 * np.pi / 3)
    r[idx2] = i[idx2] * (1 - s[idx2])
    g[idx2] = i[idx2] * (1 + (s[idx2] * np.cos(h_mod2)) / np.cos(np.pi / 3 - h_mod2))
    b[idx2] = 3 * i[idx2] - (r[idx2] + g[idx2])
    
    idx3 = (h >= 4 * np.pi / 3) & (h <= 2 * np.pi)
    h_mod3 = h[idx3] - (4 * np.pi / 3)
    g[idx3] = i[idx3] * (1 - s[idx3])
    b[idx3] = i[idx3] * (1 + (s[idx3] * np.cos(h_mod3)) / np.cos(np.pi / 3 - h_mod3))
    r[idx3] = 3 * i[idx3] - (g[idx3] + b[idx3])
    
    return np.clip(r, 0.0, 1.0), np.clip(g, 0.0, 1.0), np.clip(b, 0.0, 1.0)