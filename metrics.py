import numpy as np


def calculate_metrics_band(orig_band, fused_band):
    """
    Calculates ERGAS components and Correlation Coefficient (CC)
    for a single band mathematically to avoid standard library memory leaks.
    Used for one-shot calculation on small arrays (e.g. preview tiles).
    """
    diff = np.subtract(orig_band, fused_band, dtype=np.float32)
    np.square(diff, out=diff)

    mse = np.mean(diff, dtype=np.float64)
    del diff

    rmse = np.sqrt(mse)
    mean_orig = np.mean(orig_band, dtype=np.float64)

    ergas_ratio_sq = (rmse / (mean_orig + 1e-8)) ** 2

    orig_centered = orig_band - mean_orig
    fused_centered = fused_band - np.mean(fused_band, dtype=np.float64)

    covar = orig_centered * fused_centered
    numerator = np.sum(covar, dtype=np.float64)
    del covar

    np.square(orig_centered, out=orig_centered)
    np.square(fused_centered, out=fused_centered)

    den_orig = np.sum(orig_centered, dtype=np.float64)
    den_fused = np.sum(fused_centered, dtype=np.float64)

    cc = numerator / np.sqrt(den_orig * den_fused + 1e-16)

    return ergas_ratio_sq, cc


# ---------------------------------------------------------------------------
# Streaming (online) metrics — accumulate block-by-block, finalize at the end.
# This lets you compute EXACT full-resolution ERGAS/CC with O(1) extra memory.
# ---------------------------------------------------------------------------

def create_accumulator():
    """
    Returns a fresh per-band accumulator dict.
    Create one for R, G, and B before starting the block loop.
    """
    return {
        "n":              np.int64(0),
        "sum_sq_diff":    np.float64(0.0),   # Σ (orig - fused)²  → MSE
        "sum_orig":       np.float64(0.0),   # Σ orig             → mean_orig
        "sum_fused":      np.float64(0.0),   # Σ fused            → mean_fused
        "sum_orig_fused": np.float64(0.0),   # Σ orig*fused       → E[XY]
        "sum_orig_sq":    np.float64(0.0),   # Σ orig²            → E[X²]
        "sum_fused_sq":   np.float64(0.0),   # Σ fused²           → E[Y²]
    }


def accumulate_block(accum: dict, orig_block: np.ndarray, fused_block: np.ndarray):
    """
    Feed one block of original and fused pixel values into the accumulator.
    Both arrays must be in the same scale (e.g. both float 0-1 or both uint16).
    No temporaries larger than a single block are allocated.
    """
    orig  = orig_block.ravel().astype(np.float64)
    fused = fused_block.ravel().astype(np.float64)

    diff = orig - fused

    accum["n"]              += orig.size
    accum["sum_sq_diff"]    += np.dot(diff, diff)          # avoids temporary array
    accum["sum_orig"]       += orig.sum()
    accum["sum_fused"]      += fused.sum()
    accum["sum_orig_fused"] += np.dot(orig, fused)
    accum["sum_orig_sq"]    += np.dot(orig, orig)
    accum["sum_fused_sq"]   += np.dot(fused, fused)


def finalize_metrics(accum: dict):
    """
    Compute ERGAS ratio² and CC from the accumulated statistics.
    Equivalent to calculate_metrics_band() but over the full image.

    Returns
    -------
    ergas_ratio_sq : float   (pass 3 of these to the ERGAS formula in main)
    cc             : float   (Pearson correlation coefficient)
    """
    n          = accum["n"]
    mean_orig  = accum["sum_orig"]  / n
    mean_fused = accum["sum_fused"] / n

    mse  = accum["sum_sq_diff"] / n
    rmse = np.sqrt(mse)
    ergas_ratio_sq = (rmse / (mean_orig + 1e-8)) ** 2

    # Pearson CC via the computational formula:
    #   Cov(X,Y)  = E[XY] - E[X]·E[Y]
    #   Var(X)    = E[X²] - E[X]²
    e_xy   = accum["sum_orig_fused"] / n
    e_x2   = accum["sum_orig_sq"]    / n
    e_y2   = accum["sum_fused_sq"]   / n

    cov      = e_xy  - mean_orig * mean_fused
    var_orig  = e_x2 - mean_orig  ** 2
    var_fused = e_y2 - mean_fused ** 2

    cc = cov / (np.sqrt(max(var_orig * var_fused, 0.0)) + 1e-8)

    return ergas_ratio_sq, float(cc)