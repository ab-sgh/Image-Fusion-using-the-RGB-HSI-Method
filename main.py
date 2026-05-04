import streamlit as st
import numpy as np
import time
import os
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

# Local module imports
from hsi_core import rgb_to_hsi, histogram_match, hsi_to_rgb
from metrics import create_accumulator, accumulate_block, finalize_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_contrast_stretch(band_matrix):
    """2 % – 98 % contrast stretch; zero pixels (nodata) stay zero."""
    valid_pixels = band_matrix[band_matrix > 0]
    if len(valid_pixels) == 0:
        return band_matrix
    p2, p98 = np.percentile(valid_pixels, (2, 98))
    if p98 <= p2:
        return band_matrix
    stretched = np.clip((band_matrix - p2) / (p98 - p2), 0.0, 1.0)
    stretched[band_matrix == 0] = 0.0
    return stretched

def generate_preview(filepath, target_size=(1000, 1000)):
    """Downsampled read — only used for the difference map."""
    with rasterio.open(filepath) as src:
        data = src.read(
            out_shape=(src.count, target_size[0], target_size[1]),
            resampling=Resampling.bilinear
        )
    return data

def edge_magnitude(rgb_hwc: np.ndarray) -> np.ndarray:
    """Returns a 2-D edge-magnitude map from an H×W×3 float image."""
    gray = (0.299 * rgb_hwc[:, :, 0] + 0.587 * rgb_hwc[:, :, 1] + 0.114 * rgb_hwc[:, :, 2])
    gy = gray[2:, 1:-1] - gray[:-2, 1:-1]   
    gx = gray[1:-1, 2:] - gray[1:-1, :-2]   
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_full = np.zeros_like(gray)
    mag_full[1:-1, 1:-1] = mag
    return mag_full

def make_difference_rgb(orig_hwc: np.ndarray, fused_hwc: np.ndarray, amplify: float = 8.0) -> np.ndarray:
    """Offset-corrected signed difference image."""
    diff_hwc = fused_hwc.astype(np.float32) - orig_hwc.astype(np.float32) 
    diff_hw  = diff_hwc.mean(axis=-1)                                       
    diff_hw -= diff_hw.mean()
    diff_scaled = np.clip(diff_hw * amplify + 0.5, 0.0, 1.0)              
    r_ch = np.clip(2.0 * diff_scaled - 0.5, 0.0, 1.0)
    b_ch = np.clip(1.5 - 2.0 * diff_scaled, 0.0, 1.0)
    g_ch = np.clip((1.0 - np.abs(diff_scaled - 0.5) * 2.0) * 0.5, 0.0, 1.0)
    return np.stack([r_ch, g_ch, b_ch], axis=-1)                            

def get_nodata_mask(block: np.ndarray, nodata_threshold: float = 0.002) -> np.ndarray:
    """Valid-pixel mask."""
    return block > nodata_threshold

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="RGB-HSI Pan-Sharpening", layout="wide")
    st.title("🛰️ RGB-HSI Pan-Sharpening")
    st.markdown("Upload your imagery below. Files are written to local storage to prevent RAM crashes and enable interactive exploration.")

    # Initialize Session State
    if "processed" not in st.session_state:
        st.session_state.processed = False

    # 1. UI for File Uploads
    col1, col2 = st.columns(2)
    with col1:
        pan_file = st.file_uploader("Upload PAN Band", type=["tif", "tiff"])
        r_file   = st.file_uploader("Upload Red Band", type=["tif", "tiff"])
    with col2:
        g_file   = st.file_uploader("Upload Green Band", type=["tif", "tiff"])
        b_file   = st.file_uploader("Upload Blue Band", type=["tif", "tiff"])

    output_path = st.text_input("Output File Name (Saved locally)", value="mumbai_fused_high_res.tif")
    process_clicked = st.button("Process Image", type="primary")

    # =======================================================================
    # PHASE 1: PROCESSING (Only runs when button is clicked)
    # =======================================================================
    if process_clicked:
        if not all([pan_file, r_file, g_file, b_file]):
            st.error("Please upload all four TIFF files before processing.")
            return

        start_time = time.time()
        
        # Create persistent uploads directory
        UPLOAD_DIR = "uploads"
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        with st.spinner("Writing uploaded files to disk..."):
            pan_path = os.path.join(UPLOAD_DIR, "pan.tif")
            r_path   = os.path.join(UPLOAD_DIR, "r.tif")
            g_path   = os.path.join(UPLOAD_DIR, "g.tif")
            b_path   = os.path.join(UPLOAD_DIR, "b.tif")
            
            with open(pan_path, "wb") as f: f.write(pan_file.getbuffer())
            with open(r_path, "wb") as f: f.write(r_file.getbuffer())
            with open(g_path, "wb") as f: f.write(g_file.getbuffer())
            with open(b_path, "wb") as f: f.write(b_file.getbuffer())
            
            # Save paths to session state for dynamic reading later
            st.session_state.paths = {'pan': pan_path, 'r': r_path, 'g': g_path, 'b': b_path}

        st.subheader("⚙️ Processing High-Resolution Image")
        progress_text = st.empty()
        progress_bar  = st.progress(0)

        accum_r, accum_g, accum_b = create_accumulator(), create_accumulator(), create_accumulator()
        total_valid_pixels = 0
        total_abs_diff_sum = np.float64(0.0)

        with rasterio.open(pan_path) as pan_src:
            kwargs = pan_src.meta.copy()
            kwargs.update({"count": 3, "dtype": "uint16", "nodata": 0, "compress": "lzw"})

            vrt_opts = {"resampling": Resampling.bilinear, "crs": pan_src.crs, "transform": pan_src.transform, "height": pan_src.height, "width": pan_src.width}
            windows = [w for _, w in pan_src.block_windows(1)]
            total_windows = len(windows)

            with rasterio.open(output_path, "w", **kwargs) as dest, \
                 rasterio.open(r_path) as r_src, WarpedVRT(r_src, **vrt_opts) as vrt_r, \
                 rasterio.open(g_path) as g_src, WarpedVRT(g_src, **vrt_opts) as vrt_g, \
                 rasterio.open(b_path) as b_src, WarpedVRT(b_src, **vrt_opts) as vrt_b:

                for idx, window in enumerate(windows):
                    progress_text.text(f"Processing chunk {idx + 1} of {total_windows}…")

                    pan_block = pan_src.read(1, window=window).astype(np.float32) / 65535.0
                    r_block   = vrt_r.read(1, window=window).astype(np.float32) / 65535.0
                    g_block   = vrt_g.read(1, window=window).astype(np.float32) / 65535.0
                    b_block   = vrt_b.read(1, window=window).astype(np.float32) / 65535.0

                    hue, sat, intensity = rgb_to_hsi(r_block, g_block, b_block)
                    matched_pan         = histogram_match(pan_block, intensity).astype(np.float32)
                    fused_r, fused_g, fused_b = hsi_to_rgb(hue, sat, matched_pan)

                    dest.write((fused_r * 65535).astype("uint16"), 1, window=window)
                    dest.write((fused_g * 65535).astype("uint16"), 2, window=window)
                    dest.write((fused_b * 65535).astype("uint16"), 3, window=window)

                    valid = (get_nodata_mask(r_block) & get_nodata_mask(g_block) & get_nodata_mask(b_block))
                    if valid.any():
                        accumulate_block(accum_r, r_block[valid], fused_r[valid])
                        accumulate_block(accum_g, g_block[valid], fused_g[valid])
                        accumulate_block(accum_b, b_block[valid], fused_b[valid])
                        
                        n_valid = int(valid.sum())
                        for orig_ch, fuse_ch in [(r_block[valid], fused_r[valid]), (g_block[valid], fused_g[valid]), (b_block[valid], fused_b[valid])]:
                            total_abs_diff_sum += np.sum(np.abs(fuse_ch.astype(np.float64) - orig_ch.astype(np.float64)))
                        total_valid_pixels += n_valid

                    progress_bar.progress((idx + 1) / total_windows)

        progress_text.text("Finalizing visualizations...")
        
        # Finalize Metrics
        ergas_r, cc_r = finalize_metrics(accum_r)
        ergas_g, cc_g = finalize_metrics(accum_g)
        ergas_b, cc_b = finalize_metrics(accum_b)
        st.session_state.metrics = {
            "ergas": 100 * 0.5 * np.sqrt((ergas_r + ergas_g + ergas_b) / 3.0),
            "cc": (cc_r + cc_g + cc_b) / 3.0,
            "mean_change": total_abs_diff_sum / max(total_valid_pixels, 1)
        }

        # Previews for Difference Map
        preview_shape = (1000, 1000)
        ms_r  = generate_preview(r_path, preview_shape)[0].astype(np.float32) / 65535.0
        ms_g  = generate_preview(g_path, preview_shape)[0].astype(np.float32) / 65535.0
        ms_b  = generate_preview(b_path, preview_shape)[0].astype(np.float32) / 65535.0
        fused = generate_preview(output_path, preview_shape).astype(np.float32) / 65535.0

        st.session_state.ms_display = np.dstack([apply_contrast_stretch(ms_r), apply_contrast_stretch(ms_g), apply_contrast_stretch(ms_b)])
        st.session_state.fused_display = np.dstack([apply_contrast_stretch(fused[0]), apply_contrast_stretch(fused[1]), apply_contrast_stretch(fused[2])])
        st.session_state.diff_display = make_difference_rgb(st.session_state.ms_display, st.session_state.fused_display, amplify=8.0)
        
        st.session_state.elapsed_time = time.time() - start_time
        st.session_state.output_path = output_path
        st.session_state.processed = True


    # =======================================================================
    # PHASE 2: DISPLAY (Reads entirely from Session State)
    # =======================================================================
    if st.session_state.processed:
        
        st.subheader("📊 Quality Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("ERGAS Score", f"{st.session_state.metrics['ergas']:.4f}")
        m2.metric("Correlation (CC)", f"{st.session_state.metrics['cc']:.4f}")
        m3.metric("Mean Change", f"{st.session_state.metrics['mean_change']:.5f}")

        st.markdown("---")
        
        # -------------------------------------------------------------------
        # NEW FEATURE: Side-by-Side High-Res Urban Comparison
        # -------------------------------------------------------------------
        st.subheader("🏙️ High-Resolution Urban Comparison")
        st.caption("Native 300x300 pixel crops at Urban coordinates (80% X, 50% Y). Notice how the HSI Fused image inherits the sharp structural details of the PAN band while maintaining the spectral colors of the MS band.")
        
        crop_size_urban = 300
        pct_x_urban, pct_y_urban = 80, 50
        pths = st.session_state.paths
        
        with rasterio.open(pths['pan']) as pan_src, rasterio.open(st.session_state.output_path) as fused_src:
            w_u, h_u = pan_src.width, pan_src.height
            
            # Convert percentage to pixel coordinates
            cx_u = max(0, min(int(w_u * (pct_x_urban / 100.0)) - crop_size_urban//2, w_u - crop_size_urban))
            cy_u = max(0, min(int(h_u * (pct_y_urban / 100.0)) - crop_size_urban//2, h_u - crop_size_urban))
            
            crop_window_urban = rasterio.windows.Window(cx_u, cy_u, crop_size_urban, crop_size_urban)
            
            # 1. Read and stretch PAN crop
            pan_crop_raw = pan_src.read(1, window=crop_window_urban).astype(np.float32) / 65535.0
            pan_crop_stretched = apply_contrast_stretch(pan_crop_raw)
            
            # 2. Read and stretch MS crop
            vrt_opts_u = {"resampling": Resampling.bilinear, "crs": pan_src.crs, "transform": pan_src.transform, "height": h_u, "width": w_u}
            with rasterio.open(pths['r']) as rs, WarpedVRT(rs, **vrt_opts_u) as vr, \
                 rasterio.open(pths['g']) as gs, WarpedVRT(gs, **vrt_opts_u) as vg, \
                 rasterio.open(pths['b']) as bs, WarpedVRT(bs, **vrt_opts_u) as vb:
                 
                 ms_crop_stretched = np.dstack([
                     apply_contrast_stretch(vr.read(1, window=crop_window_urban).astype(np.float32) / 65535.0),
                     apply_contrast_stretch(vg.read(1, window=crop_window_urban).astype(np.float32) / 65535.0),
                     apply_contrast_stretch(vb.read(1, window=crop_window_urban).astype(np.float32) / 65535.0),
                 ])

            # 3. Read and stretch Fused crop
            fused_crop_raw = fused_src.read(window=crop_window_urban).astype(np.float32) / 65535.0
            fused_crop_stretched = np.dstack([
                apply_contrast_stretch(fused_crop_raw[0]),
                apply_contrast_stretch(fused_crop_raw[1]),
                apply_contrast_stretch(fused_crop_raw[2]),
            ])

        # Display the 3 images side by side
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(pan_crop_stretched, caption="Original PAN (15m)", use_container_width=True, clamp=True)
        with c2:
            st.image(ms_crop_stretched, caption="Original MS (30m)", use_container_width=True, clamp=True)
        with c3:
            st.image(fused_crop_stretched, caption="Final HSI Fused (15m)", use_container_width=True, clamp=True)


        st.markdown("---")
        st.subheader("🔍 Structural Analysis")
        d1, d2 = st.columns(2)
        with d1:
            st.image(st.session_state.diff_display, caption="Local Structural Difference ×8", use_container_width=True)
        with d2:
            st.markdown("**Per-band Mean Absolute Change** *(preview scale, 0–1)*")
            band_maes = [float(np.mean(np.abs(st.session_state.fused_display[:, :, i] - st.session_state.ms_display[:, :, i]))) for i in range(3)]
            for name, val in zip(["🔴 Red", "🟢 Green", "🔵 Blue"], band_maes):
                st.progress(min(val / 0.05, 1.0), text=f"{name}:  {val:.5f}")

        st.markdown("---")
        st.subheader("🌍 Dynamic Contextual Crop Analysis")
        st.caption("Verify physical edge gains by selecting a preset location or using sliders to manually explore the native 15m resolution data.")
        
        crop_options = {
            "Custom Location (Use Sliders below)": (50, 50)
        }

        crop_choice = st.selectbox("Select a target area:", list(crop_options.keys()))

        if crop_choice == "Custom Location (Use Sliders below)":
            c_x, c_y = st.columns(2)
            pct_x = c_x.slider("X Position (Left to Right %)", 0, 100, 50)
            pct_y = c_y.slider("Y Position (Top to Bottom %)", 0, 100, 50)
        else:
            pct_x, pct_y = crop_options[crop_choice]

        # Dynamic read from the locally saved files based on sliders/dropdown
        crop_size = 300
        
        with rasterio.open(pths['pan']) as pan_src, rasterio.open(st.session_state.output_path) as fused_src:
            w, h = pan_src.width, pan_src.height
            
            # Convert percentage to pixel coordinates
            cx = max(0, min(int(w * (pct_x / 100.0)) - crop_size//2, w - crop_size))
            cy = max(0, min(int(h * (pct_y / 100.0)) - crop_size//2, h - crop_size))
            
            crop_window = rasterio.windows.Window(cx, cy, crop_size, crop_size)
            vrt_opts_crop = {"resampling": Resampling.bilinear, "crs": pan_src.crs, "transform": pan_src.transform, "height": h, "width": w}
            
            with rasterio.open(pths['r']) as rs, WarpedVRT(rs, **vrt_opts_crop) as vr, \
                 rasterio.open(pths['g']) as gs, WarpedVRT(gs, **vrt_opts_crop) as vg, \
                 rasterio.open(pths['b']) as bs, WarpedVRT(bs, **vrt_opts_crop) as vb:
                 
                 crop_ms = np.dstack([
                     apply_contrast_stretch(vr.read(1, window=crop_window).astype(np.float32) / 65535.0),
                     apply_contrast_stretch(vg.read(1, window=crop_window).astype(np.float32) / 65535.0),
                     apply_contrast_stretch(vb.read(1, window=crop_window).astype(np.float32) / 65535.0),
                 ])

            crop_fused_raw = fused_src.read(window=crop_window).astype(np.float32) / 65535.0
            crop_fused = np.dstack([
                apply_contrast_stretch(crop_fused_raw[0]),
                apply_contrast_stretch(crop_fused_raw[1]),
                apply_contrast_stretch(crop_fused_raw[2]),
            ])

            # Apply the fixed-scale threshold logic from earlier to prevent water noise amplification
            ms_edges_raw = edge_magnitude(crop_ms)
            fused_edges_raw = edge_magnitude(crop_fused)

            ms_edges = apply_contrast_stretch(ms_edges_raw)
            fused_edges = apply_contrast_stretch(fused_edges_raw)

            edge_gain_raw = np.clip(fused_edges_raw - ms_edges_raw, 0.0, None)
            edge_gain = np.clip(edge_gain_raw / 0.15, 0.0, 1.0)

        e1, e2, e3 = st.columns(3)
        with e1:
            st.image(ms_edges, caption=f"Original MS (Native {pct_x}%, {pct_y}%)", use_container_width=True, clamp=True)
        with e2:
            st.image(fused_edges, caption="HSI Fused", use_container_width=True, clamp=True)
        with e3:
            st.image(edge_gain, caption="Edge Gain (Fixed Scale)", use_container_width=True, clamp=True)

        st.success(f"Pipeline completed in {st.session_state.elapsed_time:.2f} s. Output saved locally as: {st.session_state.output_path}")

if __name__ == "__main__":
    main()