# 🛰️ RGB-HSI Pan-Sharpening Pipeline

A robust, memory-efficient Streamlit application for fusing low-resolution multispectral (MS) satellite imagery with high-resolution panchromatic (PAN) bands using the Hue-Saturation-Intensity (HSI) color space.

This tool performs block-by-block processing to handle massive `.tif` satellite files without exhausting system RAM. It also provides an interactive dashboard to verify both structural enhancement and spectral fidelity.

---

## Key Features

- Memory-Safe Processing  
  Uses rasterio block windows to process large satellite datasets with a constant, low RAM footprint.

- HSI Fusion Algorithm  
  Separates spatial intensity from spectral color and injects high-resolution PAN details into MS imagery.

- Quantitative Metrics  
  Automatically computes:
  - ERGAS (Relative Global Error)
  - Correlation Coefficient (CC)
  - Per-band Mean Absolute Change

- Interactive Visualizations
  - Side-by-side structural comparisons
  - Local structural difference maps (×8 enhancement)
  - Dynamic crop-based analysis for edge verification

---

## Tech Stack

- Python
- Streamlit – Interactive web UI
- Rasterio – Geospatial raster processing
- NumPy – Numerical computation and image processing

---

### Installation & Setup

It is recommended to use a virtual environment.

### 1. Clone the Repository
git clone https://github.com/ab-sgh/Image-Fusion-using-the-RGB-HSI-Method
cd rgb-hsi-pansharpening

### 2. Create & Activate Virtual Environment

Mac/Linux:
python3 -m venv myenv
source myenv/bin/activate

Windows:
python -m venv myenv
myenv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the Application
streamlit run main.py

---

## Project Structure

rgb-hsi-pansharpening/

main.py          - Streamlit UI and main pipeline  
hsi_core.py      - RGB ↔ HSI conversion + histogram matching  
metrics.py       - ERGAS, CC, and evaluation metrics  
requirements.txt - Dependencies  
README.md        - Documentation  

---

## Usage

1. Open the app in your browser (default: http://localhost:8501)
2. Upload TIFF files:
   - Panchromatic (PAN)
   - Red band
   - Green band
   - Blue band
3. Enter output filename
4. Click "Process Image"

The app will:
- Process images chunk-by-chunk (memory safe)
- Save output locally
- Display metrics and comparisons

---

## Requirements & Notes

- Designed for Landsat 8 or similar satellite imagery
- Input TIFF files must be geographically aligned and share the same CRS

---

## Output Insights

The dashboard helps validate:

- Edge enhancement from PAN injection  
- Color preservation from MS bands  
- Fusion quality via quantitative metrics  

---
