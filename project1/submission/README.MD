# CS180 Project 1 — README

**Dependencies:** numpy, opencv-python (cv2), scikit-image

## How to run

1. Download and extract the course-provided `data.zip`. Download
   `dam.tif`, `suna.tif`, and `vItali.tif` from
   [this Google Drive folder](https://drive.google.com/drive/folders/1G5GLJT-FlEGLgvvawPm_bm9FfBzBxnLM?usp=sharing).
   The input images are not included in the submission ZIP.
2. Put **all extracted `.jpg`/`.tif` files directly in the same directory as
   `main.py`**, including the three custom files. Do not leave them inside
   a nested `data/` folder. For example, `main.py`, `cathedral.jpg`,
   `church.tif`, and `dam.tif` should be side by side.
3. Open a terminal in that directory and run:

   ```bash
   python -m pip install numpy opencv-python scikit-image
   python main.py
   ```

The script takes no arguments and writes results into `out/` in the current
directory. Basic (single-scale) alignment runs only on the three `.jpg`
images; pyramid alignment runs on all `.jpg` and `.tif` images.

## What it does

- Splits each glass plate into B/G/R channels, aligns G and R to B using
  either single-scale exhaustive search or a multi-scale pyramid, with
  either NCC or L2 as the similarity metric. (You can specify the metric you want to use in signature of `align`/`pyramid_align`)
- Writes aligned color images to `./out/<method>-<metric>/<name>.jpg`.
  `basic-ncc` and `basic-l2` contain only the three `.jpg` images;
  `pyramid-ncc` and `pyramid-l2` contain all images.
- After the main loop, a separate Bells & Whistles step re-aligns
  `emir.tif` using gradient magnitude (`cv2.Sobel`) and the generated image will be saved to `./out/bells-whistles/emir_grad.jpg`.

## Runtime

~7–8 minutes total

## Note

The project webpage (`page.pdf`) has interactive tabs for switching
between the 4 method x metric result sets under "Results". The PDF only
captures whichever tab was active when printed. The full
interactive version is at the URL in the page header/footer.
