# CS180 Project 1 — README

**Dependencies:** numpy, opencv-python (cv2), scikit-image

## How to run

```
python main.py
```

## Setup

- Place `main.py` in the same folder as the provided `.jpg`/`.tif` images
  (`cathedral.jpg`, `monastery.jpg`, `tobolsk.jpg`, `church.tif`, `emir.tif`, etc.)
  plus any custom images (`dam.tif`, `suna.tif`, `vItali.tif`).
- Running `main.py` with no arguments processes every image listed in
  `jpg_img` / `tif_img` / `custom_img` at the top of the file.

## What it does

- Splits each glass plate into B/G/R channels, aligns G and R to B using
  either single-scale exhaustive search or a multi-scale pyramid, with
  either NCC or L2 as the similarity metric.
- Writes aligned color images to `./out/<method>-<metric>/<name>.jpg` for
  all 4 method x metric combinations (`basic-ncc`, `basic-l2`, `pyramid-ncc`,`pyramid-l2`), for every image.
- After the main loop, a separate Bells & Whistles step re-aligns
  `emir.tif` using gradient magnitude (`cv2.Sobel`) and the generated image will be saved to `./out/bells-whistles/emir_grad.jpg`.

## Runtime

~7–8 minutes total

## Note

The project webpage (`page.pdf`) has interactive tabs for switching
between the 4 method x metric result sets under "Results". The PDF only
captures whichever tab was active when printed. The full
interactive version is at the URL in the page header/footer.
