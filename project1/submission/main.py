import os
import time
import numpy as np
import cv2 as cv
import skimage as sk

# -------------------- Helper Functions --------------------
def save_helper(im_out):
    out_uint8 = np.clip(im_out * 255.0, 0, 255).astype(np.uint8)
    return cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)

def l2Norm(img1, img2):
    v1 = img1 - np.mean(img1)
    v2 = img2 - np.mean(img2)
    v1 = v1 / np.sqrt(np.sum(v1 ** 2))
    v2 = v2 / np.sqrt(np.sum(v2 ** 2))
    return np.sum((v1 - v2) ** 2)

def ncc(img1, img2):
    v1 = img1 - np.mean(img1)
    v2 = img2 - np.mean(img2)
    v1 = v1 / np.sqrt(np.sum(v1 ** 2))
    v2 = v2 / np.sqrt(np.sum(v2 ** 2))
    res = np.sum(v1 * v2)
    return res

def gradient(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)

def pyramid_helper(img, min_size = 300):
    pyramid_list = [img]
    curr = img
    while max(curr.shape) > min_size:
        curr = sk.transform.rescale(curr, 0.5, anti_aliasing=True)
        pyramid_list.append(curr)
    return pyramid_list

# -------------------- alignment --------------------

# single-scale
def align(moving, base, window=15, metric="ncc", center_shift = (0, 0), name="", features="pixel"):
    if metric == "l2":
        best_score = float("inf")
    elif metric == "ncc":
        best_score = float("-inf")

    if features == "gradient":
        moving_proc = gradient(moving)
        base_proc = gradient(base)
    else:
        moving_proc = moving
        base_proc = base
    margin = window + max(abs(center_shift[0]), abs(center_shift[1]))
    base_cropped = base_proc[margin:-margin, margin:-margin]
    best_shift = center_shift

    for dy in range(-window, window + 1):   # row
        for dx in range(-window, window + 1): # column
            new_shift = (center_shift[0] + dy, center_shift[1] + dx)
            curr_proc = np.roll(moving_proc, shift=new_shift, axis=(0,1))
            curr_cropped = curr_proc[margin:-margin, margin:-margin]

            if metric == "l2":
                curr_score = l2Norm(base_cropped, curr_cropped)
                if curr_score < best_score:
                    best_shift = new_shift
                    best_score = curr_score

            elif metric == "ncc":
                curr_score = ncc(base_cropped, curr_cropped)
                if curr_score > best_score:
                    best_shift = new_shift
                    best_score = curr_score
    res = np.roll(moving, shift=best_shift, axis = (0, 1))
    print(f"[{name}] - best shift: {best_shift}")
    return (res, best_shift)

# pyramid-scale
def pyramid_align(moving, base, min_size = 300, metric = "ncc", name="", features = "pixel"):
    moving_pyramid = pyramid_helper(moving, min_size)
    base_pyramid = pyramid_helper(base, min_size)
    best_shift = (0, 0)
    for i in range(len(moving_pyramid)-1, -1, -1):
        curr_moving, curr_base = moving_pyramid[i], base_pyramid[i]
        window = 15 if i == len(moving_pyramid) - 1 else 3
        _, best_shift = align(curr_moving, curr_base, window=window, metric=metric, center_shift=best_shift, name=name, features=features)
        if i > 0:
            best_shift = (best_shift[0] * 2, best_shift[1] * 2)

    res = np.roll(moving, shift=best_shift, axis=(0, 1))
    return (res, best_shift)


# -------------------- Run Batch --------------------

jpg_img = ["cathedral.jpg", "monastery.jpg", "tobolsk.jpg"]
tif_img = [
    "church.tif", "emir.tif", "harvesters.tif", "icon.tif",
    "ilemselga.tif", "melons.tif", "religous_painting.tif",
    "self_portrait.tif", "siren.tif", "three_generations.tif",
    "wharf.tif",
]
custom_img = ["dam.tif", "suna.tif", "vItali.tif"]

if __name__ == "__main__":
    os.makedirs("./out", exist_ok=True)

    for imname in jpg_img + tif_img + custom_img:
        im = cv.imread(imname, cv.IMREAD_GRAYSCALE)
        im = im.astype(np.float32) / 255.0

        height = int(np.floor(im.shape[0] / 3.0))
        width = im.shape[1]

        b = im[:height]
        g = im[height: 2 * height]
        r = im[2 * height: 3 * height]

        height_helper = int(0.08 * height)
        width_helper = int(0.09 * width)
        b = b[height_helper:-height_helper, width_helper:-width_helper]
        g = g[height_helper:-height_helper, width_helper:-width_helper]
        r = r[height_helper:-height_helper, width_helper:-width_helper]

        name = os.path.splitext(imname)[0]

        methods = [("pyramid", pyramid_align)]
        if imname in jpg_img:
            methods.insert(0, ("basic", align))

        for method_name, method_func in methods:
            for metric in ["ncc", "l2"]:
                t0 = time.time()
                ag, g_shift = method_func(g, b, metric=metric, name=f"{imname}-G")
                ar, r_shift = method_func(r, b, metric=metric, name=f"{imname}-R")
                duration = time.time() - t0

                im_out = np.dstack([ar, ag, b])

                out_dir = f"./out/{method_name}-{metric}"
                os.makedirs(out_dir, exist_ok=True)
                out_path = f"{out_dir}/{name}.jpg"
                cv.imwrite(out_path, save_helper(im_out))

                print(f"{imname}:{method_name}-{metric}: G shift {g_shift}, R shift {r_shift}, time {duration:.2f}s")

    # -------------------- Bells & Whistles --------------------

    im = cv.imread("emir.tif", cv.IMREAD_GRAYSCALE)
    im = im.astype(np.float32) / 255.0

    height = int(np.floor(im.shape[0] / 3.0))
    width = im.shape[1]

    b = im[:height]
    g = im[height: 2 * height]
    r = im[2 * height: 3 * height]

    height_helper = int(0.08 * height)
    width_helper = int(0.09 * width)
    b = b[height_helper:-height_helper, width_helper:-width_helper]
    g = g[height_helper:-height_helper, width_helper:-width_helper]
    r = r[height_helper:-height_helper, width_helper:-width_helper]

    ag, g_shift = pyramid_align(g, b, metric="ncc", name="emir-G", features="gradient")
    ar, r_shift = pyramid_align(r, b, metric="ncc", name="emir-R", features="gradient")

    im_out = np.dstack([ar, ag, b])
    out_dir = "./out/bells-whistles"
    os.makedirs(out_dir, exist_ok=True)
    cv.imwrite(f"{out_dir}/emir_grad.jpg", save_helper(im_out))

    print(f"emir.tif bells-whistles: G shift {g_shift}, R shift {r_shift}")
