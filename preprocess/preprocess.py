"""
处理新数据集: DIP图像 + 有害垃圾照片
流水线: Resize -> GaussianBlur -> CLAHE -> Sharpen
"""
import cv2
import os
import glob
import numpy as np


def resize_image(img, max_size=1280):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def denoise_gaussian(img, kernel=3):
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def enhance_clahe(img, clip_limit=2.0, tile_size=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return img


def sharpen_unsharp(img, strength=1.5):
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    img = cv2.addWeighted(img, strength, blur, -(strength - 1), 0)
    return img


def preprocess(img):
    img = resize_image(img, max_size=1280)
    img = denoise_gaussian(img, kernel=3)
    img = enhance_clahe(img, clip_limit=2.0, tile_size=8)
    img = sharpen_unsharp(img, strength=1.5)
    return img


def make_comparison(original, processed, output_path):
    h = 600
    scale_o = h / original.shape[0]
    new_w_o = int(original.shape[1] * scale_o)
    scale_p = h / processed.shape[0]
    new_w_p = int(processed.shape[1] * scale_p)

    orig = cv2.resize(original, (new_w_o, h))
    proc = cv2.resize(processed, (new_w_p, h))

    label_o = np.zeros((40, new_w_o, 3), dtype=np.uint8)
    label_p = np.zeros((40, new_w_p, 3), dtype=np.uint8)
    cv2.putText(label_o, "Original", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(label_p, "Processed", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    orig = np.vstack([label_o, orig])
    proc = np.vstack([label_p, proc])
    comparison = np.hstack([orig, proc])
    cv2.imwrite(output_path, comparison)
    print(f"  对比图已保存: {output_path}")


def imread_unicode(path):
    """OpenCV imread不支持中文路径，用np.fromfile+cv2.imdecode绕过"""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """写出图片，兼容中文路径"""
    ext = os.path.splitext(path)[1]
    _, buf = cv2.imencode(ext, img)
    buf.tofile(path)


def process_directory(input_dir, output_dir, compare_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg")) +
        glob.glob(os.path.join(input_dir, "*.jpeg")) +
        glob.glob(os.path.join(input_dir, "*.png"))
    )
    if not files:
        print(f"  [跳过] {input_dir} 中没有图片文件")
        return

    print(f"找到 {len(files)} 张图片 ({input_dir})")

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        img = imread_unicode(path)
        if img is None:
            print(f"  [跳过] 无法读取 {fname}")
            continue

        processed = preprocess(img)
        out_path = os.path.join(output_dir, fname)
        imwrite_unicode(out_path, processed)

        print(f"  [{i+1}/{len(files)}] {fname}  "
              f"{img.shape[1]}x{img.shape[0]} -> {processed.shape[1]}x{processed.shape[0]}")

    print(f"完成! 保存至 '{output_dir}/'")

    print("生成对比图...")
    for idx, path in enumerate(files[:2]):
        fname = os.path.basename(path)
        original = imread_unicode(path)
        processed = imread_unicode(os.path.join(output_dir, fname))
        if original is not None and processed is not None:
            compare_path = os.path.join(compare_dir, f"comparison_{idx+1}_{fname}")
            make_comparison(original, processed, compare_path)


if __name__ == "__main__":
    # 1. DIP图像
    print("=" * 50)
    print("处理 DIP图像 (50张PNG)")
    print("=" * 50)
    process_directory("dip_images/DIP图像", "preprocessed_dip", "comparisons_dip")

    # 2. 有害垃圾照片（按类别文件夹）
    print("\n" + "=" * 50)
    print("处理 有害垃圾照片 (48张, 按类别)")
    print("=" * 50)
    harmful_base = "harmful_waste/generated_photo"
    categories = sorted(os.listdir(harmful_base))
    for category in categories:
        cat_path = os.path.join(harmful_base, category)
        if os.path.isdir(cat_path):
            print(f"\n--- {category} ---")
            process_directory(
                cat_path,
                os.path.join("preprocessed_harmful", category),
                os.path.join("comparisons_harmful", category)
            )

    print("\n" + "=" * 50)
    print("全部完成!")
    print("=" * 50)
