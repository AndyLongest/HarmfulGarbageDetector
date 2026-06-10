"""
数字图像处理 - 可回收垃圾图像预处理脚本
流水线: Resize -> GaussianBlur -> CLAHE -> Sharpen
"""
import cv2
import os
import glob
import numpy as np


def resize_image(img, max_size=1280):
    """将长边限制为max_size，保持宽高比，使用INTER_AREA缩小"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def denoise_gaussian(img, kernel=3):
    """高斯滤波去噪"""
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def enhance_clahe(img, clip_limit=2.0, tile_size=8):
    """CLAHE对比度增强 - 在LAB颜色空间的L通道上操作"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return img


def sharpen_unsharp(img, strength=1.5):
    """Unsharp Masking 锐化"""
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    img = cv2.addWeighted(img, strength, blur, -(strength - 1), 0)
    return img


def preprocess(img):
    """完整预处理流水线"""
    img = resize_image(img, max_size=1280)
    img = denoise_gaussian(img, kernel=3)
    img = enhance_clahe(img, clip_limit=2.0, tile_size=8)
    img = sharpen_unsharp(img, strength=1.5)
    return img


def make_comparison(original, processed, output_path):
    """制作原始图 vs 处理后的对比图（用于实验报告）"""
    # 统一高度以便并排对比
    h = 600
    scale_o = h / original.shape[0]
    new_w_o = int(original.shape[1] * scale_o)
    scale_p = h / processed.shape[0]
    new_w_p = int(processed.shape[1] * scale_p)

    orig = cv2.resize(original, (new_w_o, h))
    proc = cv2.resize(processed, (new_w_p, h))

    # 加标题
    label_o = np.zeros((40, new_w_o, 3), dtype=np.uint8)
    label_p = np.zeros((40, new_w_p, 3), dtype=np.uint8)
    cv2.putText(label_o, "Original", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(label_p, "Processed", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    orig = np.vstack([label_o, orig])
    proc = np.vstack([label_p, proc])

    # 拼接
    comparison = np.hstack([orig, proc])
    cv2.imwrite(output_path, comparison)
    print(f"  对比图已保存: {output_path}")


def main():
    input_dir = "1"
    output_dir = "preprocessed"
    compare_dir = "comparisons"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    print(f"找到 {len(files)} 张图片")

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"  [跳过] 无法读取 {fname}")
            continue

        processed = preprocess(img)
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, processed)

        print(f"  [{i+1}/{len(files)}] 处理完成: {fname}  "
              f"{img.shape[1]}x{img.shape[0]} -> {processed.shape[1]}x{processed.shape[0]}")

    print(f"\n全部完成! 预处理图片保存在 '{output_dir}/'")

    # ===== 生成2张对比图 =====
    print("\n生成预处理前后对比图...")
    os.makedirs(compare_dir, exist_ok=True)
    for idx, path in enumerate(files[:2]):
        fname = os.path.basename(path)
        original = cv2.imread(path)
        processed = cv2.imread(os.path.join(output_dir, fname))
        if original is not None and processed is not None:
            compare_path = os.path.join(compare_dir, f"comparison_{idx+1}_{fname}")
            make_comparison(original, processed, compare_path)


if __name__ == "__main__":
    main()
