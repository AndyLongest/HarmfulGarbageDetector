import argparse
import base64
import cgi
import io
import json
import mimetypes
import sys
import time
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import preprocess


WEB_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_WEIGHTS = ROOT / "exp12" / "weights" / "best.pt"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

# Adjust this set if the application's hazardous-waste definition changes.
HAZARDOUS_CLASSES = {
    "Powerbank",
    "CosmeticBottles",
    "DryBattery",
    "Ointment",
    "ExpiredDrugs",
}

CHINESE_NAMES = {
    "FastFoodBox": "快餐盒",
    "SoiledPlastic": "受污染塑料",
    "Cigarette": "烟头",
    "Toothpick": "牙签",
    "Flowerpot": "花盆",
    "BambooChopstics": "竹筷",
    "Meal": "剩饭",
    "Bone": "骨头",
    "FruitPeel": "果皮",
    "Pulp": "纸浆",
    "Tea": "茶叶",
    "Vegetable": "蔬菜",
    "Eggshell": "蛋壳",
    "FishBone": "鱼骨",
    "Powerbank": "充电宝",
    "Bag": "包",
    "CosmeticBottles": "化妆品瓶",
    "Toys": "玩具",
    "PlasticBowl": "塑料碗",
    "PlasticHanger": "塑料衣架",
    "PaperBags": "纸袋",
    "PlugWire": "插头电线",
    "OldClothes": "旧衣物",
    "Can": "易拉罐",
    "Pillow": "枕头",
    "PlushToys": "毛绒玩具",
    "ShampooBottle": "洗发水瓶",
    "GlassCup": "玻璃杯",
    "Shoes": "鞋",
    "Anvil": "金属砧",
    "Cardboard": "纸板",
    "SeasoningBottle": "调味瓶",
    "Bottle": "瓶子",
    "MetalFoodCans": "金属食品罐",
    "Pot": "锅",
    "EdibleOilBarrel": "食用油桶",
    "DrinkBottle": "饮料瓶",
    "DryBattery": "干电池",
    "Ointment": "药膏",
    "ExpiredDrugs": "过期药品",
}


class WasteDetector:
    def __init__(self, weights, device=""):
        self.model = torch.hub.load(
            str(ROOT),
            "custom",
            path=str(weights),
            source="local",
            device=device or None,
            _verbose=False,
        )
        self.model.iou = 0.45
        self.model.max_det = 200

    def detect(self, image, confidence):
        original_size = [image.width, image.height]
        preprocess_started = time.perf_counter()
        bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed_bgr = preprocess(bgr_image)
        processed_image = Image.fromarray(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB))
        preprocessing_ms = round((time.perf_counter() - preprocess_started) * 1000, 1)

        self.model.conf = confidence
        started = time.perf_counter()
        results = self.model(processed_image, size=640)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        predictions = results.xyxy[0].cpu().tolist()
        names = results.names

        detections = []
        for x1, y1, x2, y2, score, class_id in predictions:
            name = names[int(class_id)]
            detections.append(
                {
                    "class_id": int(class_id),
                    "name": name,
                    "display_name": CHINESE_NAMES.get(name, name),
                    "confidence": round(score, 4),
                    "hazardous": name in HAZARDOUS_CLASSES,
                    "box": [round(x1), round(y1), round(x2), round(y2)],
                }
            )

        annotated = draw_detections(processed_image, detections)
        return {
            "detections": detections,
            "annotated_image": annotated,
            "inference_ms": elapsed_ms,
            "preprocessing_ms": preprocessing_ms,
            "original_size": original_size,
            "processed_size": [processed_image.width, processed_image.height],
        }


def draw_detections(image, detections):
    canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = canvas.shape[:2]
    thickness = max(2, round((height + width) / 700))
    font_scale = max(0.5, min(0.9, (height + width) / 1800))

    for item in detections:
        x1, y1, x2, y2 = item["box"]
        color = (44, 44, 225) if item["hazardous"] else (83, 180, 55)
        label = f'{item["name"]} {item["confidence"]:.2f}'
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        top = max(0, y1 - label_height - baseline - 8)
        cv2.rectangle(canvas, (x1, top), (min(width, x1 + label_width + 10), y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 5, y1 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Failed to encode annotated image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def parse_image(handler):
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
        raise ValueError("图片为空或超过 15 MB 限制")

    content_type, parameters = cgi.parse_header(handler.headers.get("Content-Type", ""))
    if content_type != "multipart/form-data":
        raise ValueError("请求必须使用 multipart/form-data")

    parameters["boundary"] = parameters["boundary"].encode()
    parameters["CONTENT-LENGTH"] = content_length
    form = cgi.parse_multipart(handler.rfile, parameters)
    image_data = form.get("image", [None])[0]
    if not image_data:
        raise ValueError("未找到上传图片")

    confidence_raw = form.get("confidence", [b"0.15"])[0]
    if isinstance(confidence_raw, bytes):
        confidence_raw = confidence_raw.decode("utf-8")
    confidence = min(0.9, max(0.01, float(confidence_raw)))

    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_data))).convert("RGB")
    if image.width * image.height > 30_000_000:
        raise ValueError("图片尺寸过大")
    return image, confidence


class RequestHandler(SimpleHTTPRequestHandler):
    detector = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self):
        if urlparse(self.path).path == "/api/health":
            self.send_json({"status": "ok", "model": DEFAULT_WEIGHTS.name, "preprocessing": True})
            return
        super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/detect":
            self.send_error(404)
            return

        try:
            image, confidence = parse_image(self)
            result = self.detector.detect(image, confidence)
            detections = result["detections"]
            counts = Counter(item["display_name"] for item in detections)
            hazardous = [item for item in detections if item["hazardous"]]
            hazardous_counts = Counter(item["display_name"] for item in hazardous)
            self.send_json(
                {
                    "detections": detections,
                    "counts": dict(counts),
                    "hazardous_counts": dict(hazardous_counts),
                    "hazardous": bool(hazardous),
                    "total": len(detections),
                    "inference_ms": result["inference_ms"],
                    "preprocessing_ms": result["preprocessing_ms"],
                    "original_size": result["original_size"],
                    "processed_size": result["processed_size"],
                    "confidence": confidence,
                    "annotated_image": result["annotated_image"],
                }
            )
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stdout.write(f"[web] {self.address_string()} - {fmt % args}\n")


def main():
    parser = argparse.ArgumentParser(description="Hazardous waste detection web application")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="", help="cpu or CUDA device, e.g. 0")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    print(f"Loading model: {args.weights}")
    RequestHandler.detector = WasteDetector(args.weights, args.device)
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"Application ready: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
