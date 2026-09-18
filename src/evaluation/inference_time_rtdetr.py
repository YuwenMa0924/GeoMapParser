import torch
import torchvision.transforms as T
from PIL import Image
import time
import sys
import os

# ====== Important: set this to your RT-DETR root directory ======


from src.core import YAMLConfig


def load_model(config_path, checkpoint_path, device):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']

    cfg.model.load_state_dict(state)

    model = cfg.model.deploy().to(device)
    model.eval()

    postprocessor = cfg.postprocessor.deploy()

    return model, postprocessor


def preprocess(image_path, device):
    im = Image.open(image_path).convert('RGB')
    w, h = im.size

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor()
    ])

    img = transform(im)[None].to(device)
    size = torch.tensor([[w, h]]).to(device)

    return img, size


def measure_fps(model, postprocessor, img, size, device, repeat=100):
    # ===== warm-up =====
    for _ in range(10):
        with torch.no_grad():
            outputs = model(img)
            _ = postprocessor(outputs, size)

    if device == "cuda":
        torch.cuda.synchronize()

    # ===== timing =====
    start = time.time()

    for _ in range(repeat):
        with torch.no_grad():
            outputs = model(img)
            _ = postprocessor(outputs, size)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    total_time = end - start
    avg_time = total_time / repeat
    fps = 1 / avg_time

    print(f"Total time: {total_time:.4f} s")
    print(f"Average inference time: {avg_time:.6f} s")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = r"path/to/rtdetr/configs/rtdetr_r18vd_6x_coco.yml"
    checkpoint_path = r"path/to/rtdetr/output/rtdetr_r18vd_6x_coco/eval/latest.pth"
    image_path = r"path/to/test_image.jpg"   # Replace with your own image

    model, postprocessor = load_model(config_path, checkpoint_path, device)
    img, size = preprocess(image_path, device)

    measure_fps(model, postprocessor, img, size, device, repeat=100)