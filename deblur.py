import argparse
import os
import torch
import numpy as np
import time
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


from openvino_runner import run_openvino_inference
from models.mprnet import MPRNet
from models.fftformer import fftformer 
def save_image(img_array, path):
    """Збереження numpy масиву як PNG."""
    Image.fromarray(img_array).save(path)

def load_image(image_path, max_size=1024, force_divisible_by=32):
    """Завантажує RGB-зображення, масштабує (якщо > max_size) 
    та додає padding до кратності force_divisible_by."""
    try:
        img = Image.open(image_path).convert('RGB')
        orig_w, orig_h = img.size
        if max(orig_w, orig_h) > max_size:
            scale = max_size / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
        else:
            new_w, new_h = orig_w, orig_h
        new_h = ((new_h + force_divisible_by - 1) // force_divisible_by) * force_divisible_by
        new_w = ((new_w + force_divisible_by - 1) // force_divisible_by) * force_divisible_by
        return img, (orig_w, orig_h), (new_w, new_h)
    except Exception as e:
        raise RuntimeError(f"Не вдалося завантажити {image_path}: {str(e)}")

def deblur_image(image_path, output_path, model_name, device, model_cache=None):
    """Обробка одного фото."""
    if model_name == 'restormer':
        print(f"[INFO] Running {model_name} with OpenVINO...")
        run_openvino_inference(image_path, output_path, model_type=model_name)
        return None

    elif model_name == 'mprnet':
        if model_cache is None:
            model = MPRNet().to(device)
            state_dict = torch.load('weights/mprnet_finetuned_best.pth', map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model_cache = model
        else:
            model = model_cache

        img, (orig_w, orig_h), (new_w, new_h) = load_image(image_path)

        img = img.resize((new_w, new_h), Image.LANCZOS)

        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)[0]
        output_img = output.squeeze().cpu().numpy().transpose(1, 2, 0)
        output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
        output_img = Image.fromarray(output_img).resize((orig_w, orig_h), Image.LANCZOS)
        save_image(np.asarray(output_img), output_path)

        return model_cache

    elif model_name == 'fftformer':
        if model_cache is None:
            model = fftformer().to(device)
            state_dict = torch.load('weights/fftformer_finetuned_best.pth', map_location=device)  # ✅ ЗМІНЕНО
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model_cache = model
        else:
            model = model_cache

        img, (orig_w, orig_h), (new_w, new_h) = load_image(image_path)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
        output_img = output.squeeze().cpu().numpy().transpose(1, 2, 0)
        output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
        output_img = Image.fromarray(output_img).resize((orig_w, orig_h), Image.LANCZOS)
        save_image(np.asarray(output_img), output_path)

        return model_cache

    else:
        raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deblur image(s) using pretrained model.")
    parser.add_argument('--image_path', type=str, help="Path to a single blurred image.")
    parser.add_argument('--input_dir', type=str, help="Directory with blurred images.")
    parser.add_argument('--output_path', type=str, help="Path to save the deblurred image.")
    parser.add_argument('--output_dir', type=str, help="Directory for batch deblurred results.")
    parser.add_argument('--model', required=True, choices=['restormer', 'mprnet', 'fftformer'], help="Model to use.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.image_path and args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        start_time = time.time()
        deblur_image(args.image_path, args.output_path, args.model, device)
        print(f"Done in {time.time() - start_time:.2f} seconds.")

    elif args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        model_cache = None
        start_time = time.time()
        for fname in tqdm(image_files, desc=f"Deblurring with {args.model}"):
            input_path = os.path.join(args.input_dir, fname)
            output_path = os.path.join(args.output_dir, fname)
            try:
                model_cache = deblur_image(input_path, output_path, args.model, device, model_cache)
            except Exception as e:
                print(f"[!] Skipped {fname}: {e}")

        print(f"Total time: {time.time() - start_time:.2f} seconds.")
    else:
        parser.error("Provide either --image_path and --output_path, or --input_dir and --output_dir.")
