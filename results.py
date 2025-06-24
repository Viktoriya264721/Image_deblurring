import os
import csv
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from utils.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    calculate_inception_score,
    calculate_fid,
    get_inception_model
)
from utils.visualizer import save_comparison_image


def load_image_np(path):
    """Завантажує фото як NumPy масив RGB."""
    return np.array(Image.open(path).convert('RGB'))


def extract_inception_activations(image_paths, model, device):
    """Отримує Inception активації для заданого списку шляхів."""
    activations = []
    transform = torch.nn.Sequential(
        torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
    )
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
        img_tensor = transform(img_tensor)
        with torch.no_grad():
            act = model(img_tensor)[0].squeeze().cpu().numpy()
        activations.append(act)
    return np.array(activations)


def evaluate_results(original_dir, restored_dir, blurred_dir, output_csv, summary_txt, visuals_dir):
    """Обчислює PSNR, SSIM, LPIPS, IS, FID для відновлених фото."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = get_inception_model(device)

    orig_files = sorted([f for f in os.listdir(original_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    rest_files = sorted([f for f in os.listdir(restored_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    blur_files = sorted([f for f in os.listdir(blurred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    count = min(len(orig_files), len(rest_files), len(blur_files))
    results = []
    psnr_list, ssim_list, lpips_list = [], [], []
    gt_paths, pred_paths = [], []

    for i in tqdm(range(count), desc="Evaluating images"):
        gt_path = os.path.join(original_dir, orig_files[i])
        pred_path = os.path.join(restored_dir, rest_files[i])
        blurred_path = os.path.join(blurred_dir, blur_files[i])

        try:
            gt = load_image_np(gt_path)
            pred = load_image_np(pred_path)

            psnr = calculate_psnr(gt, pred)
            ssim = calculate_ssim(gt, pred)
            lp = calculate_lpips(gt, pred)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lp)

            results.append({
                'filename': rest_files[i],
                'PSNR': round(psnr, 4),
                'SSIM': round(ssim, 4),
                'LPIPS': round(lp, 4)
            })

            gt_paths.append(gt_path)
            pred_paths.append(pred_path)

            visual_out = os.path.join(visuals_dir, f"{rest_files[i]}_comparison.png")
            save_comparison_image(gt_path, pred_path, blurred_path, visual_out)

        except Exception as e:
            print(f"[!] Error processing index {i}: {e}")

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'PSNR', 'SSIM', 'LPIPS'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    if results:
        print("Calculating Inception Score (IS)...")
        is_mean, is_std = calculate_inception_score(pred_paths, inception_model, device)

        print("Extracting activations for FID...")
        act_gt = extract_inception_activations(gt_paths, inception_model, device)
        act_pred = extract_inception_activations(pred_paths, inception_model, device)
        fid = calculate_fid(act_gt, act_pred)

        with open(summary_txt, 'w') as f:
            f.write("===== AVERAGE METRICS =====\n")
            f.write(f"Mean PSNR: {np.mean(psnr_list):.4f}\n")
            f.write(f"Mean SSIM: {np.mean(ssim_list):.4f}\n")
            f.write(f"Mean LPIPS: {np.mean(lpips_list):.4f}\n\n")
            f.write("===== DISTRIBUTION METRICS =====\n")
            f.write(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}\n")
            f.write(f"FID: {fid:.4f}\n")

    else:
        print("No valid results found, skipping IS and FID calculations.")

    print(f"\nEvaluation complete.")
    print(f"→ CSV: {output_csv}")
    print(f"→ Summary: {summary_txt}")
    print(f"→ Visuals: {visuals_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, 
                        choices=['restormer', 'mprnet', 'fftformer'], 
                        help="Model name used for evaluation.")
    args = parser.parse_args()

    model = args.model_name.lower()

    ORIGINAL_DIR = "original"
    BLURRED_DIR = "blurred"
    RESTORED_DIR = "results"
    OUTPUT_CSV = f"evaluation/metrics_report_{model}.csv"
    SUMMARY_TXT = f"evaluation/summary_{model}.txt"
    VISUALS_DIR = f"evaluation/visuals_{model}"

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(VISUALS_DIR, exist_ok=True)

    evaluate_results(ORIGINAL_DIR, RESTORED_DIR, BLURRED_DIR, OUTPUT_CSV, SUMMARY_TXT, VISUALS_DIR)

