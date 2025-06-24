import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg
from PIL import Image
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_psnr(gt_img, pred_img):
    """Обчислює PSNR для пари gt_img (orig) та pred_img (відновлене)"""
    return peak_signal_noise_ratio(gt_img, pred_img, data_range=255)

def calculate_ssim(gt_img, pred_img):
    """Обчислює SSIM для пари gt_img (orig) та pred_img (відновлене)"""
    h, w, _ = gt_img.shape
    win_size = min(7, h, w)
    return structural_similarity(gt_img, pred_img,
                                 multichannel=True,
                                 data_range=255,
                                 win_size=win_size,
                                 channel_axis=2)

lpips_model = lpips.LPIPS(net='alex').to(device).eval()


def calculate_lpips(img1, img2):
    """Обчислює LPIPS для пари img1, img2"""
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    img1 = img1.to(device)
    img2 = img2.to(device)

    with torch.no_grad():
        dist = lpips_model(img1, img2)

    return dist.item()

def calculate_inception_score(image_paths, model, device, splits=10):
    """Обчислює Inception Score для заданого списку шляхів до зображень"""
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])

    preds = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = F.softmax(model(img_tensor), dim=1).cpu().numpy()
        preds.append(pred)

    preds = np.concatenate(preds, axis=0)

    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
        py = np.mean(part, axis=0)
        scores = [np.sum(pyx * (np.log(pyx) - np.log(py))) for pyx in part]
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def calculate_fid(act1, act2):
    """Обчислює FID для отриманих активацій"""
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    if not np.isfinite(fid):
        print("Warning: FID returned NaN or Inf. Setting FID to large constant (1e4).")
        fid = 1e4

    return fid


def get_inception_model(device):
    """Завантажує InceptionV3 для IS/FID"""
    model = inception_v3(
        weights=Inception_V3_Weights.DEFAULT,
        transform_input=False,
        aux_logits=True
    )
    model.to(device).eval()
    return model
