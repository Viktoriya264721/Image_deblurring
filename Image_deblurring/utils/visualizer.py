from PIL import Image
import os

def save_comparison_image(gt_path, pred_path, blurred_path, output_path):
    try:
        gt = Image.open(gt_path).convert('RGB')
        pred = Image.open(pred_path).convert('RGB')
        blurred = Image.open(blurred_path).convert('RGB')

        width = min(gt.width, pred.width, blurred.width)
        height = min(gt.height, pred.height, blurred.height)

        gt = gt.resize((width, height))
        pred = pred.resize((width, height))
        blurred = blurred.resize((width, height))

        combined = Image.new('RGB', (width * 3, height))
        combined.paste(gt, (0, 0))
        combined.paste(blurred, (width, 0))
        combined.paste(pred, (2 * width, 0))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.save(output_path)

    except Exception as e:
        print(f"Failed to save comparison image for {output_path}: {e}")
