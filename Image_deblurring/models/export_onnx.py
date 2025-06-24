import torch
import os
from models.restormer import Restormer

os.makedirs("onnx_models", exist_ok=True)

def export_restormer():
    """Експорт ONNX для Restormer."""
    print("Створюємо модель Restormer...")
    model = Restormer()
    checkpoint_path = "weights/restormer_finetuned_best.pth"

    print(f"Завантажуємо ваги Restormer із {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location="cpu")
    if "params" in state:
        state = state["params"]

    model.load_state_dict(state, strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)

    output_path = "onnx_models/restormer.onnx"
    print(f"Експортуємо ONNX у {output_path}...")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=11
        )
        print(f"Успішно експортовано Restormer у {output_path}")
    except Exception as e:
        print(f"Помилка при експорті Restormer: {e}")


if __name__ == "__main__":
    export_restormer()