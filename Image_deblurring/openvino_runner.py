import os
import cv2
import numpy as np
from openvino.runtime import Core


class OpenVINOModel:
    """Клас для запуску моделі OpenVINO."""
    def __init__(self, model_xml_path: str, model_bin_path: str):
        self.core = Core()
        self.model = self.core.read_model(model=model_xml_path)
        self.compiled_model = self.core.compile_model(self.model, device_name="CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Запускає inference для одного фото (N,H,W,C)."""
        input_blob = self._prepare_input(image)
        result = self.compiled_model([input_blob])[self.output_layer]
        return self._postprocess_output(result)

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Підготовка фото для моделі."""
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0)

    def _postprocess_output(self, output: np.ndarray) -> np.ndarray:
        """Постобробка результату."""
        output = np.squeeze(output)  # видалення batch
        output = np.transpose(output, (1, 2, 0))
        return np.clip(output * 255.0, 0, 255).astype(np.uint8)


def load_image(image_path: str) -> np.ndarray:
    """Завантаження фото у форматі BGR."""
    return cv2.imread(image_path, cv2.IMREAD_COLOR)


def save_image(image: np.ndarray, save_path: str):
    """Збереження фото."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)


def run_model(model_type: str, input_path: str, output_path: str):
    """Запускає OpenVINO для restormer."""
    if model_type != "restormer":
        raise ValueError("Підтримується лише модель 'restormer'.")

    model_dir = "openvino_models/restormer"
    model_name = "restormer"

    model_xml = os.path.join(model_dir, f"{model_name}.xml")
    model_bin = os.path.join(model_dir, f"{model_name}.bin")

    print(f"Завантаження моделі {model_name} із {model_dir}")

    if not (os.path.exists(model_xml) and os.path.exists(model_bin)):
        raise FileNotFoundError(f"Не знайдено файли моделі: {model_xml} / {model_bin}")

    model = OpenVINOModel(model_xml, model_bin)
    image = load_image(input_path)

    result = model.infer(image)

    save_image(result, output_path)
    print(f"Збережено результат у {output_path}")


def run_openvino_inference(image_path, output_path, model_type):
    """Для сумісності із deblur.py."""
    return run_model(model_type, image_path, output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run OpenVINO inference.")
    parser.add_argument('--image_path', required=True, help="Шлях до вхідного фото.")
    parser.add_argument('--output_path', required=True, help="Шлях для збереження результату.")
    parser.add_argument('--model', required=True, choices=['restormer'], help="Модель для використання.")
    args = parser.parse_args()

    run_model(args.model, args.image_path, args.output_path)
