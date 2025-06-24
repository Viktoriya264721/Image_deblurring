# Image Deblurring Project

Цей проєкт містить реалізації трьох моделей для деблюрингу фото:
- **Restormer**
- **MPRNet**
- **FFTFormer**

Включає:
- Тренування моделей
- Інференс для деблюрингу
- Точне оцінювання результатів (PSNR, SSIM, LPIPS, IS, FID)

---

## Структура папок
```
Image_deblurring/
├─ evaluation/ # Збереження результатів оцінки
├─ fine_tuning/ # Тренування моделей
├─ models/
│ └─ restormer.py # Реалізація Restormer
│ └─ mprnet.py # Реалізація MPRNet
│ └─ fftformer.py # Реалізація FFTFormer
├─ utils/
│ └─ metrics.py # Реалізації PSNR, SSIM, LPIPS, IS, FID
│ └─ visualizer.py # Допоміжні функції для візуалізації
├─ weights/ # Збережені ваги моделей
├─ deblur.py # Інференс для деблюрингу
├─ openvino_runner.py # Інтеграція моделей із OpenVINO
├─ requirements.txt # Залежності
```
