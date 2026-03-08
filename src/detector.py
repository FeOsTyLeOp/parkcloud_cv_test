# src/detector.py
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple


class CarDetector:
    def __init__(self, model_path: str = "yolo11n.pt"):  # <--- ИЗМЕНИЛИ НА YOLO11
        """
        Инициализация детектора. Используем современный YOLOv11 (nano).
        """
        self.model = YOLO(model_path)
        self.target_classes = [2, 5, 7]

    def get_car_bottom_centers(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        Находит машины на изображении и возвращает координаты
        нижней центральной точки их bounding box'ов (точка контакта с землей).
        """
        results = self.model(image, classes=self.target_classes, verbose=False)

        centers = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2.0
                center_y = y2
                centers.append((center_x, center_y))

        return centers