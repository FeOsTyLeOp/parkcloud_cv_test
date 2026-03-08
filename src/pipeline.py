import cv2
import numpy as np
from typing import Dict, Any, Tuple
from shapely.geometry import Point
from .detector import CarDetector
from .geometry import get_homography_matrix, transform_points, parse_parking_spots


class ParkingAnalyzer:
    def __init__(self):
        self.detector = CarDetector()

    def analyze(self, image: np.ndarray, geojson_data: dict, calibration_data: dict,
                park_idx: int = 1, test_idx: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Полный цикл обработки: от картинки до готового JSON и визуализации.
        """
        parking_spots = parse_parking_spots(geojson_data)

        h_matrix = get_homography_matrix(calibration_data)

        car_img_points = self.detector.get_car_bottom_centers(image)
        car_map_points = transform_points(car_img_points, h_matrix)

        h_inv = np.linalg.inv(h_matrix)

        spot_status = {}
        result_image = image.copy()

        for spot_id, polygon in parking_spots.items():
            is_occupied = False
            for map_pt in car_map_points:
                if polygon.contains(Point(map_pt)):
                    is_occupied = True
                    break

            spot_status[spot_id] = {"detected": is_occupied}

            poly_map_coords = list(polygon.exterior.coords)
            poly_img_coords = transform_points(poly_map_coords, h_inv)

            pts = np.array(poly_img_coords, np.int32).reshape((-1, 1, 2))
            color = (0, 0, 255) if is_occupied else (0, 255, 0)

            overlay = result_image.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.4, result_image, 0.6, 0, result_image)
            cv2.polylines(result_image, [pts], isClosed=True, color=color, thickness=2)

            if len(poly_img_coords) > 0:
                cv2.putText(result_image, str(spot_id),
                            (int(poly_img_coords[0][0]), int(poly_img_coords[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        result_json = {
            "params": {
                "park_idx": park_idx,
                "calibrate_idx": calibration_data.get("camera_idx", 1)
            },
            "result": spot_status
        }

        return result_image, result_json