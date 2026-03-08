import cv2
import numpy as np
from shapely.geometry import Polygon
from typing import Dict, List, Tuple


def get_homography_matrix(calibration_data: dict) -> np.ndarray:
    """
    Вычисляет матрицу гомографии (перспективы) из пар точек.
    calibration_data: dict, где есть пары точек map_pt (карта) и img_pt (камера).
    Возвращает матрицу, переводящую из координат ИЗОБРАЖЕНИЯ в координаты КАРТЫ.
    """
    map_pts = []
    img_pts = []

    for point_pair in calibration_data.get("points", []):
        map_pts.append(point_pair["map_pt"])
        img_pts.append(point_pair["img_pt"])

    src_pts = np.array(img_pts, dtype=np.float32)
    dst_pts = np.array(map_pts, dtype=np.float32)

    matrix, _ = cv2.findHomography(src_pts, dst_pts)
    return matrix


def transform_points(points: List[Tuple[float, float]], matrix: np.ndarray) -> List[Tuple[float, float]]:
    """
    Переводит список точек через матрицу гомографии.
    """
    if not points:
        return []

    # Форматируем массив для OpenCV
    pts_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts_array, matrix)

    result = []
    for pt in transformed:
        result.append((float(pt[0][0]), float(pt[0][1])))
    return result


def parse_parking_spots(geojson_data: dict) -> Dict[str, Polygon]:
    """
    Преобразует GeoJSON в словарь объектов Shapely Polygon.
    Ключ - ID парковочного места.
    """
    spots = {}
    features = geojson_data.get("features", [])

    for feature in features:
        spot_id = feature.get("properties", {}).get("id", "unknown")
        coords = feature.get("geometry", {}).get("coordinates", [[]])[0]
        if coords:
            spots[str(spot_id)] = Polygon(coords)

    return spots
