import cv2
import json

IMAGE_PATH = "1.jpg"
points = []
polygons = []


def mouse_callback(event, x, y, fags, param):
    global points, polygons, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)

        if len(points) > 1:
            cv2.line(img_copy, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)

        cv2.imshow("Annotator", img_copy)

    elif event == cv2.EVENT_RBUTTONDOWN:  # Правый клик - замкнуть полигон
        if len(points) >= 3:
            cv2.line(img_copy, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
            polygons.append(points.copy())
            print(f"Полигон {len(polygons)} сохранен!")
            points.clear()
            cv2.imshow("Annotator", img_copy)


# Загружаем картинку
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Ошибка: не удалось загрузить {IMAGE_PATH}")
    exit()

img_copy = img.copy()

print("ИНСТРУКЦИЯ:")
print("1. Левый клик -поставить точку парковочного места")
print("2. Правый клик -замкнуть полигон")
print("3. Нажмите q или Enter чтобы сохранить GeoJSON и выйти")

cv2.namedWindow("Annotator")
cv2.setMouseCallback("Annotator", mouse_callback)

while True:
    cv2.imshow("Annotator", img_copy)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 13:
        break

cv2.destroyAllWindows()

features = []
for idx, poly in enumerate(polygons):
    features.append({
        "type": "Feature",
        "properties": {"id": str(idx + 1)},
        "geometry": {
            "type": "Polygon",
            "coordinates": [poly + [poly[0]]]
        }
    })

geojson_data = {"type": "FeatureCollection", "features": features}

with open("park.geojson", "w") as f:
    json.dump(geojson_data, f, indent=4)

print("Разметка успешно сохранена в park.geojson!")