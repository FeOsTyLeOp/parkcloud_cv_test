import cv2
import streamlit as st
import json
import numpy as np
from PIL import Image
from src.pipeline import ParkingAnalyzer

st.set_page_config(page_title="ParkCloud CV Test(Vitaliy)", layout="wide")

st.title("ParkCloud: Анализ занятости парковки (CV/ML Concept, Vitaliy)")
st.markdown("""
Система принимает на вход изображение с камеры, разметку в формате GeoJSON и калибровку перспективы. 
Алгоритм вычисляет матрицу гомографии, детектирует автомобили и проверяет их пересечение с парковочными местами.
""")


@st.cache_resource
def get_analyzer():
    return ParkingAnalyzer()


analyzer = get_analyzer()

with st.sidebar:
    st.header("Входные данные")

    img_file = st.file_uploader("1. Тестовое изображение (.jpg)", type=["jpg", "jpeg", "png"])
    geojson_file = st.file_uploader("2. Разметка (park.geojson)", type=["json", "geojson"])
    calib_file = st.file_uploader("3. Калибровка (calibrate.json)", type=["json"])

    analyze_btn = st.button("Проанализировать парковку", type="primary")

if img_file and geojson_file and calib_file:
    image = np.array(Image.open(img_file))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image

    geojson_data = json.load(geojson_file)
    calib_data = json.load(calib_file)

    with st.expander("Показать исходное изображение"):
        st.image(image, use_container_width=True)

    if analyze_btn:
        with st.spinner('Анализируем изображение'):
            try:
                result_img_bgr, result_json = analyzer.analyze(
                    image=image_bgr,
                    geojson_data=geojson_data,
                    calibration_data=calib_data
                )

                result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("Визуализация результата")
                    st.image(result_img_rgb, use_container_width=True)
                    st.caption("green - Свободно | red - Занято")

                with col2:
                    st.subheader("Итоговый JSON")
                    st.json(result_json)

            except Exception as e:
                st.error(f"Произошла ошибка при анализе: {str(e)}")
else:
    st.info("Пожалуйста, загрузите все три файла (изображение, geojson и калибровку) в панели слева.")

    with st.expander("Справка по форматам файлов"):
        st.code("""
        // calibrate.json пример:
        {
            "camera_idx": 1,
            "points": [
                {"map_pt": [0, 0], "img_pt": [250, 400]},
                {"map_pt": [10, 0], "img_pt": [800, 400]},
                {"map_pt":[10, 10], "img_pt": [600, 200]},
                {"map_pt": [0, 10], "img_pt":[300, 200]}
            ]
        }
        """, language="json")