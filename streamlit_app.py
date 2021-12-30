# Завантаження необхідних бібліотек

import os
import streamlit as st
import joblib
import urllib
from urllib.request import urlopen
import pandas as pd
from PIL import Image
import  tensorflow as tf
from tensorflow import keras
from joblib import dump, load

# Додавання назви, опису та зображення

st.title('Додаток з Передбачення Діабету')
st.write('У цій роботі набір даних про діабет був взятий з лікарні Франкфурта, Німеччина. Він містить 2000 випадків спостережень за пацієнтами і 9 характеристик: Вагітність, Глюкоза, Кров\'яний тиск, Товщина шкіри, Інсулін, ІМТ, Родова Функція Діабету, Вік, Результат. Для передбачення діабету була обрана та натренована модель глибокої нейронної мережі. Тренування відбувалося на 80% випадків, тестування - на 20%. Остаточний показник точності передбачень складає 99.5%.')
image = Image.open('Diabetes.jpeg')
st.image(image, use_column_width=True)
st.write('Укажіть ваші показніки та натисніть кнопку "Статус". ')

# Додавання функції з підтягаваняя моделі та масштабера та обробки даних

def inference(row, scaler, model, feat_cols):
    df = pd.DataFrame([row], columns = feat_cols)
    X = scaler.transform(df)
    features = pd.DataFrame(X, columns = feat_cols)
    if (model.predict(features)==0):
        return "Ви здорова людина!"
    else: return "У вас великі шанси захворіти на діабет, зверніться до лікаря!" 
       

age =           st.sidebar.number_input("Вік", 1, 150, 25, 1)
pregnancies =   st.sidebar.number_input("Вагітність", 0, 20, 0, 1)
glucose =       st.sidebar.slider("Рівень Глюкози", 0, 200, 25, 1)
skinthickness = st.sidebar.slider("Товща Шкіри", 0, 99, 20, 1)
bloodpressure = st.sidebar.slider('Кров\'яний Тиск', 0, 122, 69, 1)
insulin =       st.sidebar.slider("Рівень Інсуліну", 0, 846, 79, 1)
bmi =           st.sidebar.slider("Індекс Маси Тіла", 0.0, 67.1, 31.4, 0.1)
dpf =           st.sidebar.slider("Родова Функція Діабету", 0.000, 2.420, 0.471, 0.001)

row = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]

if (st.button('Статус')):
    feat_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    sc = load('scaler.joblib')
    model = keras.models.load_model('my_model')
    result = inference(row, sc, model, feat_cols)
    
    #display the output (Step 4)
    st.write(result) 

# Декодер для збереження фунції в кеші

@st.cache 
def load(scaler_path, model_path):
    sc = joblib.load('scaler.joblib')
    model = keras.models.load_model('my_model')
    return sc , model
