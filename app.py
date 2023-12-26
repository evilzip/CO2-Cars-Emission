# Importing libraries-----------------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsRegressor
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)

# Загрузка датасэта
df = pd.read_csv('Original MY2000-2014 Fuel Consumption Ratings.csv', low_memory=False)

#переименовываем столбцы
df = df.rename(columns={'ENGINE SIZE': 'ENGINE SIZE (L)', 
                  'FUEL':'FUEL TYPE', 
                  'FUEL CONSUMPTION':'FUEL CONSUMPTION - CITY (L/100 km)',
                  'Unnamed: 9': 'FUEL CONSUMPTION - HWY (L/100 km)',
                  'Unnamed: 10': 'FUEL CONSUMPTION - COMB (L/100 km)',
                  'Unnamed: 11':'FUEL CONSUMPTION - COMB (mpg)',
                  'CO2 EMISSIONS ':'CO2 EMISSIONS (g/km)'
})
df = df.rename(str.lower, axis='columns')
#удаляем неинформативные столбцы и строки
for i in df.columns:
    if i.startswith('unnamed'):
        df = df.drop(columns=i)
df = df.drop(index=0, columns =['model','make','model.1'])
#проверяем наличие дупликатов
dupl_columns = list(df.columns)
dupl_columns.remove('co2 emissions (g/km)')
mask = df.duplicated(subset=dupl_columns)
df_dupl=df[mask]
#удаляем дупликаты
df = df.drop_duplicates(subset=dupl_columns, ignore_index=True)
#изменяет типы данных в столбцах
df_col = list(df.columns)
df_col.remove('vehicle class')
df_col.remove('transmission')
df_col.remove('fuel type')
for i in df_col:
    df[df_col] = df[df_col].astype(float)

df_correlation = df[['engine size (l)','cylinders','fuel consumption - comb (l/100 km)','co2 emissions (g/km)']]
df_model = df_correlation[(np.abs(stats.zscore(df_correlation)) < 1.65).all(axis=1)]


# Подготовка данных на моделях
X = df_model[['engine size (l)','cylinders','fuel consumption - comb (l/100 km)']]
y = df_model['co2 emissions (g/km)']

# Тренируем модель - Метод k-ближайших соседей (k-nearest neighbors algorithm) победивший по итогам сравнения
model = KNeighborsRegressor()
model.fit(X.values, y.values)

# Создание Streamlit web app
# Утсновка фона по урл ссылке
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://mtdata.ru/u16/photoA46B/20632779702-0/original.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)
# Оглавление и описание
st.title('Предсказание выбросов СО2')
st.write('Введите параметры двигателя для предсказания выбросов СО2')

# Ввод пользователя
engine_size = st.number_input('Объем двигателя (Л)', step=0.1, format="%.1f")
cylinders = st.number_input('Количество цилиндров', min_value=2, max_value=16, step=1)
fuel_consumption = st.number_input('Расход топлива в смешанном режиме (л/100 км)', step=0.1, format="%.1f")

# Предсказание
input_data = [[cylinders, engine_size, fuel_consumption]]
predicted_co2 = model.predict(input_data)

# Вывод результата предсказания
st.write(f'Предсказанный выброс CO2: {predicted_co2[0]:.2f} g/km')
