import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Next Cost ''')
st.image("gasto.jpg", caption="Planea tus gastos con predicciones inteligentes.")

st.header('Datos de evaluación')

def user_input_features():
  # Entrada
  Presupuesto = st.number_input('Presupuesto (MXN):', min_value=1, max_value=1000000, value = 1, step = 1)
  Tiempo_invertido = st.number_input('Tiempo invertido (min):', min_value=1, max_value=1440, value = 1, step = 1)
  Tipo = st.number_input("Tipo <br> 0.Entretenimiento ocio <br> 1.Ahorro inversion <br> 2.Ejercicio Deporte <br> 3.Alimentos salud <br> 4.Transporte <br> 5.Academico:", min_value=0, max_value=5, value = 0, step = 1)
  Momento = st.number_input("Momento <br> 0.mañana <br> 1.tarde <br> 2.noche:",min_value=0, max_value=2, value = 0, step = 1)
  No_de_personas = st.number_input('No. de personas', min_value=1, max_value=500, value = 1, step = 1)

  user_input_data = {'Presupuesto': Presupuesto,
                     'Tiempo invertido': Tiempo_invertido,
                     'Tipo': Tipo,
                     'Momento': Momento,
                     'No. de personas': No_de_personas}


  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

df2 =  pd.read_csv('df2.csv', encoding='latin-1')
X = df2.drop(columns='Costo')
y = df2['Costo']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613080)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['Presupuesto'] + b1[1]*df['Tiempo invertido'] + b1[2]*df['Tipo'] + b1[3]*df['Momento'] + b1[4]*df['No. de personas']

st.subheader('Cálculo del costo')
st.write('Tu siguiente gasto será de: ', prediccion)
