##### Regresiones Lineales #####
## Metodos de Machine Learning con python ##

## Paqueterias a utilizar ##
import os
import pandas as pd
import numpy as np
import sklearn as sk
from plotnine import *
from siuba import *
from sklearn.linear_model import LinearRegression
from sklearn import metrics


#ruta = 'https://raw.githubusercontent.com/scidatmath2020/ML_Py_23/main/data/datos_regresion.csv'
ruta = "C:/Users/Usuario/Desktop/ML_Python"
os.chdir(ruta)## importar la base
mi_tabla = pd.read_csv('datos_regresion.csv')
print(mi_tabla.head())
## Seleccionar las columnas que se van a utilizar ##
print(mi_tabla.columns)

grafica = (ggplot(data = mi_tabla) +
 geom_point(mapping=aes(x='caracteristica_1', y='valor_real'),color = 'red') 
)
grafica.save('Grafica')
print(grafica)

variables_independientes = mi_tabla >> select(_.caracteristica_1)
objetivo = mi_tabla >> select(_.valor_real)

## Regresion Lineal del Modelo 

Modelo = LinearRegression()
Modelo.fit(X = variables_independientes,y =  objetivo)# Ajusta el modelo 
print(Modelo.coef_)## el coeficiente betha del modelo
print(Modelo.intercept_)## el el alpha del modelo

alpha = Modelo.intercept_
betha = Modelo.coef_
mi_tabla['Prediccion'] = Modelo.predict(variables_independientes)## Agregamos las predicciones al dataframe
print(mi_tabla.head())

grafica2 = (ggplot(data = mi_tabla) +
            geom_point(mapping=aes(x='caracteristica_1', y='valor_real'),color = 'blue') 
            + geom_point(mapping=aes(x='caracteristica_1', y='Prediccion'),color = 'green') 
            + geom_abline(slope=1.85, intercept=5.711)
)
grafica2.save('Grafica2')
print(grafica2)

error1 = metrics.mean_squared_error(objetivo, Modelo.predict(variables_independientes))
error2 = metrics.mean_absolute_error(objetivo, Modelo.predict(variables_independientes))
error2 = np.sqrt(error2)
print(error1, error2)
#mi_tabla['Error'] = variables_independientes-Modelo.predict(variables_independientes)
#print(mi_tabla.head())

mi_tabla >> mutate(Error =_.valor_real-_.Prediccion)
R2 = metrics.r2_score(objetivo, Modelo.predict(variables_independientes))

print(R2)
CoefDeterminacion = 1-((1-R2)*(50-1)/(50-1-1))
print(CoefDeterminacion)