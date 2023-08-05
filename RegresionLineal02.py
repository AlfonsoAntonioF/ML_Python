## infromacion cruzada 

import os
import pandas as pd
import numpy as np
import sklearn as sk
from plotnine import *
from siuba import *
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split # para separacion de los datos


ruta = "C:/Users/Usuario/Desktop/ML_Python"
os.chdir(ruta)
mi_tabla = pd.read_csv("casas_boston.csv")
print(mi_tabla.head())
'''
En este ejemplo se quiere predecir la variable MEDV en funcion de las demas, 
es decir, se va a intentar predecir el precio de la casa 

'''
variables_independientes = mi_tabla >> select(-_.MEDV,-_.RAD)# se selecciona las columnas que se utilizaran como predictores
objetivo = mi_tabla >> select(_.MEDV)# se selecciona la variable a predecir 

# se empieza a generar el modelo de regresion

modelo_regresion = LinearRegression()
modelo_regresion.fit(X=variables_independientes,y=objetivo)# Se crea el modelo de regresion

mi_tabla = mi_tabla >> mutate(predicciones = modelo_regresion.predict(variables_independientes)) >> select(-_.RAD)

print(mi_tabla.columns)
''' 
MUY IMPORTANTE
En este momento mi_tabla es la tabla original PERO CON UNA COLUMNA EXTRA: LA DE LAS PREDICCIONES DE LA COMPUTADORA
'''

'''Función para evaluar el modelo. Sus argumentos son:
    - independientes: tabla de columnas predictoras (es la tabla azul)
    - nombre_columna_objetivo(nco): es el nombre de la columna objetivo de la tabla original
    - tabla_full: es la tabla completa del comentario anterior'''

def evaluar_regresion(independientes,nco,tabla_full):
    n = independientes.shape[0]# numero de filas
    k = independientes.shape[1]# comlejidad = num columnas
    mae = metrics.mean_absolute_error(tabla_full[nco],tabla_full["predicciones"])# error absoluto medio
    rmse = np.sqrt(metrics.mean_squared_error(tabla_full[nco],tabla_full["predicciones"]))# la desviacion estandar del precio de la casa
    r2 = metrics.r2_score(tabla_full[nco],tabla_full["predicciones"])
    r2_adj = 1-(1-r2)*(n-1)/(n-k-1) # R2 ajustado
    return {"r2_adj":r2_adj,"mae":mae,"rmse":rmse}
    
    
EVALUACION = evaluar_regresion(variables_independientes,"MEDV",mi_tabla)# Se evalua el modelo de regresion
print(EVALUACION,'\n')
print('---- SEPARACION DE DATOS----')
'''
###############################################################################
################                                          #####################
################ SEPARACION EN ENTRENAMIENTO Y PRUEBA     #####################
################                                          #####################
###############################################################################
'''

'''Dividiemos en entrenamiento y prueba. El 33% de los datos es para prueba y
utilizamos una semilla igual a 13'''

indepen_entrenamiento, indepen_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(variables_independientes,
                                                                                                  objetivo,
                                                                                                  test_size=0.33,# tamaño de la prueba 
                                                                                                  random_state=13)
 
 
indepen_entrenamiento.shape[0]
objetivo_entrenamiento.shape[0]
indepen_prueba.shape[0]
objetivo_prueba.shape[0]


mi_tabla_entrenamiento = indepen_entrenamiento >> mutate(objetivo = objetivo_entrenamiento)
mi_tabla_prueba = indepen_prueba >> mutate(objetivo = objetivo_prueba)

modelo_entrenamiento = LinearRegression()

modelo_entrenamiento.fit(X=indepen_entrenamiento,y=objetivo_entrenamiento)

mi_tabla_entrenamiento = mi_tabla_entrenamiento >> mutate(predicciones = modelo_entrenamiento.predict(indepen_entrenamiento))


print(mi_tabla_entrenamiento.columns)

evaluar_regresion(indepen_entrenamiento,"objetivo",mi_tabla_entrenamiento)


mi_tabla_prueba = mi_tabla_prueba >> mutate(predicciones = modelo_entrenamiento.predict(indepen_prueba))
Evaluacion = evaluar_regresion(indepen_prueba,"objetivo",mi_tabla_prueba)
print(Evaluacion)

Resultados = {}
Resultados["tabla_original"] = evaluar_regresion(variables_independientes,"MEDV",mi_tabla)
Resultados["tabla_entrenamiento"] = evaluar_regresion(indepen_entrenamiento,"objetivo",mi_tabla_entrenamiento)
Resultados["tabla_prueba"] = evaluar_regresion(indepen_prueba,"objetivo",mi_tabla_prueba)

Resultados = pd.DataFrame(Resultados)
print(Resultados)


'''Cambiando random_state a 42'''

indepen_entrenamiento, indepen_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(variables_independientes,
                                                                                                                objetivo,
                                                                                                                test_size=0.33,
                                                                                                                random_state=42)

mi_tabla_entrenamiento = indepen_entrenamiento >> mutate(objetivo = objetivo_entrenamiento)
mi_tabla_prueba = indepen_prueba >> mutate(objetivo = objetivo_prueba)

modelo_entrenamiento = LinearRegression()
modelo_entrenamiento.fit(X=indepen_entrenamiento,y=objetivo_entrenamiento)
mi_tabla_entrenamiento = mi_tabla_entrenamiento >> mutate(predicciones = modelo_entrenamiento.predict(indepen_entrenamiento))
mi_tabla_prueba = mi_tabla_prueba >> mutate(predicciones = modelo_entrenamiento.predict(indepen_prueba))

Resultados = {}
Resultados["tabla_original"] = evaluar_regresion(variables_independientes,"MEDV",mi_tabla)
Resultados["tabla_prueba"] = evaluar_regresion(indepen_prueba,"objetivo",mi_tabla_prueba)
Resultados["tabla_entrenamiento"] = evaluar_regresion(indepen_entrenamiento,"objetivo",mi_tabla_entrenamiento)


Resultados = pd.DataFrame(Resultados)
print(Resultados)

'''
###############################################################################
################                                          #####################
#########################   Validación cruzada      ###########################
################                                          #####################
###############################################################################
'''

from sklearn.model_selection import cross_val_score



modelo_regresion_validacion = LinearRegression()

variables_independientes = mi_tabla >> select(-_.MEDV,-_.predicciones)# se selecciona las columnas que se utilizaran como predictores
objetivo = mi_tabla >> select(_.MEDV)

cross_val_score(modelo_regresion_validacion,
                variables_independientes,
                objetivo,
                scoring = "neg_root_mean_squared_error",
                cv=10)

rmse_validacion = [-cross_val_score(modelo_regresion_validacion,
                variables_independientes,
                objetivo,
                scoring = "neg_root_mean_squared_error",
                cv=x).mean() for x in range(10,150) 
]


evaluacion_cruzada = {"particiones":list(range(10,150)),
                      "rmse_validacion":rmse_validacion}

evaluacion_cruzada = pd.DataFrame(evaluacion_cruzada)
print(evaluacion_cruzada)
grafica = (ggplot(data = evaluacion_cruzada) +
 geom_line(mapping=aes(x="particiones",y="rmse_validacion")) 
 )
print(grafica)