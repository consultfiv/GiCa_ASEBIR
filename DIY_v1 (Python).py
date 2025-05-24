#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


"hola mundo"


# ## Sección 1: Instalación y carga de paquetes

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


# ## Sección 2: Cargar el dataset
# 
# Usaremos el dataset de diabetes de sklearn:

# In[5]:


from sklearn.datasets import load_diabetes
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diabetes'] = data.target


# Vamos a ver las primeras 5 filas del dataset cargado

# In[7]:


print(df.head())


# ## Sección 3: Definir parámetros generales

# In[9]:


OUTPUT = "diabetes"
LEVEL1 = 1  # Tiene diabetes
LEVEL0 = 0  # No tiene diabetes


# ## Sección 4: Preparación de datos

# Para saber el tipo de variables que tenemos,se usa la función *print(df.dtypes)*, pero como solo queremos saber el tipo de variable de **diabetes**, usamos 
# 
# print(df.['diabetes'].dtypes)

# In[12]:


print(df['diabetes'].dtypes) #siempre hay que poner '' en el nombre


# Tenemos que convertir una variable continua (float) en una categórica. 
# 
# Para ello, utilizamos la función *pd.cut()*, que nos divide los valores en 2 categorías. 
# 
# Usaremos *bins = 2* para crear dos intervalos.
# 
# También usaremos *labels = [LEVEL0, LEVEL1]*, que nos asigna etiquetas a los dos intervalos que hemos creado con bins.
# 
# El resultado lo vamos a almacenar en *df[OUTPUT]*, lo que reemplaza los valores originales.

# In[14]:


df[OUTPUT] = pd.cut(df[OUTPUT], bins=2, labels=[LEVEL0, LEVEL1])


# Si ahora volvemos a preguntar qué tipo de variable es diabetes, nos debe aparecer *category*

# In[16]:


print(df['diabetes'].dtypes)


# Vamos a dividir nuestro modelo en entrenamiento y prueba. Para ello: 
# 1. Creamos una variable (X) que es una copia del Dataframe inicial (df) y **quitamos** la columna OUTPUT (diabetes): *X = df.drop(OUTPUT, axis = 1)*
# 2. Creamos otra variable (Y) que contiene **SOLO** la columna OUTPUT del df inicial: *y = df[OUTPUT]*
# 3. Dividimos los datos en entrenamiento y prueba con esta función: *X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)*, donde:
# 
#        - X_train y y_train: datos de entrenamiento (80% de los datos).
#    
#        - X_test y y_test: datos de prueba (20% de los datos, ya que test_size=0.2).
#    
#        - random_state=123 fija la semilla para que sea reproducible.
# 
# Esta operación prepara los datos para entrenar y evaluar un modelo de aprendizaje automático.

# In[18]:


X = df.drop(OUTPUT, axis=1)
y = df[OUTPUT]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# ## Sección 5: Entrenamiento del modelo

# Vamos a crear y entrenar un modelo de Random Forest:
# 
# 1º Creamos el modelo Random Forest Classifier con la fórmula *rf = RandomForestClassifier(n_estimators=1000, random_state=123)* donde 
# 
#         n_estimators = 1000 especifica que tiene 1000 árboles de decisión 
#         random_state = 123 fija la semilla para poder reproducirlo
# 
# 2º Entrenamos el modelo con los datos con la fórmula *rf.fit(X_train, y_train)*, donde
# 
#         X_train son las características de entrenamiento 
#         y_train son las etiquetas de entrenamiento
# 
# La fórmula quedaría así: 
#         

# In[21]:


rf = RandomForestClassifier(n_estimators=1000, random_state=123)

rf.fit(X_train, y_train)


# Ahora hacemos las prediciones en el test con la fórmula
# 
# **y_pred_proba = rf.predict_proba(X_test)[:, 1]**
# 
# **y_pred = (y_pred_proba > 0.5).astype(int)**
# 
# 
# donde *rf.predict_proba(X_test)* calcula las probabilidades de predicción para cada clase usando el modelo Random Forest en los datos de prueba; 
# 
# y *[:, 1]* selecciona la segunda columna que corresponde a la probabilidad de tener diabetes (la de Level 1).
# 
# 
# 
# 
# donde *y_pred_proba > 0.5* crea un booleando donde True indica que hay una probabilidad > 0.5; 
# 
# y *astype(int)* transforma True en 1 y False en 0.
# 

# In[23]:


y_pred_proba = rf.predict_proba(X_test)[:, 1]

y_pred = (y_pred_proba > 0.5).astype(int)


# In[39]:


# Matriz de confusión
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_mat)

# Métricas
report = classification_report(y_test, y_pred, output_dict=True)
metrics_total = pd.DataFrame({
    'F1Score': [report['1']['f1-score']],
    'Sens': [report['1']['recall']],
    'Spec': [report['0']['recall']],
    'PPV': [report['1']['precision']],
    'NPV': [report['0']['precision']]
})


# ## Sección 6: Análisis de los resultados

# In[42]:


# Calcular curva ROC y AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Añadir AUC al dataframe de métricas
metrics_total['AUC'] = roc_auc

# Crear el gráfico de la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='#008CBD', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Especificidad')
plt.ylabel('Sensibilidad')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Importancia de variables
importances = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
importances = importances.sort_values('importance', ascending=False)
print("Variable Importance:")
print(importances)

# Visualizar importancia de variables
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importances)
plt.title('Importancia de Variables')
plt.show()

# Mostrar métricas finales
print("\nMétricas Finales:")
print(metrics_total)

