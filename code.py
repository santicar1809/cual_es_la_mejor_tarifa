# %% [markdown]
# # ¿Cuál es la mejor tarifa?
# 
# Trabajas como analista para el operador de telecomunicaciones Megaline. La empresa ofrece a sus clientes dos tarifas de prepago, Surf y Ultimate. El departamento comercial quiere saber cuál de las tarifas genera más ingresos para poder ajustar el presupuesto de publicidad.
# 
# Vamos a realizar un análisis preliminar de las tarifas basado en una selección de clientes relativamente pequeña. Tendemos los datos de 500 clientes de Megaline: quiénes son los clientes, de dónde son, qué tarifa usan, así como la cantidad de llamadas que hicieron y los mensajes de texto que enviaron en 2018. Nuestro trabajo es analizar el comportamiento de los clientes y determinar qué tarifa de prepago genera más ingresos.

# %% [markdown]
# El propósito de este proyecto, es análizar cual es el plan mobil más rentable, para esto, vamos a análizar una serie de dataframes que incluyen la información de cada usuario, en cuanto a consumo y tarifas, vamos a análizar cada dataframe, haciendo limpieza de datos, eliminando ausentes y duplicados, posteriormente vamos a identificar que información nos es útil para el análisis estadistico, y posteriormente uniremos los dataframes con dicha información importante. Posteriormente haremos el análisis mediante pruebas de hipotesis y análisis exploratorio de datos.

# %% [markdown]
# ## 1. Inicialización

# %%
# Cargar todas las librerías
import pandas as pd
import numpy as np
import math
from scipy import stats as st
import matplotlib.pyplot as plt


# %% [markdown]
# ## Cargar datos

# %%
# Carga los archivos de datos en diferentes DataFrames
df_calls=pd.read_csv('datasets/megaline_calls.csv')
df_internet=pd.read_csv('datasets/megaline_internet.csv')
df_messages=pd.read_csv('datasets/megaline_messages.csv')
df_plans=pd.read_csv('datasets/megaline_plans.csv')
df_user=pd.read_csv('datasets/megaline_users.csv')


# %% [markdown]
# ## 2. Preparar los datos

# %%
print(df_calls.head())
print(df_calls.info())
print()
print(df_internet.head())
print(df_internet.info())
print()
print(df_messages.head())
print(df_messages.info())
print()
print(df_plans.head())
print(df_plans.info())
print()
print(df_user.head())
print(df_user.info())

df_calls['user_id']=df_calls['user_id'].astype('str')
df_internet['user_id']=df_internet['user_id'].astype('str')
df_messages['user_id']=df_messages['user_id'].astype('str')
df_user['user_id']=df_user['user_id'].astype('str')

# %% [markdown]
# Pasamos la columna 'user_id' a datos de tipo string, debido a que este numero identifica a cada usuario, pero no es necesario hacer cálculos con este numero y puede generarnos un error.

# %% [markdown]
# ## Identificación de ausentes y duplicados.

# %%
print('Ausentes llamadas:\n', df_calls.isna().sum())
print('Ausentes internet:\n',df_internet.isna().sum())
print('Ausentes mensajes:\n',df_messages.isna().sum())
print('Ausentes planes:\n',df_plans.isna().sum())
print('Ausentes usuarios:\n',df_user.isna().sum())

# %%
print('Duplicados llamadas:\n', df_calls.duplicated().sum())
print('Duplicados internet:\n',df_internet.duplicated().sum())
print('Duplicados mensajes:\n',df_messages.duplicated().sum())
print('Duplicados planes:\n',df_plans.duplicated().sum())
print('Duplicados usuarios:\n',df_user.duplicated().sum())

# %%
print('Duplicados llamadas:\n', df_calls['id'].duplicated().sum())
print('Duplicados internet:\n',df_internet['id'].duplicated().sum())
print('Duplicados mensajes:\n',df_messages['id'].duplicated().sum())

# %% [markdown]
# ## Tarifas

# %%
# Imprime la información general/resumida sobre el DataFrame de las tarifas

print(df_plans.info())

# %%
# Imprime una muestra de los datos para las tarifas
print(df_plans.head())


# %% [markdown]
# Podemos ver las características de cada plan en gigas y costo, además que los datos están correctos. No tenemos que editar nada y la información está completa.

# %% [markdown]
# ## Enriquecer los datos

# %%
df_plans['gb_per_month_included']=[15,30]
df_plans

# %% [markdown]
# Agregamos las gigas por mes que incluye cada plan.

# %% [markdown]
# ## Usuarios/as

# %%
# Imprime la información general/resumida sobre el DataFrame de usuarios
df_user.info()


# %%
# Imprime una muestra de datos para usuarios
df_user.sample(10)



# %% [markdown]
# Podemos ver que los datos están bien, sin embargo, los ausentes de la columna 'churn_date' los podemos reemplazar por la palabra 'actual', para identificar que aún están suscritos al plan.

# %% [markdown]
# ### Corregir los datos

# %%
df_user['churn_date']=df_user['churn_date'].fillna('actual')
df_user.sample(10)

# %% [markdown]
# Los datos se ven bien, no tenemos valores ausentes y la información está completa.

# %% [markdown]
# ## Enriquecer los datos

# %%
df_user['reg_date']=pd.to_datetime(df_user['reg_date'])
df_user['month']=df_user['reg_date'].dt.month
df_user.head(10)

# %% [markdown]
# Agregamos la columna 'month' al data frame para los análisis.

# %% [markdown]
# ## Llamadas

# %%
# Imprime la información general/resumida sobre el DataFrame de las llamadas
df_calls.info()


# %%
# Imprime una muestra de datos para las llamadas
df_calls.sample(10)


# %% [markdown]
# Los datos se ven bien, no tenemos valores ausentes y la información está completa.

# %% [markdown]
# ## Enriquecer los datos

# %%
df_calls['call_date']=pd.to_datetime(df_calls['call_date'])
df_calls['month']=df_calls['call_date'].dt.month
df_calls.head(10)

# %% [markdown]
# Agregamos la columna 'month' al data frame para los análisis.

# %% [markdown]
# ## Mensajes

# %%
# Imprime la información general/resumida sobre el DataFrame de los mensajes
df_messages.info()


# %%
# Imprime una muestra de datos para los mensajes
df_messages.sample(10)


# %% [markdown]
# Los datos se ven bien, no tenemos valores ausentes y la información está completa.

# %% [markdown]
# ## Enriquecer los datos

# %%
df_messages['message_date']=pd.to_datetime(df_messages['message_date'])
df_messages['month']=df_messages['message_date'].dt.month
df_messages.head(10)

# %% [markdown]
# Agregamos la columna 'month' al data frame para los análisis.

# %% [markdown]
# ## Internet

# %%
# Imprime la información general/resumida sobre el DataFrame de internet
df_internet.info()


# %%
# Imprime una muestra de datos para el tráfico de internet
df_internet.sample(10)


# %% [markdown]
# Los datos se ven bien, no tenemos valores ausentes y la información está completa.

# %% [markdown]
# ## Enriquecer los datos

# %%
df_internet['session_date']=pd.to_datetime(df_internet['session_date'])
df_internet['month']=df_internet['session_date'].dt.month
df_internet.head(10)

# %% [markdown]
# Agregamos la columna 'month' al data frame para los análisis.

# %% [markdown]
# ## Estudiar las condiciones de las tarifas

# %%
# Imprime las condiciones de la tarifa y asegúrate de que te quedan claras
df_plans.head()


# %% [markdown]
# ## Agregar datos por usuario

# %%
# Calcula el número de llamadas hechas por cada usuario al mes. Guarda el resultado.
calls_per_month=df_calls.groupby(['user_id','month'],as_index=False)['id'].count()
calls_per_month.head(10)
calls_per_month.columns=['user_id','month','calls']
calls_per_month

# %%
# Calcula la cantidad de minutos usados por cada usuario al mes. Guarda el resultado.
#Se redondea antes del groupby
df_calls['duration']=np.ceil(df_calls['duration'])
minutes_per_month=df_calls.groupby(['user_id','month'],as_index=False)['duration'].sum()
minutes_per_month.columns=['user_id','month','minutes']
minutes_per_month.head(10)

# %%
# Calcula el número de mensajes enviados por cada usuario al mes. Guarda el resultado.
messages_per_month=df_messages.groupby(['user_id','month'],as_index=False)['id'].count()
messages_per_month.columns=['user_id','month','messages']
messages_per_month.head(10)


# %%
# Calcula el volumen del tráfico de Internet usado por cada usuario al mes. Guarda el resultado.
internet_per_month=df_internet.groupby(['user_id','month'],as_index=False)['mb_used'].sum()
internet_per_month['gb_used']=(internet_per_month['mb_used']/(1024))
internet_per_month['gb_used']=np.ceil(internet_per_month['gb_used'])
internet_per_month.head(10)


# %% [markdown]
# Fusionamos todos los dataframes en uno llamado 'df_merged'

# %%
# Fusiona los datos de llamadas, minutos, mensajes e Internet con base en user_id y month
df_merged=calls_per_month.merge(minutes_per_month.merge(messages_per_month.merge(internet_per_month,how='outer'),how='outer'),how='outer')
df_merged=df_merged.fillna(0)
df_merged.head(10)


# %%
# Añade la información de la tarifa
df_plans


# %%
# Calcula el ingreso mensual para cada usuario
df_name_plan=df_user[['user_id','plan','city']]
df_merged=df_merged.merge(df_name_plan)
df_merged

# %%
def cero(ingreso):
    if ingreso<0:
        ingreso=0

    return ingreso

# %% [markdown]
# Calculamos los ingresos itilizando la función where.

# %%
df_merged=df_merged.merge(df_plans,right_on='plan_name',left_on='plan')

# %%
df_merged=df_merged.drop(columns='plan_name')

# %%
df_merged['minutes_total']=(df_merged['minutes']-df_merged['minutes_included'])
df_merged['messages_total']=df_merged['messages']-df_merged['messages_included']
df_merged['gb_total']=df_merged['gb_used']-df_merged['gb_per_month_included']



# %%
df_merged.head(10)

# %%
df_merged['minutes_total']=df_merged['minutes_total'].apply(cero)
df_merged['messages_total']=df_merged['messages_total'].apply(cero)
df_merged['gb_total']=df_merged['gb_total'].apply(cero)

# %%
df_merged.info()

# %%
df_merged.head(10)

# %%
df_merged['ingreso']=0
df_merged['ingreso']=df_merged['ingreso'].where((df_merged['plan']!='surf'),(df_merged['minutes_total']*0.03)+(df_merged['messages_total']*0.03)+((df_merged['gb_total'])*10)+20)
df_merged['ingreso']=df_merged['ingreso'].where((df_merged['plan']=='surf'),(df_merged['minutes_total']*0.01)+(df_merged['messages_total']*0.01)+((df_merged['gb_total'])*7)+70)
df_surf=df_merged[df_merged['plan']=='surf']
df_ult=df_merged[df_merged['plan']=='ultimate']
df_surf['ingresos']=df_merged['ingreso'].where((df_merged['ingreso']>0),20)
df_ult['ingresos']=df_merged['ingreso'].where((df_merged['ingreso']>0),70)


# %% [markdown]
# Creamos otro dataframe conjunto para unir los resultados del plan surf y el plan ultimate mediante concat()

# %%
df_merged2=pd.concat([df_surf,df_ult],axis=0)
df_merged2

# %%
df_merged[df_merged['plan']=='ultimate'].tail(20)

# %% [markdown]
# ## 3. Estudia el comportamiento de usuario

# %% [markdown]
# Calculamos el consumo de cada usuario por mes, de llamadas, minutos, mensajes y datos.

# %% [markdown]
# ### Llamadas

# %%
# Compara la duración promedio de llamadas por cada plan y por cada mes. Traza un gráfico de barras para visualizarla.
mean_surf=df_merged[df_merged['plan']=='surf'].groupby(['month'],as_index=False)['minutes'].mean()
mean_ultimate=df_merged[df_merged['plan']=='ultimate'].groupby(['month'],as_index=False)['minutes'].mean()


# %%
print(mean_surf)
print(mean_ultimate)

# %%
mean_minutes=mean_surf.merge(mean_ultimate, on='month')
mean_minutes.columns=['month','minutes_surf','minutes_ultimate']
mean_minutes.plot(kind='bar', 
           x='month',
           y=['minutes_surf','minutes_ultimate'],
           xlabel='Mes',
           ylabel='Minutos',
           title='Promedio de minutos por mes',
            figsize=[12,5],legend=True)
plt.show()

# %%
# Compara el número de minutos mensuales que necesitan los usuarios de cada plan. Traza un histograma.
df_surf['minutes'].plot(kind='hist',alpha=0.5)
df_ult['minutes'].plot(kind='hist',alpha=0.3)
plt.legend(['surf','ultimate'])
plt.show()

# %%
# Calcula la media y la varianza de la duración mensual de llamadas.
mean_surf=df_surf['minutes'].mean()
mean_ult=df_ult['minutes'].mean()
var_surf=np.var(df_surf['minutes'])
var_ult=np.var(df_ult['minutes'])
print('Media surf: ',mean_surf,'Media ultimate: ',mean_ult,'Varianza surf: ',var_surf,'Varianza ultimate: ',var_ult)

# %%
# Traza un diagrama de caja para visualizar la distribución de la duración mensual de llamadas
plt.boxplot(mean_minutes[['minutes_surf','minutes_ultimate']], vert=False,labels=['minutes_surf','minutes_ultimate'])
#plt.boxplot(df_ult['minutes'])
plt.show()

# %% [markdown]
# Podemos evidenciar un comportamiento parecido de cada plan, de cara a las llamadas, debido a que, su media es similar, y su distribución también. El comportamiento durante el año es creciente, viendo que en enero se hacen menos llamadas y el numero va creciendo, llegando al máximo en diciembre. Adicionalmente, en ninguno de los planes se excede del límite de llamadas.
# 
# En cuanto a la media y la varianza, el comportamiento de los datos de cada plan es muy parecidos, con la característica de que el plan ultimate está más disperso que el plan surf.
# 
# Podemos agregar la presencia de valores atípicos de algunos usuarios que no consumen muchos minutos, lo cual es normal, debido a que es un servicio que no se usa mucho actualmente.

# %% [markdown]
# ### Mensajes

# %%
# Comprara el número de mensajes que tienden a enviar cada mes los usuarios de cada plan
mean_surf_messages=df_merged[df_merged['plan']=='surf'].groupby(['month'],as_index=False)['messages'].mean()
mean_ultimate_messages=df_merged[df_merged['plan']=='ultimate'].groupby(['month'],as_index=False)['messages'].mean()

# %%
mean_messages=mean_surf_messages.merge(mean_ultimate_messages, on='month')
mean_messages.columns=['month','messages_surf','messages_ultimate']
mean_messages.plot(kind='bar', 
           x='month',
           y=['messages_surf','messages_ultimate'],
           xlabel='Mes',
           ylabel='Mensajes',
           title='Promedio de mensajes por mes',
            figsize=[10,5],legend=True)
plt.show()

# %%
df_surf['messages'].plot(kind='hist',alpha=0.5)
df_ult['messages'].plot(kind='hist',alpha=0.3)
plt.legend(['surf','ultimate'])
plt.show()

# %%
mean_surf=df_surf['messages'].mean()
mean_ult=df_ult['messages'].mean()
var_surf=np.var(df_surf['messages'])
var_ult=np.var(df_ult['messages'])
print('Media surf: ',mean_surf,'Media ultimate: ',mean_ult,'Varianza surf: ',var_surf,'Varianza ultimate: ',var_ult)

# %%
plt.boxplot(mean_messages[['messages_surf','messages_ultimate']], vert=False,labels=['messages_surf','messages_ultimate'])
#plt.boxplot(df_ult['minutes'])
plt.show()

# %% [markdown]
# En el caso de los mensajes, sus comportamientos también son similares en ambos planes, destacando que en promedio, ninguno de los planes se pasa del límite mensual de mensajes de texto, adicionalmente, sus medias son similares y su distribución también.
# 
# Al igual que los minutos, los mnesajes  varían en la agrupación de los datos, debido a que los datos del plan ultimate están más lejos entre sí que los del plan surf, esto se evidenció calculando la varianza y mirando sus histogramas, podemos ver que los datos del plan últimate están más distribuidos.
# 
# Podemos decir también que los datos de ambos planes están sesgados hacia la derecha.

# %% [markdown]
# ### Internet

# %%
mean_surf_internet=df_merged[df_merged['plan']=='surf'].groupby(['month'],as_index=False)['gb_used'].mean()
mean_ultimate_internet=df_merged[df_merged['plan']=='ultimate'].groupby(['month'],as_index=False)['gb_used'].mean()

# %%
mean_internet=mean_surf_internet.merge(mean_ultimate_internet, on='month')
mean_internet.columns=['month','internet_surf','internet_ultimate']
mean_internet.plot(kind='bar', 
           x='month',
           y=['internet_surf','internet_ultimate'],
           xlabel='Mes',
           ylabel='Internet',
           title='Promedio de megabytes por mes',
            figsize=[10,5],legend=True)
plt.show()

# %%
df_surf['gb_used'].plot(kind='hist',alpha=0.5)
df_ult['gb_used'].plot(kind='hist',alpha=0.3)
plt.legend(['surf','ultimate'])
plt.show()

# %%
mean_surf=df_surf['gb_used'].mean()
mean_ult=df_ult['gb_used'].mean()
var_surf=np.var(df_surf['mb_used'])
var_ult=np.var(df_ult['mb_used'])
print('Media surf: ',mean_surf,'Media ultimate: ',mean_ult,'\nVarianza surf: ',var_surf,'Varianza ultimate: ',var_ult)

# %%
plt.boxplot(mean_internet[['internet_surf','internet_ultimate']], vert=False,labels=['internet_surf','internet_ultimate'])
#plt.boxplot(df_ult['minutes'])
plt.show()

# %% [markdown]
# En el caso de los datos de internet, los comportamientos son más variados, en el caso del plan surf, en los primeros meses no se supera el límite, sin embargo , a partír de la segunda mitad de año, todos los meses se sobrepasa el límite, por lo cual sale más rentable que los usuarios se inscriban a este plan, debido a que pagan más adicional. Por otro lado el plan ultimate, no sobrepasa el límite de datos.
# 
# Los datos tienen distribuciones normales, sin sesgo, y al igual que los anteriores análisis, la varianza del plan surf es menor que la del plan ultimate, por lo que los datos de ultimate están más distribuidos. 
# 
# Podemos evidenciar también la presencia de atipicos que no utilizan mucho internet, esto podria darse por la población mayor que podemos análizar en futuras ocasiones con la edad de los usuarios.

# %% [markdown]
# ## Ingreso

# %%
mean_surf_ingresos=df_surf.groupby(['month'],as_index=False)['ingresos'].mean()
mean_ultimate_ingresos=df_ult.groupby(['month'],as_index=False)['ingresos'].mean()

# %%
mean_ingresos=mean_surf_ingresos.merge(mean_ultimate_ingresos, on='month')
mean_ingresos.columns=['month','ingresos_surf','ingresos_ultimate']
mean_ingresos.plot(kind='bar', 
           x='month',
           y=['ingresos_surf','ingresos_ultimate'],
           xlabel='Mes',
           ylabel='Ingresos',
           title='Promedio de megabytes por mes',
            figsize=[10,15],legend=True)
plt.show()

# %%
df_surf['ingresos'].plot(kind='hist',alpha=0.3,bins=20)
df_ult['ingresos'].plot(kind='hist',alpha=0.5,bins=10)
plt.legend(['surf','ultimate'])
plt.show()

# %%
mean_surf=df_surf['ingresos'].mean()
mean_ult=df_ult['ingresos'].mean()
var_surf=np.var(df_surf['ingresos'])
var_ult=np.var(df_ult['ingresos'])
print('Media surf: ',mean_surf,'Media ultimate: ',mean_ult,'Varianza surf: ',var_surf,'Varianza ultimate: ',var_ult)

# %%
plt.boxplot(mean_ingresos[['ingresos_surf','ingresos_ultimate']], vert=False,labels=['ingresos_surf','ingresos_ultimate'])
plt.show()

# %% [markdown]
# En el caso de los ingresos, podemos ver una diferencia evidente en cuanto a los planes. Podemos ver que el plan ultimate, invierte mucho más que el plan surf, debido a su alto costo, por otro lado el plan surf, a pesar de la cantidad que gasta en servicios adicionales, no sobrepasa a los ingresos del plan ultimate.
# 
# Por otro lado, podemos ver que los dos planes tienen un sesgo a la derecha. 
# 
# En el caso de la varianza, es mayor en el plan surf, y podemos ver en el diagrama de caja y bigotes que los datos están más distribuidos que los del plan ultimate.

# %% [markdown]
# ## 4. Prueba las hipótesis estadísticas

# %% [markdown]
# Prueba la hipótesis de que son diferentes los ingresos promedio procedentes de los usuarios de los planes de llamada Ultimate y Surf.

# %% [markdown]
# h0: La media de ingresos del plan surf, es igual a la media de los ingresos del plan ultimate.
# 
# h1: La media de ingresos del plan surf, es diferente a la media de los ingresos del plan ultimate.
# 
# Indice de significancia del 5%

# %%
# Prueba las hipótesis nula
alpha=0.05
results1 =st.ttest_ind(df_surf['ingresos'],df_ult['ingresos'],equal_var=False) # tu código: prueba la hipótesis de que las medias de las dos poblaciones independientes son iguales

print('valor p:', results1.pvalue)

if (results1.pvalue<alpha): # tu código: compara los valores p obtenidos con el nivel de significación estadística):
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


# %% [markdown]
# Se rechaza la hipotesis nula, quiere decir que los ingresos de los dos planes son diferente en promedio.

# %% [markdown]
# Prueba la hipótesis de que el ingreso promedio de los usuarios del área NY-NJ es diferente al de los usuarios de otras regiones.

# %% [markdown]
# h0: La media de ingresos de la región de New York - Newark - Jersey City, es igual a la media de los ingresos de las demás regiones.
# 
# h1: La media de ingresos de la región de New York - Newark - Jersey City, es diferente a la media de los ingresos de las demás regiones.
# 
# Indice de significancia del 5%

# %%
df_new_york=df_merged2[df_merged2['city']=='New York-Newark-Jersey City, NY-NJ-PA MSA']  
df_outside=df_merged2[df_merged2['city']!='New York-Newark-Jersey City, NY-NJ-PA MSA']

# %%
# Prueba las hipótesis
results2 =st.ttest_ind(df_new_york['ingresos'],df_outside['ingresos'],equal_var=False) # tu código: prueba la hipótesis de que las medias de las dos poblaciones independientes son iguales

print('valor p:', results2.pvalue)

if (results2.pvalue<alpha): # tu código: compara los valores p obtenidos con el nivel de significación estadística):
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


# %% [markdown]
# Se rechaza la hipotesis nula, quiere decir que los ingresos de las dos regiones es diferente en promedio.

# %% [markdown]
# ## Conclusión general
# En conclusión, el plan más rentable para los usuarios es el plan surf, debido a que a pesar de que gastan un poco más del límite, este no excede los 70 dolares del otro plan ultimate, además en promedio, la cantidad de minutos que se usa es menor a 500, los mensajes que se usan es menor a 40 mensajes y las gigas de internet menores a 18 gb que se acercan más a los límites del plan surf.
# 
# En cuanto a los ingresos que más le aportan a la compañia, podemos decir que el plan ultimate, aporta más ingresos a la compañia por su alto costo, a pesa de que las personas de este plan no utilizan más del límite, el costo normal, aporta más que el costo del plan surf y sus distintas adiciones.


