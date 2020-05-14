'''Importamos los paquetes a utilizar'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

'''Preprocesamiento de datos'''
#Las llaves utilizadas son las proporcionadas en el documento
keys = ['PersonId', 'Name', 'LastName', 'CurrentRole', 'Country', 'Industry', 'NumberOfRecommendations',
'NumberOfConnections']
data=pd.read_csv('people.in', delimiter = '|', names=keys)


'''Preprocesamos los datos. Iniciamos por llenar los valores nulos de cada columna teniendo en cuenta los tipos de estas'''
#Nota: Para este ejercicio en especial no tenemos valores nulos para ciertas columnas, pero vale la pena tener en cuenta rellenar valores nulos para futuras ocasiones
#Lista que contiene los valores por los que reemplazo 
fill=[000000,' ', ' ', ' ', ' ',0,0]

#Reemplazamos los valores que se encuentran vacíos por 0 para número de recomendaciones y conecciones para que no sumen
#Para Name, Lastname, Currentrole y Country reemplazamos la falta de datos por espacios 
for llave, f in zip(keys,fill):
    data[llave].fillna(f, inplace = True)

'''A partir de los datos y una revisión de primera mano exiten profesiones con una baja probabilidad de requerir
 nuestros servicios y una cantidad de ellas que sí pueden estar interesados en obtenerlos. De manera que 
 realizamosun filtro  con una lista de palabras clave que cuentan con un puesto suficientemente alto como
 para tener poder de decisión sobre los servicios que en su área la empresa puede tomar. '''


 #Lista de palabras clave en currentJob
Up_roles=['ceo', 'chief',  'president', 'manager', 'director', 'head', 'vp', 'owner', 'board', 'chair']

'''Existe mayor probabilidad (según) los principales clientes de la página web de BairesDev relacionados con
software y electrónica, de manera que realizo una lista con palabras relacionadas'''
#Lista de palabras clave en Industry
Up_Industries=['Develop', 'Investigation', 'Telecommunication', 'Internet', 'Tech', 'compu', 'Software', 'Electro', 'Information', 'Intelligence', 'digital']


#Introducimos una nueva variable que represente el puntaje para rol e industria para utilizar  
data['ScoreRole']=0
data['ScoreIndus']=0
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Si algún perfil tiene en su CurrentJob alguna palabra clave obtiene un puntaje dado
for role,indus in zip(Up_roles, Up_Industries):
  data['ScoreRole'][pd.Series(data['CurrentRole']).str.contains(role)]+=10
  data['ScoreIndus'][pd.Series(data['Industry']).str.contains(indus)]+=10

data = data[data['ScoreRole']>0]

'''Realizamos t-Distributed Stochastic Neighbor Embedding (t-SNE) con el modelo cuantitativo obtenido anteriormente'''

#solamente utilizaremos los datos cuantitativos
cuanti=data.iloc[:,6:]

#t-SNE

x = StandardScaler().fit_transform(cuanti)
tsne = TSNE(random_state=1)
ts = tsne.fit_transform(x)
tsne = pd.DataFrame(data = ts
             , columns = ['1', '2'])
#plt.plot(tsne.iloc[:,0],tsne.iloc[:,1], 'ro')


'''Diferencio cada espacio utilizando clustering y exploro los datos'''
#Número de clusters
numclus=10
#Clustering
clusterer = AgglomerativeClustering(n_clusters=numclus)
y_pred = clusterer.fit_predict(x)

#La siguiente linea muestra la visualización de los clusters. 
#plt.scatter(tsne.iloc[:,0],tsne.iloc[:,1], c=y_pred)

#Creo una columna auxiliar para poder obtener el índice de los que pertenecen a un grupo específico
tsne['grupo']= y_pred
#Asigno a la columna Group de data el grupo
data['Group']=0
for i in range (numclus):
  index=tsne[tsne['grupo']==i].index
  data.iloc[index,10]=i  


#Nuestro orden de interés luego de explorar las características en común de cada cluster es es 4,9,5,8,6,3,0,7,1,2
inter=[4,9,5,8,6,3,0,7,1,2]

#Se crea un dataframe con los datos del grupo 4 que es el de mejor "rendimiento" y agrego en orde los grupos de interés
sal=pd.DataFrame(data[data['Group']==4])
for i in range (len(inter)):
  gr=data[data['Group']==inter[i]]
  gr=gr.sort_values(by=['ScoreRole','ScoreIndus'], ascending=[False,False])
  if (i!=0):
    sal=sal.append(gr)
    
#Creo un dataframe con las columnas iniciales dadas
out=pd.DataFrame(sal.iloc[:100, :8])
expor=out.iloc[:100,0]
#Exporto el archivo con solamente las ID.
expor.to_csv('people.out',index=False, header= False)
#print((expor))
