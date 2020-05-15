# Code-Challenge
Este es el repositorio de respuesta al codeChallenge. 
El modelo propuesto recibe people.in donde existe información sobre el nombre, la identificación, el país, el rol en la empresa, la industria, el número de recomendaciones y el número de conexiones. El lenguaje seleccionado fue Python y utilizamos las librerías de manejo de DataFrames panda y de Machine Learning scikit-learn.
Los pasos usados en el modelo están mencionados a continuación:
1. A partir de los datos obtenidos se realiza una revisión de los datos y se rellenan los datos faltantes. Es posible realizarlo para este ejemplo en específico pues la manera en que se asignan los valores no interfieren en el modelo matemático.
2. Se utilizan una lista de palabras clave que servirán como indicadores de jerarquía en rol y sector de la empresa. Se les asigna un valor numérico a una columna creada para dar un valor numérico a las industrias y roles que contiengan las palabras clave.
3. Se utiliza tSNE para los valores numéricos.
4. Se identifican los Clusters y sus características referentes a los features.
5. Se seleccionan los clusters relevantes para el problema y se agregan sus datos organizados a un dataframe.
6. Se exportan los primeros 100 datos del dataframe.
El archivo de salida es people.out
