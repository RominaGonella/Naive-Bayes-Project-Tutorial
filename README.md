# Resumen del proceso

1. En el primer paso se importan las librerías y se cargan los datos desde la web. El dataset contiene 891 reviews de apps en Google Play, el objetivo es crear un modelo Naive Bayes para predecir si el review es positivo o negativo interpretando el texto.
2. Se realiza un breve preprocesamiento quitando una variable que no se necesita, sacando espacios y pasando texto a minúscula.
3. Se separan los datos en muestras de entrenamiento (train) y control o evaluación (test) estratificando la variable target.
4. Se aplica la función *CounterVectorizer* para transformar la variable de texto a numérico, creando una columna para cada palabra distinta. Se eliminan las palabras que no aportan usando como referencia el idioma inglés.
5. Se manipula el diccionario de palabras que devuelve la función, creando un data frame con la frecuencia de cada palabra. Se intentó indentificar qué palabra representa cada columna de X, pero no se logró. No lo pedía el ejercicio pero me pareción interesante explorarlo.
6. Se ajusta un modelo NaiveBayes de tipo multinomial.
7. Se prueba el modelo con algunos textos arbitrarios.
8. Se guarda el modelo en la carpeta 'models'.