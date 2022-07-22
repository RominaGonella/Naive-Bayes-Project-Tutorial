# primero ejecutar desde consola: pip install -r requirements.txt

# librerías a utilizar
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# datos
data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')

# elimino la primera variable
data = data.drop(columns = 'package_name')

# elimino espacios al inicio y al final, paso caracteres a minúscula
data['review'] = data['review'].str.strip().str.lower()

# separo en X e y
X = data['review']
y = data['polarity']

# separo en train y test, eligiendo proporcionalmente valores de y = 0 o 1 de acuerdo al dataset completo
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.25, random_state = 42)

# función para vectoriza la variable reviews (X)
vec = CountVectorizer(stop_words = 'english')

# se aplica la vectorización a X_train y X_test
X_train = vec.fit_transform(X_train).toarray()
X_test = vec.transform(X_test).toarray()

# ajusto modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# se guarda el modelo
filename = '../models/nb_model.sav'
pickle.dump(model, open(filename,'wb'))