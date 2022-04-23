from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np

# 0 - Computer  Science
# 1 - Electrical  Engineering
# 2 - Psychology
# 3 - Mechanical  Engineering
# 4 - Civil  Engineering
# 5 - Medical  Science
# 6 - Biochemistry

DATA_PATH = './data_WOS/WebOfScience/WOS11967'
Y_FILE = "YL1"
RANDOM_STATE = 42
labels = ['Computer  Science', 'Electrical  Engineering', 'Psychology',
          'Mechanical  Engineering', 'Civil  Engineering', 'Medical  Science', 'Biochemistry']

include_labels = [0, 1, 2, 3, 4, 5, 6]
# include_labels = [5, 6]  # similar categories
# include_labels = [0, 5]  # different categories

with open(f"{DATA_PATH}/X.txt", "r") as X_text_file:
    X = X_text_file.read().splitlines()

with open(f"{DATA_PATH}/{Y_FILE}.txt", "r") as y_text_file:
    y = y_text_file.read().splitlines()

y = np.array(y)
y = y.astype(np.int)
X = np.array(X)

mask = [False for _ in y]
for l in include_labels:
    mask = mask | (y == l)

y = y[mask]
X = X[mask]
labels = np.take(labels, include_labels)


for i, l in enumerate(include_labels):
    y[y == l] = i

include_labels = range(len(include_labels))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE)


def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with", str(np.array(X_train).shape[1]), "features")
    return (X_train, X_test)


def Build_Model_DNN_Text(shape, nClasses, dropout=0.5):
    """
    buildModel_DNN_Tex(shape, nClasses,dropout)
    Build Deep neural networks Model for text classification
    Shape is input feature space
    nClasses is number of classes
    """
    model = Sequential()
    node = 512  # number of nodes
    nLayers = 4  # number of  hidden layer

    model.add(Dense(node, input_dim=shape, activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0, nLayers):
        model.add(Dense(node, input_dim=node, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


X_train, X_test = TFIDF(X_train, X_test)
text_clf = Build_Model_DNN_Text(X_train.shape[1], len(include_labels))
text_clf.summary()
text_clf.fit(X_train, y_train,
             validation_data=(X_test, y_test),
             epochs=10,
             batch_size=128,
             verbose=2)

predicted = text_clf.predict_classes(X_test)

classification_report = metrics.classification_report(y_test, predicted)
confusion_matrix = metrics.confusion_matrix(y_test, predicted)
print(classification_report)
print(confusion_matrix)

disp = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix)
disp.plot()
plt.show()
