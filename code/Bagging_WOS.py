from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', BaggingClassifier(
                         KNeighborsClassifier(), random_state=RANDOM_STATE)),
                     ])


text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)

classification_report = metrics.classification_report(y_test, predicted)
confusion_matrix = metrics.confusion_matrix(y_test, predicted)
print(classification_report)
print(confusion_matrix)

disp = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix)
disp.plot()
plt.show()


# psychology test article
test_article = "Whether world war or a global pandemic, these periods of history can feel nightmarish, especially given the bedrocks of American belief. Much of our national identity stems from our belief in meritocracy: that if we work hard, we will be rewarded. In a nation that values optimism, practicality, and planning, it’s difficult to cope with the suddenness of trauma: the fact that some questions cannot be fully answered. It’s why we struggle to mourn, to sit with our suffering and to see it as real, to be awake with pain and not to understand it as “the irrational reality of a dream."

test_art_pred = text_clf.predict([test_article])

print(test_art_pred)
