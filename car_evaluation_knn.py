import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import mlflow

data = 'data/car_evaluation.csv'
df = pd.read_csv(data, index_col=False)

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names
#col_names = ['buying', 'maint', 'safety', 'class']
features_col_names = col_names
#df = df[col_names]
features_col_names.remove('class')

for col in features_col_names:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

    
#encoder = LabelEncoder()
#df['class'] = encoder.fit_transform(df['class'])
#df['class'].value_counts())
#display(df.head(5))


X_train, X_test, y_train, y_test = train_test_split(df[features_col_names], df['class'], test_size=0.2, random_state=0)
mlflow.set_experiment('car-evaluation')

with mlflow.start_run():
    n_neighbors = 10
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy_score}")
    mlflow.sklearn.log_model(knn, "knn-classifier")
    mlflow.log_metric("accuracy", accuracy_score)
    mlflow.log_param("n_neighbors", n_neighbors)
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    sn.set(font_scale=1.4)
    sn.heatmap(conf_matrix, cmap="Blues", annot=True,annot_kws={"size": 16})
    mlflow.log_figure(fig, 'conf_matrix.png')