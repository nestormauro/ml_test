import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
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
features_col_names = col_names
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
    c = 4
    svmc = SVC(C=c, kernel='rbf', degree=3, gamma='auto')
    svmc.fit(X_train,y_train)
    y_pred=svmc.predict(X_test)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy_score}")
    mlflow.sklearn.log_model(svmc, "svm-classifier")
    mlflow.log_metric("accuracy", accuracy_score)
    mlflow.log_param("C", c)
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    sn.set(font_scale=1.4)
    sn.heatmap(conf_matrix, cmap="Blues", annot=True,annot_kws={"size": 16})
    mlflow.log_figure(fig, 'conf_matrix.png')