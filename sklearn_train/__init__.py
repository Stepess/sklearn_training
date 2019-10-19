import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing as prep
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

class_name = 'Cultivars'


def show_all_histograms(data):
    for column in data.columns:
        sns.distplot(data[column], kde=False)
        plt.show()


def show_all_boxplots(data: pd.DataFrame):
    for column in data.columns:
        if column == class_name: continue
        sns.boxplot(x=column, y=class_name, data=df, orient='h')
        plt.show()


def knn_fit(data_, n, scaler_):
    data_ = scaler_.fit_transform(np.asarray(data_))
    x_train, x_test, y_train, y_test = train_test_split(data_, target, test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: {0}'.format(score))
    return knn


def dtc_fit(data_):
    x_train, x_test, y_train, y_test = train_test_split(data_, target, test_size=0.3)
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    y_pred = dtc.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: {0}'.format(score))

    dot_data = StringIO()
    export_graphviz(dtc, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['0', '1', '2'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('wines.png')
    Image(graph.create_png())


if __name__ == '__main__':
    feature_cols = [
        'Alcohol',
        'Malic_acid',
        'Ash',
        'Alcalinity_of_ash',
        'Magnesium',
        'Total_phenols',
        'Flavanoids',
        'Nonflavanoid_phenols',
        'Proanthocyanins',
        'Color_intensity',
        'Hue',
        'OD280/OD315_of_diluted_wines',
        'Proline'
    ]

    cols = feature_cols.copy()
    cols.insert(0, class_name)

    df = pd.read_csv('../dataset/wine.data', header=None, names=cols)
    print(df.info())

    # show_all_histograms(df)
    # show_all_boxplots(df)
    scaler = prep.MinMaxScaler()
    data = df[[x for x in df.columns if x != class_name]]
    target = df[class_name]

    dtc_fit(data)
