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
from sklearn.svm import SVC
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


class_name = 'Cultivars'

# how to build heatmap for my dataset?
# what is expectation by other plots?
# SVC is it correct?
# Data scaling what method to use?

def show_all_histograms(data):
    for column in data.columns:
        sns.distplot(data[column], kde=False)
        plt.show()


def show_all_boxplots(data: pd.DataFrame):
    for column in data.columns:
        if column == class_name: continue
        sns.boxplot(x=column, y=class_name, data=df, orient='h')
        plt.show()


#n=6
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


def svc_fit(data_, scaler_):
    data_ = scaler_.fit_transform(np.asarray(data_))
    svc = SVC()
    x_train, x_test, y_train, y_test = train_test_split(data_, target, test_size=0.3)
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: {0}'.format(score))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def find_the_most_important_features(rfc):
    feature_imp = pd.Series(rfc.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(feature_imp)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()


def rfc_fit(data_):
    rfc = RandomForestClassifier(n_estimators=100)
    x_train, x_test, y_train, y_test = train_test_split(data_, target, test_size=0.3)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def adaboost_fit(data_):
    #svc = SVC(probability=True, kernel='linear')
    ab = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    x_train, x_test, y_train, y_test = train_test_split(data_, target, test_size=0.3, random_state=42)
    ab.fit(x_train, y_train)
    y_pred = ab.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


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

    adaboost_fit(data)



