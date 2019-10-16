import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    df = pd.read_csv('../dataset/wine.data')
    print(df.info())

    # show_all_histograms(df)
    # show_all_boxplots(df)
