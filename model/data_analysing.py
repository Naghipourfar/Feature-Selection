import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

"""
    Created by Mohsen Naghipourfar on 4/25/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""
x_train = pd.read_csv('../Data/fpkm_normalized.csv', header=None)
labels = pd.read_csv('../Data/disease.csv', header=None)
diseases = labels[0].value_counts().index

for i in range(x_train.shape[1]):
    data_to_plot = pd.concat([x_train[i], labels], axis=1)
    data_to_plot.columns = ['feature', 'label']
    f, ax = plt.subplots(figsize=(20, 20))
    plt.xticks(rotation=90)
    sns.boxplot(x='label', y='feature', data=data_to_plot)
    plt.ylabel('feature {0}'.format(i))
    plt.savefig('./Plots/box_{0}'.format(i))
    plt.close()
