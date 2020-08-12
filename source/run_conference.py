import os
import random
import itertools
import sys
import pickle

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import pandas as pd
import numpy as np

import utils as u


np.random.seed(42)
random.seed(42)


cf_measures = [
    'measure_alin',
    'measure_aml',
    'measure_dome',
    'measure_lily',
    'measure_logmap',
    'measure_logmaplt',
    'measure_ontmat1',
    'measure_sanom',
    'measure_wiktionary'
]

cf_ontologies = [
    'cmt',
    'conference',
    'confof',
    'edas',
    'ekaw',
    'iasted',
    'sigkdd',
]


conference_data_processed_path = 'data/df_conference.csv'
res_dir = os.path.join('data','conference-data')

if not os.path.isfile(conference_data_processed_path):
    dfs_data, dfs_refs = [],[]
    for ont1, ont2 in itertools.combinations(cf_ontologies,2):
        ref_path = os.path.join(
            "data",
            "conference-ref-data",
            "{}-{}.rdf".format(ont1,ont2),
        )
        df_data, df_ref = u.load_rdf('conference', res_dir,ref_path,ont1,ont2)
        df_data = u.negative_sampling_target(cf_measures, df_data,df_ref)
        df_data["ontologies"] = f"{ont1}-{ont2}"
        dfs_data.append(df_data)
        dfs_refs.append(df_ref)

    df_data = pd.concat(dfs_data, ignore_index = True)
    df_data.to_csv(conference_data_processed_path, index = False)
else:
    df_data = pd.read_csv(conference_data_processed_path)


X, y = df_data[cf_measures], df_data['label']
#fill missing values with 0
X = X.fillna(0)
print(X.shape, y.shape)


df_data2 = X.copy()
df_data2['label'] = y
print(df_data2.corr()['label'])


#for m in cf_measures:
#    u.feature_dist(X, m)


X_bins = u.bin_features(X.copy(), 0,1, cf_measures)
#for m in cf_measures:
#    u.feature_dist(X_bins, m)


df_data_bins = X_bins.copy()
df_data_bins['label'] = y
df_data_bins['ontologies'] = df_data['ontologies']
print(df_data_bins.corr()['label'])


print(df_data_bins.head())


print(df_data_bins.label.value_counts())


def get_conference_data(measures,ont_comb_train,df_data):
    lst_ont_comb = []
    for ont1, ont2 in itertools.combinations(cf_ontologies,2): lst_ont_comb.append(f"{ont1}-{ont2}")
    random.shuffle(lst_ont_comb)

    train_combs = np.array(lst_ont_comb)[:ont_comb_train]
    df_train = df_data[np.isin(df_data['ontologies'].values, train_combs)]
    df_test =  df_data[np.isin(df_data['ontologies'].values, np.array(lst_ont_comb)[-(len(lst_ont_comb)-ont_comb_train):])]

    #get just needed columns: measures and label
    columns_take = list(measures).copy()
    columns_take.append('label')

    df_train = df_train[columns_take]
    df_test = df_test[columns_take]

    return ([df_train], [df_test], train_combs)


ont_comb_train = 18
cross_tuples = []
for _ in range(2):
    cross_tuples.append(get_conference_data(cf_measures, ont_comb_train, df_data_bins))

classifiers = [
    RandomForestClassifier,
    KNeighborsClassifier,
    DecisionTreeClassifier,
    MLPClassifier,
    GaussianNB,
    GradientBoostingClassifier,
    LogisticRegression,
    AdaBoostClassifier
]

classifier_kwargs = [
    {"param_grid": {'n_estimators': list(range(50,250,50)) , 'criterion': ['gini', 'entropy']}},
    {"param_grid": {'n_neighbors': list(range(1,7)), 'p': [1,2]}},
    {"param_grid": {'criterion': ['gini', 'entropy'], 'min_samples_leaf': list(np.arange(0.2,1.2,0.2))}},
    {"param_grid": {'hidden_layer_sizes':[(10,), (40,), (100,), (10, 10), (40, 40), (100, 100)], 'learning_rate_init': [0.01, 0.05, 0.1,]}},
    {"param_grid": {}},
    {"param_grid": {'n_estimators':list(range(50,250,50)),'learning_rate':[0.01, 0.1, 0.2], 'min_samples_leaf': list(np.arange(0.2,1.2,0.2))}},
    {"param_grid": {'C':[0.1,0.5,1,10], 'tol': [1e-2,1e-3,1e-4]}},
    {"param_grid": {'base_estimator': [LogisticRegression()], 'n_estimators': [50,100,150,200]}}
]

df_results = u.train_and_eval(cross_tuples, classifiers, classifier_kwargs, undersample = True, save='data/conference_paper.pkl')


print(df_results.loc[:,df_results.columns!='training_df'])


lb_measures = [
    'measure_agm',
    'measure_aml',
    'measure_dome',
    'measure_fcamap',
    'measure_logmap',
    'measure_logmapbio',
    'measure_logmaplt',
    'measure_pomap++',
    'measure_wiktionary'
]

conf_lb_columns = np.array(lb_measures)[np.isin(np.array(lb_measures), df_data_bins.columns)].tolist()

print(conf_lb_columns)

ont_comb_train = 18
cross_tuples2 = []
for _ in range(2):
    cross_tuples2.append(get_conference_data(conf_lb_columns,ont_comb_train, df_data_bins))

df_results2 = u.train_and_eval(cross_tuples2, classifiers, classifier_kwargs,  undersample = True, save='data/conference_lb_inter_paper.pkl')

print(df_results2.loc[:,df_results2.columns!='training_df'])