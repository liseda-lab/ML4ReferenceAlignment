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


def read_rdf(ont1, ont2):

    largebio_data_processed_path = 'data/df_largebio_{}_{}.csv'.format(ont1, ont2)
    largebio_ref_processed_path = 'data/df_largebio_{}_{}_ref.csv'.format(ont1, ont2)

    if not os.path.isfile(largebio_data_processed_path):
        # Specify path for the alignments and reference alignments
        res_dir = os.path.join("data","largebio-results-2019")
        ref_path = os.path.join(
                "data",
                "oaei2019_umls_flagged_reference",
                "oaei_{}_{}_mappings_with_flagged_repairs.rdf".format(ont1,ont2)
                )
        # Load rdf data
        df_data, df_ref = u.load_rdf('largebio', res_dir,ref_path,ont1,ont2)

        # Negative sampling
        df_data = u.negative_sampling_target(lb_measures, df_data,df_ref)

        # Save results to csv
        largebio_data_processed_path = 'data/df_largebio_{}_{}.csv'.format(ont1, ont2)
        largebio_ref_processed_path = 'data/df_largebio_{}_{}_ref.csv'.format(ont1, ont2)

        # Save results to csv
        df_data.to_csv(largebio_data_processed_path, index = False)
        df_ref.to_csv(largebio_ref_processed_path, index = False)

    else:
        print('File already exists')
        df_data = pd.read_csv(largebio_data_processed_path)
        df_ref = pd.read_csv(largebio_ref_processed_path)

    return df_data, df_ref


df_lb1, df_lb1_ref = read_rdf('fma', 'nci')
df_lb2, df_lb2_ref = read_rdf('fma', 'snomed')
df_lb3, df_lb3_ref = read_rdf('snomed', 'nci')


print(df_lb1.shape, df_lb2.shape, df_lb3.shape)


df_lb1.fillna(0)
df_lb2.fillna(0)
df_lb3.fillna(0)


X_bins_lb1 = u.bin_features(df_lb1.copy(), 0,1, lb_measures)
X_bins_lb2 = u.bin_features(df_lb2.copy(), 0,1, lb_measures)
X_bins_lb3 = u.bin_features(df_lb3.copy(), 0,1, lb_measures)


Xy_bins_lb1 = X_bins_lb1.copy()
Xy_bins_lb1['label'] = df_lb1['label']
print(Xy_bins_lb1.corr()['label'])


Xy_bins_lb2 = X_bins_lb2.copy()
Xy_bins_lb2['label'] = df_lb2['label']
print(Xy_bins_lb2.corr()['label'])


Xy_bins_lb3 = X_bins_lb3.copy()
Xy_bins_lb3['label'] = df_lb3['label']
print(Xy_bins_lb3.corr()['label'])


df_an_path = os.path.join("data", "df_an.csv")
df_an_ref_path = os.path.join("data", "df_an_ref.csv")

if not os.path.isfile(df_an_path) or not os.path.isfile(df_an_ref_path):

    #load reference
    anatomy_reference_path = os.path.join(
        "data",
        "anatomy-2019-results",
        "reference.rdf"
    )
    df_an_ref = u.extract_mappings(anatomy_reference_path)
    df_an_ref.shape

    #load ontology matching algorithms outputs
    an_res_dir = os.path.join("data", "anatomy-2019")
    an_agm = u.extract_mappings(os.path.join(an_res_dir, "AGM.rdf"))
    an_aml = u.extract_mappings(os.path.join(an_res_dir, "AML.rdf"))
    an_dome = u.extract_mappings(os.path.join(an_res_dir, "DOME.rdf"))
    an_fcamap = u.extract_mappings(os.path.join(an_res_dir, "FCAMap-KG.rdf"))
    an_logmap = u.extract_mappings(os.path.join(an_res_dir, "LogMap.rdf"))
    an_logmapbio = u.extract_mappings(os.path.join(an_res_dir, "LogMapBio.rdf"))
    an_logmaplt = u.extract_mappings(os.path.join(an_res_dir, "LogMapLt.rdf"))
    an_pomappp = u.extract_mappings(os.path.join(an_res_dir, "POMAP++.rdf"))
    an_wiktionary = u.extract_mappings(os.path.join(an_res_dir, "Wiktionary.rdf"))

    an_tool_mappings = {
        "agm": an_aml,
        "aml": an_aml,
        "dome": an_dome,
        "fcamap": an_fcamap,
        "logmap": an_logmap,
        "logmapbio": an_logmapbio,
        "logmaplt": an_logmaplt,
        "pomap++": an_pomappp,
        "wiktionary": an_wiktionary,
    }

    #merge them all in a dataframe
    df_an = u.merge_mappings(an_tool_mappings)
    df_an.shape

    #export data
    df_an_ref.to_csv(df_an_ref_path, index=False)

#read files
else:
    print('read preprocess files')
    df_an = pd.read_csv(df_an_path)
    df_an_ref = pd.read_csv(df_an_ref_path)


# Missing values
df_an.fillna(0)

#Make data binary
X_bins_an = u.bin_features(df_an.copy(), 0,1, lb_measures)
Xy_bins_an = X_bins_an.copy()
Xy_bins_an['label'] = df_an['label']
print(Xy_bins_an.corr()['label'])


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

cross_tuples = [
    ([Xy_bins_an, Xy_bins_lb1, Xy_bins_lb2, Xy_bins_lb3], [Xy_bins_an], "i1"),
    ([Xy_bins_an, Xy_bins_lb1, Xy_bins_lb2, Xy_bins_lb3], [Xy_bins_lb1, Xy_bins_lb2, Xy_bins_lb3], "i2"),
    ([Xy_bins_an], [Xy_bins_lb1, Xy_bins_lb2, Xy_bins_lb3], "ii"),
    ([Xy_bins_lb1], [Xy_bins_lb2, Xy_bins_lb3], "iii1"),
    ([Xy_bins_lb2], [Xy_bins_lb1, Xy_bins_lb3], "iii2"),
    ([Xy_bins_lb3], [Xy_bins_lb1, Xy_bins_lb2], "iii3"),
    ([Xy_bins_lb2, Xy_bins_lb3], [Xy_bins_lb1], "iv1"),
    ([Xy_bins_lb1, Xy_bins_lb3], [Xy_bins_lb2], "iv2"),
    ([Xy_bins_lb1, Xy_bins_lb2], [Xy_bins_lb3], "iv1"),
    ]

df_results = u.train_and_eval(cross_tuples, classifiers, classifier_kwargs,undersample=True, save='data/largebio_paper.pkl')


print(df_results.loc[:,df_results.columns!='training_df'])