import xml.etree.ElementTree as ET
from functools import reduce
import pandas as pd
import re
import numpy as np
import pickle
from scipy import stats
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    ParameterGrid,
)
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import os

# This function helps to read the rdf ontology alignment files for each tool and also the reference
# @param ont1, ont2: ontologies that participate in the alignment
# @param measures: list of tools available in the track
# @param track: the OAEI track in use. Current 
# @param data_path: path to the ontology alignment files
# @param ref_path: path to the reference alignment
# @param data_processed_path: path to which save the processed data files
# @param data_processed_path: path to which save the processed reference file
def read_rdf(ont1, ont2, measures, track, data_path, ref_path, data_processed_path, ref_processed_path):

    if not os.path.isfile(data_processed_path):
        # Load rdf data
        df_data, df_ref = load_rdf(track, data_path, ref_path, ont1, ont2)

        # Save results to csv
        df_data.to_csv(data_processed_path, index = False)
        df_ref.to_csv(ref_processed_path, index = False)

    else:
        print('File already exists')
        df_data = pd.read_csv(data_processed_path)
        df_ref = pd.read_csv(ref_processed_path)

    return df_data, df_ref


def load_rdf(track, data_path, ref_path, ont1, ont2):
    df_ref = extract_mappings(ref_path, is_ref=True)

    if track == "conference":
        alin = extract_mappings(
            os.path.join(data_path, "ALIN-{}-{}.rdf".format(ont1, ont2))
        )
        aml = extract_mappings(
            os.path.join(data_path, "AML-{}-{}.rdf".format(ont1, ont2))
        )
        dome = extract_mappings(
            os.path.join(data_path, "DOME-{}-{}.rdf".format(ont1, ont2))
        )
        lily = extract_mappings(
            os.path.join(data_path, "Lily-{}-{}.rdf".format(ont1, ont2))
        )
        logmap = extract_mappings(
            os.path.join(data_path, "LogMap-{}-{}.rdf".format(ont1, ont2))
        )
        logmaplt = extract_mappings(
            os.path.join(data_path, "LogMapLt-{}-{}.rdf".format(ont1, ont2))
        )
        ontmat1 = extract_mappings(
            os.path.join(data_path, "ONTMAT1-{}-{}.rdf".format(ont1, ont2))
        )
        sanom = extract_mappings(
            os.path.join(data_path, "SANOM-{}-{}.rdf".format(ont1, ont2))
        )
        wiktionary = extract_mappings(
            os.path.join(data_path, "Wiktionary-{}-{}.rdf".format(ont1, ont2))
        )
        tool_mappings = {
            "alin": alin,
            "aml": aml,
            "dome": dome,
            "lily": lily,
            "logmap": logmap,
            "logmaplt": logmaplt,
            "ontmat1": ontmat1,
            "sanom": sanom,
            "wiktionary": wiktionary,
        }

    elif track == "largebio":
        agm = extract_mappings(
            os.path.join(
                data_path, "AGM-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        aml = extract_mappings(
            os.path.join(
                data_path, "AML-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        dome = extract_mappings(
            os.path.join(
                data_path, "DOME-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        fcamap = extract_mappings(
            os.path.join(
                data_path, "FCAMap-KG-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        logmap = extract_mappings(
            os.path.join(
                data_path, "LogMap-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        logmapbio = extract_mappings(
            os.path.join(
                data_path, "LogMapBio-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        logmaplt = extract_mappings(
            os.path.join(
                data_path, "LogMapLt-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        pomap = extract_mappings(
            os.path.join(
                data_path, "POMAP++-largebio-{}_{}_small_2019.rdf".format(ont1, ont2)
            )
        )
        wiktionary = extract_mappings(
            os.path.join(
                data_path, "Wiktionary-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        tool_mappings = {
            "agm": agm,
            "aml": aml,
            "dome": dome,
            "fcamap": fcamap,
            "logmap": logmap,
            "logmapbio": logmapbio,
            "logmaplt": logmaplt,
            "pomap++": pomap,
            "wiktionary": wiktionary,
        }
   
    else: 
        print("That track is not specified. Extend the load_rdf() function in utils.py with the paths to the files of the track you are working with.")
        return

    # merge everything in one dataframe
    df_data = merge_mappings(tool_mappings)
    return df_data, df_ref

def extract_mappings(rdf_path, is_ref=False):
    tree = ET.parse(rdf_path)
    root = tree.getroot()
    ns = re.match(r"\{.*\}", root.tag).group(0)

    ont_web_format = (
        "{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}"
        if is_ref
        else "{http://knowledgeweb.semanticweb.org/heterogeneity/alignment#}"
    )
    namespaces = {"rdf": "", "ont": ont_web_format}
    current_namespace = (
        namespaces["ont"]
        if "knowledgeweb.semanticweb.org" in root[0].tag
        else namespaces["rdf"]
    )
    mappings = []

    for _map in root[0].findall("{}map".format(current_namespace)):
        cell = _map.find("{}Cell".format(current_namespace))
        ent1 = cell.find("{}entity1".format(current_namespace)).get(f"{ns}resource")
        ent2 = cell.find("{}entity2".format(current_namespace)).get(f"{ns}resource")
        measure = float(cell.find("{}measure".format(current_namespace)).text)

        mappings.append(
            pd.DataFrame(
                data={
                    "entity1": [ent1],
                    "entity2": [ent2],
                    "measure": [measure]
                }
            )
        )
    return pd.concat(mappings, ignore_index=True)


def merge_mappings(mappings):
    keys = list(mappings.keys())
    assert len(keys) >= 2

    merged = mappings[keys[0]].merge(
        mappings[keys[1]],
        how="outer",
        on=["entity1", "entity2"],
        suffixes=(f"_{keys[0]}", f"_{keys[1]}"),
    )

    for tool in keys[2:]:
        merged = merged.merge(
            mappings[tool], how="outer", on=["entity1", "entity2"],
        )
        merged[f"measure_{tool}"] = merged["measure"]
        merged = merged.drop(columns="measure")

    return merged


def labels(df, df_neg, df_pos):
    return df.apply(
        lambda x: 1
        if df_neg[
            (df_neg["entity1"] == x.entity1) & (df_neg["entity2"] == x.entity2)
        ].empty
        and not df_pos[
            (df_pos["entity1"] == x.entity1) & (df_pos["entity2"] == x.entity2)
        ].empty
        else 0,
        axis=1,
    )


def numpify_merge_dataframes(training_dfs, testing_dfs, strategy):
    if strategy == "intersection":
        measures = [
            list(filter(lambda col: "measure_" in col, df.columns))
            for df in training_dfs
        ]
        measures.extend(
            [
                list(filter(lambda col: "measure_" in col, df.columns))
                for df in testing_dfs
            ]
        )
        measures = list(
            reduce(
                lambda acc, x: list(set(acc).intersection(set(x))),
                measures[1:],
                measures[0],
            )
        )
        merged_df = pd.concat(
            [df[measures + ["label"]] for df in training_dfs], ignore_index=True
        )
        X = merged_df[measures].to_numpy()
        y = merged_df["label"].to_numpy()
        return X, y, measures


def feature_bin(x, minimum, maximum):
    if x > 0:
        return maximum
    # elif x == 0: print('?')
    else:
        return minimum


def bin_features(df, minimum, maximum):
    for m in df.columns[2:]:
        df[m] = [feature_bin(x, minimum, maximum) for x in df[m]]
    return df


def feature_dist(df, column_name):
    plt.figure(figsize=(15, 4))
    sns.distplot(df[column_name], fit=norm)

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df[column_name])
    print("\n mu = {:.2f} and sigma = {:.2f}\n".format(mu, sigma))
    plt.legend(
        ["Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )".format(mu, sigma)],
        loc="best",
    )
    plt.ylabel("Frequency")
    plt.title("{} distribution".format(column_name))
    plt.show()

def train_and_eval(
    cross_tuples,
    classifiers,
    classifier_kwargs,
    missing_feature_strategy="intersection",
    undersample=False,
    save=None,
    save_rate=20
):
    """
    cross_tuples: A list of tuples with the following shape:
        (*[Training DataFrames], *[Testing DataFrames], name : string)
    classifiers: list of classifier classes
    classifier_kwargs: list of dictionaries that will be used as keyword arguments for the classifier.
                    If the kwargs includes a key 'param_grid' with a dictionary of value ranges,
                    the optimum hyperparameters will be searched for using a GridSearch.
    missing_feature_strategy: Either intersection or substitution. Intersection will remove
                    features not in common. Substitution will substitute the prediction
                    of the missing tool with a 0.
    undersample: Boolean. Indicates whether to try undersampling.
    save: String. Path to which to save the pickled dataframe.
        This function may be useful as the dataframe includes the objects of the classifiers, which may
        become useful to store to analyze later (beta coefficients, weights, etc.)
    save_rate: The rate of save, in number of models trained. Every N models, the results are saved.
    """

    reports = []

    classifiers = list(zip(classifiers, classifier_kwargs))
    sampling_strategies = ["oversample"]

    n_models = 0

    if undersample:
        sampling_strategies.extend(["undersample"])
    for (training_dfs, testing_dfs, name) in tqdm(cross_tuples, desc="Cross Tuples"):
        X, y, measures = numpify_merge_dataframes(
            training_dfs, testing_dfs, missing_feature_strategy
        )
        for sampl_stg in tqdm(
            sampling_strategies, desc="Sampling Strategy", leave=False
        ):
         
            for (classifier, kwargs) in tqdm(
                classifiers, desc="Classifiers", leave=False
            ):
                if sampl_stg == "oversample":
                    pipe = Pipeline([
                        ('sampling', SMOTETomek(random_state= 42, n_jobs= -1)),
                        ('classifier', classifier())
                    ]) 
                else:
                    pipe = Pipeline([
                        ('sampling', TomekLinks(n_jobs= -1)),
                        ('classifier', classifier())
                    ]) 
                    
                new_params = {'classifier__' + key: kwargs["param_grid"][key] for key in kwargs["param_grid"]}
            
                grid_search = GridSearchCV(
                    pipe,
                    new_params,
                    n_jobs=-1,
                    cv=10,
                    refit="f1",
                    scoring=["f1", "precision", "recall", "accuracy"],
                    return_train_score=True,
                )
                grid_search.fit(X, y)
                clf = grid_search.best_estimator_
                best_score = grid_search.best_score_
            
                df_test = pd.concat(testing_dfs)
                
                X_test, y_test, _ = numpify_merge_dataframes(
                            [df_test[measures + ["label"]]], [], "intersection"
                        )
                
                y_pred = clf.predict(X_test)
                f1 = f1_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)

                report_data = {
                    "name": [name],
                    'measures': [measures],
                    "classifier": [clf],
                    "training_df": [training_dfs],
                    "testing_df": [df_test],
                    "sampling_strategy": [sampl_stg],
                    "f1_train": [best_score],
                    "f1_test": [f1],
                    "acc_test": [acc],
                    "recall_test": [recall],
                    "precision_test": [precision],
                }

                report_data["grid_search"] = [grid_search]

                reports.append(pd.DataFrame(data=report_data))

                n_models += 1

                if save is not None and n_models % save_rate == 0:
                    df = pd.concat(reports, ignore_index=True)
                    df = df.sort_values('f1_test', ascending = False)
                    with open(save, "wb") as fw:
                        pickle.dump(df, fw, pickle.HIGHEST_PROTOCOL)

    df = pd.concat(reports, ignore_index=True)
    df = df.sort_values('f1_train', ascending = False)
    
    #remove traing_df column that is not readable
    #sort values by test accuracy
    readable_results = df.loc[:,df.columns!='training_df'].sort_values(by=['f1_test'],ascending=False)
    readable_results['classifier_name'] = [c[0:c.index('(')] for c in readable_results.classifier.astype(str)]
    
    if save is not None:
        with open(save, "wb") as fw:
            pickle.dump(readable_results, fw, pickle.HIGHEST_PROTOCOL)

    return readable_results

def plot_top_results(df_data,title, sort_by = ['f1_train'], top= 5, logs = True):
    df_data = df_data.sort_values(by=sort_by,ascending=False).iloc[:top]
    bar_width = 0.15

    # Set position of bar on X axis
    r1 = np.arange(top)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3] 
    r5 = [x + bar_width for x in r4]

    plt.figure(figsize=(15,5))
    # Make the plot
    plt.bar(r1, df_data.f1_train.values, width=bar_width, edgecolor='white', label='F1 Train')
    plt.bar(r2, df_data.f1_test.values, width=bar_width, edgecolor='white', label='F1 Test')
    plt.bar(r3, df_data.acc_test.values, width=bar_width, edgecolor='white', label='Accuracy Test')
    plt.bar(r4, df_data.precision_test.values, width=bar_width, edgecolor='white', label='Precision Test')
    plt.bar(r5, df_data.recall_test.values, width=bar_width, edgecolor='white', label='Recall Test')

    plt.xlabel('Classifier', fontweight='bold')
    plt.xticks([r + bar_width for r in range(top)], df_data.classifier_name.values, rotation=45)
    plt.yticks(np.arange(0,1.1,0.1))
    plt.ylim(0,1.1)
    plt.grid(axis='y')
    plt.legend(loc='upper center', ncol=5)
    plt.title('{} top {} sorted by {}'.format(title, top, str(sort_by)))
    
    if logs:
        for i in range(top):
            r = df_data.iloc[i]
            print('------\nTop:{}\nSampling strategy: {}\nF1 Train: {:.3f}\nF1 Test: {:.3f}\nAccuracy Test: {:.3f} \
            \nPrecision Test: {:.3f}\nRecall Test: {:.3f}'.format(i+1,r.sampling_strategy, r.f1_train, r.f1_test, \
                                                                  r.acc_test, r.precision_test, r.recall_test))