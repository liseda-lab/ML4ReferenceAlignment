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

from imblearn.over_sampling import SMOTE
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
        relation = cell.find("{}relation".format(current_namespace)).text

        mappings.append(
            pd.DataFrame(
                data={
                    "entity1": [ent1],
                    "entity2": [ent2],
                    "measure": [measure],
                    "relation": [relation],
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
        on=["entity1", "entity2", "relation"],
        suffixes=(f"_{keys[0]}", f"_{keys[1]}"),
    )

    for tool in keys[2:]:
        merged = merged.merge(
            mappings[tool], how="outer", on=["entity1", "entity2", "relation"],
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


def negative_sampling(measures, df, df_reference):
    """This function produces negative examples given the reference and
    mappings dataset.
    Inputs: df => mappings dataset
            df_reference => reference dataset
    Outputs: negative and positive examples
    """
    print(df.shape)

    ############# TRUE POSITIVES #################
    df_pos = df_reference[["entity1", "entity2", "relation"]].merge(
        df[["entity1", "entity2", "relation"]],
        on=["entity1", "entity2", "relation"],
        how="inner",
    )

    ######### negatives 1+2 (FALSE POSITIVES AND TRUE NEGATIVES TO SOME TOOLS) ############
    df_neg = df_reference[["entity1", "entity2", "relation"]].merge(
        df[["entity1", "entity2", "relation"]],
        on=["entity1", "entity2", "relation"],
        how="outer",
        indicator=True,
    )
    df_neg = df_neg[(df_neg["_merge"] == "right_only") & (df_neg["relation"] != "?")]
    df_neg = df_neg.drop(columns="_merge")

    print("size negatives 1: {}".format(len(df_neg)))

    ####### negatives 2 (TRUE NEGATIVES) ##############
    negs_ids = []
    for measure in measures:
        if measure in df.columns:
            nan_examples = df[df[measure].isna()]
            true_negs = 0

            for i, row in nan_examples.iterrows():
                if df_reference[
                    (df_reference["entity1"] == row["entity1"])
                    & (df_reference["entity2"] == row["entity2"])
                ].empty:
                    negs_ids.append(i)

            # print(f"Minimum for {measure} is {df_an[measure].min()}")
            # print(f"NA: {df_an[measure].isna().sum()}/{len(df)}")
            # print(f"True Negatives: {len(negs_ids)}")

    negs_ids = np.unique(np.array(negs_ids))
    print("size negatives 2: {}".format(len(negs_ids)))

    # select just the negative ids identified above
    df_neg2 = df.iloc[negs_ids][["entity1", "entity2", "relation"]]
    # merge two dataframes of negatives and drop duplicates
    df_neg = pd.concat([df_neg, df_neg2]).drop_duplicates(
        ["entity1", "entity2", "relation"]
    )

    print("size negatives after concatenate both: {}".format(len(df_neg)))

    print("# negative examples: {}".format(len(df_neg)))
    print("# positive examples: {}".format(len(df_pos)))

    return df_neg, df_pos


def feature_bin(x, minimum, maximum):
    if x > 0:
        return maximum
    # elif x == 0: print('?')
    else:
        return minimum


def bin_features(df, minimum, maximum, measures):
    for m in measures:
        df[m] = [feature_bin(x, minimum, maximum) for x in df[m]]
    return df


def negative_sampling_target(measures, df_mappings, df_mappings_ref):
    # compute positives and negatives
    df_neg, df_pos = negative_sampling(measures, df_mappings, df_mappings_ref)
    df_mappings["label"] = labels(df_mappings, df_neg, df_pos)
    return df_mappings


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
            if sampl_stg == "oversample":
                sm = SMOTE(random_state=42, n_jobs=-1)
                X_sampled, y_sampled = sm.fit_resample(X, y)
            else:
                tl = TomekLinks(n_jobs=-1)
                X_sampled, y_sampled = tl.fit_resample(X, y)
            for (classifier, kwargs) in tqdm(
                classifiers, desc="Classifiers", leave=False
            ):
                if "param_grid" in kwargs:
                    try:
                        clf = classifier(random_state=42)
                    except:
                        clf = classifier()
                    grid_search = GridSearchCV(
                        clf,
                        kwargs["param_grid"],
                        n_jobs=-1,
                        cv=10,
                        refit="f1",
                        scoring=["f1", "precision", "recall", "accuracy"],
                        return_train_score=True,
                    )
                    grid_search.fit(X_sampled, y_sampled)
                    clf = grid_search.best_estimator_
                    best_score = grid_search.best_score_
                else:
                    try:
                        clf = classifier(**kwargs, random_state=42)
                    except:
                        clf = classifier(**kwargs)
                    clf.fit(X_sampled, y_sampled)

                for testing_df in testing_dfs:
                    X_test, y_test, _ = numpify_merge_dataframes(
                        [testing_df[measures + ["label"]]], [], "intersection"
                    )
                    y_pred = clf.predict(X_test)
                    f1 = f1_score(y_test, y_pred)
                    acc = accuracy_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)

                    report_data = {
                        "name": [name],
                        "classifier": [clf],
                        "training_df": [training_dfs],
                        "testing_df": [testing_df],
                        "sampling_strategy": [sampl_stg],
                        "f1_train": [best_score],
                        "f1_test": [f1],
                        "acc_test": [acc],
                        "recall_test": [recall],
                        "precision_test": [precision],
                    }

                    if "param_grid" in kwargs:
                        report_data["grid_search"] = [grid_search]

                    reports.append(pd.DataFrame(data=report_data))

                n_models += 1

                if save is not None and n_models % save_rate == 0:
                    df = pd.concat(reports, ignore_index=True)
                    df = df.sort_values('f1_test', ascending = False)
                    with open(save, "wb") as fw:
                        pickle.dump(df, fw, pickle.HIGHEST_PROTOCOL)

    df = pd.concat(reports, ignore_index=True)
    df = df.sort_values('f1_test', ascending = False)
    if save is not None:
        with open(save, "wb") as fw:
            pickle.dump(df, fw, pickle.HIGHEST_PROTOCOL)

    return df

def train_and_eval2(
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
            if sampl_stg == "oversample":
                sm = SMOTE(random_state=42, n_jobs=-1)
                X_sampled, y_sampled = sm.fit_resample(X, y)
            else:
                tl = TomekLinks(n_jobs=-1)
                X_sampled, y_sampled = tl.fit_resample(X, y)
            for (classifier, kwargs) in tqdm(
                classifiers, desc="Classifiers", leave=False
            ):
                if "param_grid" in kwargs:
                    try:
                        clf = classifier(random_state=42)
                    except:
                        clf = classifier()
                    grid_search = GridSearchCV(
                        clf,
                        kwargs["param_grid"],
                        n_jobs=-1,
                        cv=10,
                        refit="f1",
                        scoring=["f1", "precision", "recall", "accuracy"],
                        return_train_score=True,
                    )
                    grid_search.fit(X_sampled, y_sampled)
                    clf = grid_search.best_estimator_
                    best_score = grid_search.best_score_
                else:
                    try:
                        clf = classifier(**kwargs, random_state=42)
                    except:
                        clf = classifier(**kwargs)
                    clf.fit(X_sampled, y_sampled)
                
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

                if "param_grid" in kwargs:
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
    if save is not None:
        with open(save, "wb") as fw:
            pickle.dump(df, fw, pickle.HIGHEST_PROTOCOL)

    return df


def load_rdf(track, res_dir, ref_path, ont1, ont2):
    df_ref = extract_mappings(ref_path, is_ref=True)

    if track == "conference":
        alin = extract_mappings(
            os.path.join(res_dir, "ALIN-{}-{}.rdf".format(ont1, ont2))
        )
        aml = extract_mappings(
            os.path.join(res_dir, "AML-{}-{}.rdf".format(ont1, ont2))
        )
        dome = extract_mappings(
            os.path.join(res_dir, "DOME-{}-{}.rdf".format(ont1, ont2))
        )
        lily = extract_mappings(
            os.path.join(res_dir, "Lily-{}-{}.rdf".format(ont1, ont2))
        )
        logmap = extract_mappings(
            os.path.join(res_dir, "LogMap-{}-{}.rdf".format(ont1, ont2))
        )
        logmaplt = extract_mappings(
            os.path.join(res_dir, "LogMapLt-{}-{}.rdf".format(ont1, ont2))
        )
        ontmat1 = extract_mappings(
            os.path.join(res_dir, "ONTMAT1-{}-{}.rdf".format(ont1, ont2))
        )
        sanom = extract_mappings(
            os.path.join(res_dir, "SANOM-{}-{}.rdf".format(ont1, ont2))
        )
        wiktionary = extract_mappings(
            os.path.join(res_dir, "Wiktionary-{}-{}.rdf".format(ont1, ont2))
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
                res_dir, "AGM-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        aml = extract_mappings(
            os.path.join(
                res_dir, "AML-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        dome = extract_mappings(
            os.path.join(
                res_dir, "DOME-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        fcamap = extract_mappings(
            os.path.join(
                res_dir, "FCAMap-KG-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        logmap = extract_mappings(
            os.path.join(
                res_dir, "LogMap-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        logmapbio = extract_mappings(
            os.path.join(
                res_dir, "LogMapBio-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        logmaplt = extract_mappings(
            os.path.join(
                res_dir, "LogMapLt-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
            )
        )
        pomap = extract_mappings(
            os.path.join(
                res_dir, "POMAP++-largebio-{}_{}_small_2019.rdf".format(ont1, ont2)
            )
        )
        wiktionary = extract_mappings(
            os.path.join(
                res_dir, "Wiktionary-largebio-{}_{}_whole_2019.rdf".format(ont1, ont2)
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

    # merge them all in a dataframe
    df_data = merge_mappings(tool_mappings)
    return df_data, df_ref

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