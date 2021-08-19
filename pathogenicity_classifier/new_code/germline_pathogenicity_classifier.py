import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, StratifiedShuffleSplit
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score 
import argparse
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import shap
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
from lime import submodular_pick
import matplotlib.pyplot as plt
import xgboost as xgb
from collections import Counter
import matplotlib.style as style
import matplotlib
# from extract_feature_effect_per_prediction import create_prediction_factors_df
from tcxp import rf_explain

def remove_muts(df, types):
    
    if 'Missense' in types:
        df = df[~df.Variant_Classification.str.startswith('Missense_Mutation')]
        df.drop('Consequence_missense_variant', axis = 1, inplace = True, errors='ignore') 
    if 'Splice' in types:
        df = df[~df.Variant_Classification.str.startswith('Splice')]
        df.drop('Consequence_splice_acceptor_variant', axis = 1, inplace = True, errors='ignore')   
        df.drop('Consequence_splice_donor_variant', axis = 1, inplace = True, errors='ignore') 
        df.drop('Consequence_splice_region_variant', axis = 1, inplace = True, errors='ignore') 
    if 'Truncating' in types:
        df = df[~df.Variant_Classification.str.startswith('Frame')]
        df.drop('Consequence_frameshift_variant', axis = 1, inplace = True, errors='ignore') 
        df = df[~df.Variant_Classification.str.startswith('Non')] # Non stop and Nonsense
        df = df[~df.Variant_Classification.str.startswith('Translation_Start_Site')]
    if 'Misc' in types:
        df = df[~df.Variant_Classification.str.startswith('In_Frame')]
        df = df[~df.Variant_Classification.str.startswith('Silent')]
        df = df[~df.Variant_Classification.str.contains("UTR")]
        df = df[~df.Variant_Classification.str.startswith('Intron')]
        df = df[~df.Variant_Classification.str.startswith('IGR')]
        df = df[~df.Variant_Classification.str.startswith('RNA')]
        df.drop('Consequence_stop_gained', axis = 1, inplace = True, errors='ignore') 
    if 'In_Frame' in types:
        df = df[~df.Variant_Classification.str.startswith('In_Frame')]
    if 'Silent' in types:
        df = df[~df.Variant_Classification.str.startswith('Silent')]
    if 'UTR' in types:
        df = df[~df.Variant_Classification.str.contains('UTR')]
    if 'Intron' in types:
        df = df[~df.Variant_Classification.str.startswith('Intron')]
    if 'IGR' in types:
        df = df[~df.Variant_Classification.str.startswith('IGR')]
    if 'RNA' in types:
        df = df[~df.Variant_Classification.str.startswith('RNA')]
        
    return df

# Helper function that does cross validation
def error(clf, X, y, ntrials=10, test_size=0.2, train_size = None) : 

    train = [] 
    test = [] 
    f1 = [] 
    prec = [] 
    rec = []
  
    sss = StratifiedShuffleSplit(n_splits=ntrials, test_size = test_size, train_size= train_size, random_state=42) 

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred2 = clf.predict(X_train)

        # measures of model performance
        test.append(1 - accuracy_score(y_test, y_pred, normalize=True)) 
        train.append(1 - accuracy_score(y_train, y_pred2, normalize=True)) 
        f1.append(f1_score(y_test, y_pred))
        prec.append(precision_score(y_test, y_pred, zero_division= 0))
        rec.append(recall_score(y_test, y_pred))
    

    return np.mean(train), np.std(train), np.mean(test), np.std(test), np.mean(prec), np.std(prec), np.mean(rec), np.std(rec), np.mean(f1), np.std(f1)

def get_df_prob(predictions):
    df_predictions = pd.DataFrame(data=predictions, columns=['Prob_0', 'Prob_1'])
    df_predictions['Predicted_label'] = np.where(df_predictions['Prob_1']> df_predictions['Prob_0'], 1, 0)
    df_predictions['Predicted_label_soft'] = np.where(df_predictions['Prob_1']> 0.45, 1, 0)
    df_predictions['prediction_prob'] = df_predictions['Prob_1']
    return(df_predictions)

def get_gene_level_annotations():
    #get annotations for germline genes as oncogenes / tumor suppressors or genes whose function is via gain vs. loss
    gene_level_annotation = pd.read_csv(gene_level_input_file, sep = '\t', low_memory = False)
    gene_annotation_columns = gene_level_annotation.columns.tolist()
    gene_annotation_columns = gene_annotation_columns.remove('Hugo_Symbol')

    oncogenes = gene_level_annotation[gene_level_annotation['OncoKB Oncogene']==1]['Hugo_Symbol'].unique().tolist()
    oncogenes.append('TP53')
    #POLE and POLD1 are added as oncogenes because we know that only missense mutations in POLE lead to signature 10. 
    oncogenes.append('POLE')
    oncogenes.append('POLD1')
    non_cancergenes = gene_level_annotation[gene_level_annotation['OMIM']==0]['Hugo_Symbol'].unique().tolist()


    tumor_suppressors =  gene_level_annotation[gene_level_annotation['OncoKB TSG']==1]['Hugo_Symbol'].unique().tolist()
    tumor_suppressors.remove('POLE')
    tumor_suppressors.remove('POLD1')
    tumor_suppressors.append('EPCAM')

    #this file contains some additional gene annotations from CB.
    function_map_other_genes = pd.read_csv(gene_function_map_input, sep='\t', low_memory = False)
    other_gof_genes = function_map_other_genes[function_map_other_genes['oncogenic mechanism']=='gain-of-function']['Hugo_Symbol'].tolist()
    other_lof_genes = function_map_other_genes[function_map_other_genes['oncogenic mechanism']=='loss-of-function']['Hugo_Symbol'].tolist()
    #print gene_level_annotation.head()

    tumor_suppressors =list(set(tumor_suppressors + other_lof_genes))
    oncogenes =list(set(oncogenes + other_gof_genes+['VTCN1', 'YES1', 'XPO1', 'TRAF7']))
    print(sorted(oncogenes[:5]))
    print(sorted(tumor_suppressors[:5]))
    return oncogenes, tumor_suppressors, gene_level_annotation


def main():

    type_list = ['Missense', 'Splice', 'Truncating', 'Misc','In_Frame', 'Silent',  "UTR", "Intron", "IGR", "RNA"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier_input', type=str, default='data/test_annotated.maf', help='This should be an annotated .maf file that serves as the test data for the classifier.')
    parser.add_argument('--classifier_output', type=str, default='data/test.classifier_output.maf', help='This file will store the output of the classifier after running it on the test data.')
    parser.add_argument('--training_data', type=str, default=None, help='This should be a .maf or .maf.gz file that serves as the training data for the classifier.')
    parser.add_argument('--type', type=str, nargs='+',default=[], help='This defines what types of mutations should be removed. The possible arguments are: %s -- or any combination of the options (separated by space).' % ', '.join(map(str, type_list)))
    parser.add_argument('--features', type=str, default='features_to_keep.txt', help='This should be a .txt file containg the features to retain.')
    parser.add_argument('--scripts_dir', type=str, default=os.path.join(os.getcwd()), help='This should be the path to where the annotation file directory resides.')

    args = parser.parse_args()
    classifier_input = args.classifier_input
    classifier_output = args.classifier_output
    training_data_file = args.training_data
    features_file = args.features
    scripts_dir = args.scripts_dir
    type_vals = args.type

    global gene_level_input_file, gene_function_map_input

    #this is the path where all the scripts and files exist. 
    annotation_path = os.path.join(scripts_dir, "annotation_files")

    #paths to annotation sources
    gene_level_input_file = os.path.join(annotation_path, "gene_level_annotation.txt")
    gene_function_map_input = os.path.join(annotation_path, "gene_function_other_genes.txt")
    
    if training_data_file == None:
        training_data_file = os.path.join(scripts_dir, "input_files/classifier_training_data_V2.maf.gz") 

    # read in the training data, print its shape, and get the training labels
    X = pd.read_csv(training_data_file, sep="\t", compression = 'infer', low_memory = False)
    print("The shape of the training data (%s) on initial load is %s\n"%(training_data_file, str(X.shape)))

    # remove all mutations specified by type for the training data
    X = remove_muts(X, type_vals)
    print("The shape of the training data (%s) after removing specified mutations is %s\n"%(training_data_file, str(X.shape)))

    # deal with differing names for training data labels
    if 'pathogenic' in X.columns.tolist():
        y_train = X['pathogenic']
    elif 'signed_out' in X.columns.tolist():
        y_train = X['signed_out']

    # read in the test data and print its shape
    X_test = pd.read_csv(classifier_input, sep = '\t', low_memory = False)
    
    # rename columns to adhere to old naming convention - Might want to change later
    if 'dbnsfp.fathmm.mkl.coding_rankscore' in X:
        X_test = X_test.rename(columns={'dbnsfp.fathmm-mkl.coding_rankscore': 'dbnsfp.fathmm.mkl.coding_rankscore'})
    if 'dbnsfp.eigen.pc.raw' in X:
        X_test = X_test.rename(columns={'dbnsfp.eigen-pc.raw_coding': 'dbnsfp.eigen.pc.raw'})

    print("The shape of the testing data (%s) on initial load is %s\n"%(classifier_input, str(X_test.shape)))

    # remove all mutations not specified by type for the test data
    X_test = remove_muts(X_test, type_vals)

    print("The shape of the testing data (%s) after removing specified mutations is %s\n"%(classifier_input, str(X_test.shape)))

    # ######################################################
    # keep_columns = X_test.columns.tolist()
    # keep_columns[:] = [feature for feature in keep_columns if feature in X]
    # print(str(len(keep_columns)) + "\n")
    # #######################################################

    # get list of features to retain from annotated file
    with open(features_file) as features:
        keep_columns = features.read().splitlines()

    # if a feature is not in our testing data, we don't want to retain it (the : modifies list in place)
    keep_columns[:] = [feature for feature in keep_columns if feature in X_test]
    keep_columns[:] = [feature for feature in keep_columns if feature in X]

    # keep only the subset of features deemed useful
    X_train = X[keep_columns]

    # X_train = X_train.loc[:, X_train.isnull().sum() < 0.10*X_train.shape[0]]

    # convert categorical variables into binary features (dummy encoding)
    # cats = X_train.select_dtypes(exclude=np.number).columns.tolist()
    # X_train = X_train.drop(columns = X_train[cats].loc[:,X_train[cats].apply(pd.Series.nunique) >= 3].columns.tolist())
    # final_columns = X_train.columns.tolist()
    X_train = pd.get_dummies(X_train)
    
    # change NaN values to 0s
    X_train = X_train.fillna(value=0)

    # get columns that we want for our test data
    final_columns = X_train.columns.tolist()
    print("The shape of the training data after removing features, NaNs, and dummy encoding is %s\n"%(str(X_train.shape)))
    
    ##############################################################
    # # use GridSearchCV to find optimal hyperparameters
    # estimators = np.arange(100, 1000, 50)
    # random_grid = {'n_estimators':estimators}

    # # Use the random grid to search for best hyperparameters
    # # First create the base model to tune
    # clf = RandomForestClassifier(criterion = 'gini', min_samples_leaf= 1,
    #                                 min_samples_split= 2, max_features=12, random_state=42)
    # # Random search of parameters, using 3 fold cross validation, 
    # # search across 100 different combinations, and use all available cores
    # clf_random = GridSearchCV(estimator = clf, param_grid = random_grid, cv = 3, verbose=2, n_jobs = -1)
    # # Fit the random search model
    # clf_random.fit(X_train, y_train)

    # print(clf_random.best_params_)
    # exit()
    ##############################################################
    
    # define and train the model
    clf = RandomForestClassifier(n_estimators=300, criterion= 'gini', min_samples_leaf= 1,
                                min_samples_split= 2, max_features=12, random_state=42, 
                                class_weight={0: 1, 1: 4.5})

    # counter = Counter(y_train)
    # est = counter[0]/counter[1]
    # print(est)
    # clf = xgb.XGBClassifier(scale_pose_weight = est, verbosity = 0,  use_label_encoder = False)
    
    # clf = MLPClassifier(hidden_layer_sizes = (100, 10),activation='logistic', solver = 'adam', learning_rate='adaptive')
    clf.fit(X_train, y_train)
    model_name = type(clf).__name__

    # cols = X_train.columns
    # nums = X_train._get_numeric_data().columns
    # cats = list(set(cols) - set(nums))

    # nums_index = X_train.columns.get_indexer(nums)
    # cats_index = X_train.columns.get_indexer(cats)

    # explainer = lime_tabular.LimeTabularExplainer(X_train.values, mode='classification',training_labels=y_train,feature_names=X_train.columns.tolist(), discretize_continuous=False)
    # l=[]
    # for n in range(0,1000):
    #     exp = explainer.explain_instance(X_train.values[n], clf.predict_proba, num_features=32)
    #     a=dict(exp.as_list(exp.available_labels()[0]))
    #     a['prediction'] = y_train.values[n]
    #     l.append(a)
    # W_pick = pd.DataFrame(l).fillna(0)

    
    # explainer = LimeTabularExplainer(X_train.values,class_names = [1,0], feature_names = X_train.columns, mode = 'classification', discretize_continuous=False)
    # exp=explainer.explain_instance(X_train.to_numpy()[0],clf.predict_proba,top_labels=32)
    # print(exp.available_labels())
    # sp_obj = submodular_pick.SubmodularPick(data=X_train.to_numpy(),explainer=explainer,num_exps_desired=X_train.shape[0], sample_size=X_train.shape[0], predict_fn=clf.predict_proba, num_features=32)
    # W_pick=pd.DataFrame([dict(this.as_list(this.available_labels()[0])) for this in sp_obj.sp_explanations]).fillna(0)
    # W_pick['prediction'] = [this.available_labels()[0] for this in sp_obj.sp_explanations]
    # W_pick.to_csv('/home/dalalt1/PathoClass/somatic-germline/pathogenicity_classifier/output_files/predictions.maf', sep = '\t', index = None)
    

    # exp.save_to_file('/home/dalalt1/PathoClass/somatic-germline/pathogenicity_classifier/fig.html')

    # exit()

    # experimenting with shapely values
    # explainer = shap.KernelExplainer(clf.predict_proba, shap.kmeans(X_train, 5))
    # shap_values = explainer.shap_values(X_train)
    # f = plt.figure()
    # shap.summary_plot(shap_values, X_train)
    # f.savefig(os.getcwd() + "/summary_plot1.png", bbox_inches='tight', dpi=600)

    # Use StratifiedKFold Cross Validation for an accurate representation of the model performance
    k_fold = StratifiedKFold(n_splits=10, random_state=42, shuffle = True)

    scoring = { 'accuracy' : make_scorer(accuracy_score), 
                'precision' : make_scorer(precision_score),
                'recall' : make_scorer(recall_score), 
                'f1_score' : make_scorer(f1_score)}

    results_kfold = cross_validate(clf, X_train,y_train, cv = k_fold, scoring = scoring, return_train_score = True)
    print('CV with sample distributions held across folds.')
    print("StratifiedKFold Train Mean Accuracy Across 10 Folds: %.5f (+/- %0.5f)" % (np.mean(results_kfold["train_accuracy"]), 2 * np.std(results_kfold["train_accuracy"])))
    print("StratifiedKFold Test Mean Accuracy Across 10 Folds: %.5f (+/- %0.5f)" % (np.mean(results_kfold["test_accuracy"]), 2 * np.std(results_kfold["test_accuracy"])))
    print("StratifiedKFold Test Mean Precision Across 10 Folds: %.5f (+/- %0.5f)" % (np.mean(results_kfold["test_precision"]), 2 * np.std(results_kfold["test_precision"])))
    print("StratifiedKFold Test Mean Recall Across 10 Folds: %.5f (+/- %0.5f)" % (np.mean(results_kfold["test_recall"]), 2 * np.std(results_kfold["test_recall"])))
    print("StratifiedKFold Test Mean F1 Score Across 10 Folds: %.5f (+/- %0.5f)\n" % (np.mean(results_kfold["test_f1_score"]), 2 * np.std(results_kfold["test_f1_score"])))

    # k_fold = KFold(n_splits=10)

    # cv_score = cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='accuracy')
    # print("\nAccuracy: %0.4f (+/- %0.4f)" % (cv_score.mean(), cv_score.std() * 2))

    # cv_score = cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='precision')
    # print("\nPrecision: %0.4f (+/- %0.4f)" % (cv_score.mean(), cv_score.std() * 2))

    # cv_score = cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='recall')
    # print("\nRecall: %0.4f (+/- %0.4f)" % (cv_score.mean(), cv_score.std() * 2))
    # exit()

    #get feature importances
    coeffs = np.array(clf.feature_importances_)

    # create a dictionary 
    feature_labels = X_train.columns.tolist()
    print("\nRandom Forest Feature importances")
    feature_importance = {}
    for i in range(len(feature_labels)):
            feature_importance[feature_labels[i]] = float(coeffs[i])
        
    # print the various features and their importance levels   
    for feature in sorted(feature_importance, key=feature_importance.get, reverse=True):
        print("%-40s %20.5f" % (feature,feature_importance[feature]))
    
    # remove all features with less than 1% significance level
    list_keepcols = []
    for feature in sorted(feature_importance, key=feature_importance.get, reverse=True):
        if(feature_importance[feature]>0.01):
            list_keepcols.append((feature,feature_importance[feature]))
    
    # sort the list of features by their feature importance in descending order
    list_keepcols.sort(key=lambda x: x[1], reverse=True)

    # print out the list of the features with their feature importances
    num = 1
    print("\n\ntop features")
    for w in list_keepcols:
        print("%d. %-40s %20.5f" % (num, w[0],w[1]))
        num+=1
    print('')
    
    # ############################################################################
    f = [tup[0] for tup in list_keepcols]
    imp = [tup[1] for tup in list_keepcols]
    indices = np.argsort(imp)

    importances = []
    for i in indices:
        importances.append(imp[i])

    style.use('seaborn-poster')
    matplotlib.rcParams['font.family'] = "sans-serif"
    fig, ax = plt.subplots()

    plt.title('Feature Importances')
    ax.barh(range(len(indices)), importances, color='tab:blue', align='center')
    plt.yticks(range(len(indices)), [f[i] for i in indices])

    for i, v in enumerate(importances):
        plt.text(v - 0.011, i, " "+str(round(v, 3)), color='white', verticalalignment="center",horizontalalignment="left")

    plt.xlabel('Relative Importance')
    fig.savefig('feature_importances.jpg', dpi = 500, bbox_inches = 'tight')

    # ############################################################################

    # keep only the subset of features deemed useful
    X_test_trimmed = X_test[keep_columns]

    # convert categorical variables into binary features (dummy encoding)
    X_test_trimmed = pd.get_dummies(X_test_trimmed)

    # change NaN values to 0s
    X_test_trimmed = X_test_trimmed.fillna(value=0)

    # make sure that the data we pass to the classifier has the same number of columns as the classifier expects
    try:
        X_test_trimmed = X_test_trimmed[final_columns]
    except KeyError:
        for colname in final_columns:
            if(colname not in X_test_trimmed.columns.tolist()):
                X_test_trimmed[colname] = 0
        X_test_trimmed = X_test_trimmed[final_columns]


    df_predictions_prob = clf.predict_proba(X_test_trimmed)
    df_predictions_prob = get_df_prob(df_predictions_prob)
    df_predictions = clf.predict(X_test_trimmed)
    # explainer = lime_tabular.LimeTabularExplainer(X_test_trimmed.values, mode='classification',feature_names=X_test_trimmed.columns.tolist(), discretize_continuous=False, kernel_width=3)
    # l=[]
    # for n in range(0,X_test_trimmed.shape[0]):
    #     exp = explainer.explain_instance(X_test_trimmed.values[n], clf.predict_proba, num_features=32)
    #     features = exp.as_list(exp.available_labels()[0])
    #     a=dict([(feature[0] + '_weight', feature[1]) for feature in features])
    #     l.append(a)
    # W_pick = pd.DataFrame(l).fillna(0)

    # explainer = shap.KernelExplainer(clf.predict_proba, shap.kmeans(X_train, 100))
    # shap_values = explainer.shap_values(X_test_trimmed[:5])
    # print(type(shap_values))
    # exit(0)
    # W_pick = create_prediction_factors_df(shap_values, X_test_trimmed)
    # print(W_pick.head())
    # print(W_pick.shape)
    # exit()

    tc_exps, p0  = rf_explain(clf, X_test_trimmed)
    W_pick = pd.DataFrame(tc_exps, columns = [x + '_weight' for x in X_test_trimmed.columns.tolist()])
    
    e = pd.Series(df_predictions)
    f = pd.Series(df_predictions_prob['prediction_prob'])
    X_test['prediction'] = e.values
    X_test['prediction_probability'] = f.values
    
    oncogenes, tumor_suppressors, gene_level_annotation = get_gene_level_annotations()
    
    #identify truncating mutations in oncogene
    X_test['truncating_oncogene'] = np.where((X_test['Hugo_Symbol'].isin(oncogenes))&
                                                    (~(X_test['Hugo_Symbol'].isin(tumor_suppressors))) &
                                        (X_test['Variant_Classification'].isin([
                    'Nonsense_Mutation', 'Frame_Shift_Ins', 'Frame_Shift_Del', 'Splice_Site'
                ])), 1, 0)

    X_test['prediction'] = np.where(X_test['truncating_oncogene']==1,
                                            0, X_test['prediction'])

    # f_cols = ['Chromosome', 'Start_Position', 'End_Position', 'Reference_Allele', 'Alternate_Allele', 'Variant_Type', 'Variant_Classification', 'Hugo_Symbol', 'HGVSc', 'Protein_position', 'HGVSp_Short', 'Normal_Sample', 'n_alt_count', 'n_depth', 'ExAC2_AF', 'ExAC2_AF_ASJ', 'Consequence', 'CLINICAL_SIGNIFICANCE', 'GOLD_STARS', 'Exon_Number', 'oncogenic', 'last_exon_terminal', 'truncating_oncogene','prediction_probability','prediction']

    # final_columns_to_report = [column for column in f_cols if column in X_test]

    # X_test[final_columns_to_report].to_csv(classifier_output, sep = '\t', index = None)
    # # final_columns_to_report = ['Chromosome', 'Start_Position', 'End_Position', 'Reference_Allele', 'Alternate_Allele', 'Variant_Type', 'Variant_Classification', 'Hugo_Symbol', 'HGVSc', 'Protein_position', 'HGVSp_Short', 'Normal_Sample', 'n_alt_count', 'n_depth', 'ExAC2_AF', 'ExAC2_AF_ASJ', 'Consequence', 'CLINICAL_SIGNIFICANCE', 'GOLD_STARS', 'Exon_Number', 'oncogenic', 'last_exon_terminal', 'prediction', 'prediction_probability', 'truncating_oncogene',]
    # # .extend(W_pick.columns.tolist())

    # # if 'exon_number' not in X_test and 'Consequence' not in X_test:
    # #     final_columns_to_report = ['Chromosome', 'Start_Position', 'End_Position', 'Reference_Allele', 'Alternate_Allele', 'Variant_Type', 'Variant_Classification', 'Hugo_Symbol', 'HGVSc', 'Protein_position', 'HGVSp_Short', 'Normal_Sample', 'n_alt_count', 'n_depth', 'ExAC2_AF', 'ExAC2_AF_ASJ', 'CLINICAL_SIGNIFICANCE', 'GOLD_STARS', 'oncogenic', 'prediction', 'prediction_probability', 'truncating_oncogene',]
    # #     final_columns_to_report.extend(W_pick.columns.tolist())
    engineered_cols = sorted(X_test_trimmed.columns.tolist() + W_pick.columns.tolist())
    orig_cols_subset = ['Chromosome', 'Start_Position', 'End_Position', 'Reference_Allele', 'Alternate_Allele', 'Variant_Type', 'Variant_Classification', 'Hugo_Symbol', 'HGVSc', 'Protein_position', 'HGVSp_Short', 'Normal_Sample', 'n_alt_count', 'n_depth', 'CLINICAL_SIGNIFICANCE']
    added_cols = ['truncating_oncogene', 'prediction_probability', 'prediction']
    
    total_columns = orig_cols_subset + engineered_cols + added_cols

    # # # print(X_test[orig_cols_subset + added_cols].shape)
    # # # print(X_test_trimmed.shape)
    # # # print(W_pick.shape)

    X_test_extended = pd.concat([X_test[orig_cols_subset + added_cols], X_test_trimmed],axis = 1)

    X_test_extended.index = W_pick.index
    X_test_extended_preds = pd.concat([X_test_extended, W_pick], axis = 1)
    
    X_test_extended_preds[total_columns].to_csv(classifier_output, sep = '\t', index = None)

if __name__ == '__main__':
  main()  

