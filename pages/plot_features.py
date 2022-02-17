import numpy as np
import pandas as pd

import scipy as sp
from scipy import stats
from scipy.stats import fisher_exact
from scipy.stats import ranksums

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import sklearn
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

def plot_pca_2_components_normalize_scale(df, features, y_cat):
    '''
    Input: df:dataframe, features: feature col names as a list, y_cat: col containing treatment flag
    '''
    pca = PCA(n_components=2)
    # without scaling
    #principalComponents = pca.fit_transform(df[features])
    
    # normalize and scale
    Xn = normalize(df[features])
    XsXn = scale(Xn)
    pca = PCA(2)
    principalComponents = pca.fit_transform(XsXn)
    
    
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    plotDf = pd.concat([principalDf, df[y_cat].replace({0:'non-response',1:'response'})], axis=1)

    fig = plt.figure(figsize = (6,4))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('PCA for top features', fontsize = 20)

    targets = ["response","non-response"]
    colors = ['red','blue']
    for target, color in zip(targets,colors):
        indicesToKeep = plotDf['Treatment_Flag'] == target
        ax.scatter(plotDf.loc[indicesToKeep, 'principal component 1'], plotDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 20)

    ax.legend(targets)
    ax.grid()
    
    return fig


def calculate_feature_impt_tree_based(df, features, y_cat):
    clf = ExtraTreesClassifier()
    clf = clf.fit(df[features], df[y_cat])
    impt = clf.feature_importances_
    #f_top1, f_top2, f_top3 = df_feature_impt['feature'].iloc[0],df_feature_impt['feature'].iloc[1],df_feature_impt['feature'].iloc[2]
    df_feature_impt = pd.DataFrame({'feature': features,'impt': impt,}).sort_values(by=['impt'],ascending=False)
    
    return df_feature_impt


def feature_spearman_corr_plot(df, features):
    spearman_matrics = df[features].corr(method='spearman')

    mask = np.zeros_like(spearman_matrics, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 10))

    fig = sns.heatmap(spearman_matrics, vmin=spearman_matrics.values.min(), mask = mask, vmax=1, square=True, 
            linewidths=0.1, annot=True, annot_kws={"size":10}, cmap="BuPu")
    
    return fig


def categorical_feature_contingency_table(df, cat_feature,y_cat):
    '''
    cat_feature = 'SEXN' (use string)
    y_cat = 'Treatment_Flag' (use string)
    '''

    fig = plt.figure(num=None, figsize=(4, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    data_crosstab = pd.crosstab(df[cat_feature],df[y_cat], margins = False)
    sns.heatmap(data_crosstab, annot=True, fmt='.2f')
    if data_crosstab.shape[0] == 2:
        oddsr, p = fisher_exact(table=data_crosstab.to_numpy())
        title = "Fisher's Exact p-value: " + "{:.2f}".format(p)
        plt.title(title)

    return plt


def continous_feature_ranksum_test_and_plot(df, cont_var,y_cat):
    group1 = df[df[y_cat]==1][cont_var]
    group2 = df[df[y_cat]==0][cont_var]
    stat, p = ranksums(group1, group2)

    fig = plt.figure(figsize=(6, 5))
    title = cont_var + " between response groups\n" + "ranksum p-value: " + "{:.4f}".format(p)

    ax = sns.boxplot(x=y_cat, y=cont_var, data=df).set(title=title)
    ax = sns.swarmplot(x=y_cat, y=cont_var, data=df, color=".25")

    return plt


### Test examples
df = pd.read_csv('../data/dataset_clean.csv')
features = ['SEXN', 'MADCAM1', 'RDW_base', 'STOOLFN','PLATE_base', 'LDH_base', 'CREATN_base', 'BEMCS', 'BMCS', 'TPROT_base', \
    'BFECAL', 'BCRP']
y_cat = ['Treatment_Flag']
y_cont = ['CTR1']
cat_features = ['SEXN', 'STOOLFN','BEMCS']
cont_features = ['MADCAM1', 'RDW_base', 'PLATE_base', 'LDH_base', 'CREATN_base', 'BMCS', 'TPROT_base', \
    'BFECAL', 'BCRP']




test_scale_normalize = plot_pca_2_components_normalize_scale(df, features, y_cat)
spearman_plot = feature_spearman_corr_plot(df, features)
feature_impt_table = calculate_feature_impt_tree_based(df, features, y_cat)

cat_feature = 'STOOLFN'
y_cat = 'Treatment_Flag'
categorical_feature_contingency_table(df, 'STOOLFN', y_cat)
categorical_feature_contingency_table(df, 'SEXN', y_cat)
categorical_feature_contingency_table(df, 'BEMCS',y_cat)

continous_feature_ranksum_test_and_plot(df, 'MADCAM1', y_cat)
continous_feature_ranksum_test_and_plot(df, 'BMCS', y_cat)

