import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.pylab import rcParams
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, \
confusion_matrix, mean_squared_error, balanced_accuracy_score,roc_curve,auc

def performance_metrics(y_true, y_pred, threshold=0.5):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    y_pred_bin = np.round(y_pred)
    #y_pred_bin = [0 if y < threshold else 1 for y in y_pred]
    d={}
    d['accuracy'] = accuracy_score(y_true, y_pred_bin)
    d['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred_bin)
    d['precision'] = precision_score(y_true, y_pred_bin)
    d['recall'] = recall_score(y_true, y_pred_bin)
    d['RMSE'] = mean_squared_error(y_true, y_pred, squared=False)
    cm = confusion_matrix(y_true, y_pred_bin)
    cm = pd.DataFrame(cm, index=[['Observed','Observed'],['False','True']],\
                      columns=[['Predicted','Predicted'],['False','True']])
    text = 'Performance Metrics\n'
    p = 20
    for i,j in d.items():
        text += '\n{}{} = {:.3f}'.format(i.title(),' '*(p-len(i)),j)
    print(text,'\n\nConfusion Matrix')
    return print(cm)

def plot_auc2(ax, y_train, y_train_pred, y_test, y_test_pred, th=0.5):

    y_train_pred_labels = (y_train_pred>th).astype(int)
    y_test_pred_labels  = (y_test_pred>th).astype(int)

    fpr_train, tpr_train, _ = roc_curve(y_train,y_train_pred)
    ks_train = max(tpr_train-fpr_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    acc_train = accuracy_score(y_train, y_train_pred_labels)

    fpr_test, tpr_test, _ = roc_curve(y_test,y_test_pred)
    ks_test = max(tpr_test-fpr_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    acc_test = accuracy_score(y_test, y_test_pred_labels)

    ax.plot(fpr_train, tpr_train,color="blue")
    ax.plot(fpr_test, tpr_test,color="red")

    ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    
    train_text = 'Train KS = {:.2f}, AUC = {:.2f}'.format(ks_train, roc_auc_train)
    test_text = 'Test KS = {:.2f}, AUC = {:.2f}'.format(ks_test, roc_auc_test)
    no_skill_text = 'no skill'
    ax.legend([train_text, test_text, no_skill_text])
    
def plot_auc1(y_true,y_pred,model_name='Model'):
    y_true = np.ravel(y_true)
    # for i in array of y_pred
    y_pred = np.ravel(y_pred)
    fpr,tpr,thresholds = roc_curve(y_true,y_pred)
    ks = max(tpr-fpr)
    AUC = roc_auc_score(y_true,y_pred)
    plt.plot(fpr,tpr,color='tab:blue',label='{}\nAUC = {}\nKS = {}'.format(model_name,round(AUC,3),round(ks,3)))
    
    plt.plot([0,1],[0,1],linestyle='--',color='black',label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    return

def check_missing(data):
    MissTotal = data.isnull().sum().sort_values(ascending=False)
    percent = round((data.isnull().sum()/list(data.shape)[0]*100),2).sort_values(ascending=False)
    count = data.isnull().count().sort_values(ascending=False)
    missing_data = pd.concat([MissTotal, percent, count], axis=1, keys=['MissingTotal', 'MissingPercent', 'Total'])
    print(missing_data[missing_data['MissingPercent']>0].head(100))
    
def univariate(df,feature,target):
    
    import sys
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    
    #print(df.info())
    
    #CREATE THE SUMMARY TABLE
    
    non_events = pd.DataFrame(df[df[target]==0].groupby([feature],dropna=False)[target].count()).rename(columns={target: 'non_events'})
    non_events_total = df[df[target]==0].groupby([target],dropna=False)[target].count()
    events = pd.DataFrame(df[df[target]==1].groupby([feature],dropna=False)[target].count()).rename(columns={target: 'events'})
    events_total = df[df[target]==1].groupby([target],dropna=False)[target].count()
    grand_total = df[target].count()
    result = pd.merge(non_events, events, how="left",on=feature)
    result['total'] = result['non_events'] + result['events']
    result['event rate %'] = round(result['events']/result['total']*100,1)
    result['% event dist'] = result['events']/(events_total.values)
    result['% non-event dist'] = result['non_events']/(non_events_total.values)
    result['% total dist'] = round(result['total']/grand_total*100,1)
    cond = ((result['% non-event dist']==0) | (result['% event dist'] ==0)) # Dont calculate WOE and IV if the bin has missing values of events and/or non-events
    result.loc[~cond,"WoE"] = np.log(result.loc[~cond,"% non-event dist"]/result.loc[~cond,"% event dist"])
    result.loc[~cond,"IV"] = (result['% non-event dist']-result['% event dist'])*result['WoE']

    # Calculate IV
    IV = round(result.IV.sum(),6)
    #print('Information Value of',"'"+feature+"'",'is', IV)
    
    #round off the decimal places for better display
    result['% event dist'] = round(result['% event dist']*100,1)
    result['% non-event dist'] = round(result['% non-event dist']*100,1)
    result['WoE'] = round(result['WoE'],2)
    result['IV'] = round(result['IV'],2) 
    
    # convert index (feature) to column
    result.reset_index(inplace=True)
    
    # Replace nan with Missing
    result[feature]=result[feature].replace(np.nan, 'Missing', regex=True)
    
    # PLOT THE CHART OF THE FEATURE BINS AND EVENT RATE
    
    if df[feature].dtype == 'object':
        #result = result.sort_values('IV', ascending = False).head(30).reset_index() 
        result = result.head(30).reset_index() 
    
    # convert from a wide to long format
    dfl = pd.melt(result, id_vars=feature,var_name='Type', value_name="Values")

    # select the desired data
    dist = dfl[dfl.Type.str.contains('total dist')]
    rate = dfl[dfl.Type.str.contains('rate')]
    
    # create the figure and primary axes
    fig, ax = plt.subplots(figsize=(6, 3))

    Title1 = "Distribution by " + feature +' : IV - ' + str(round(IV,4))
    plt.title(Title1, y=1.15, color='dimgrey', fontdict= { 'fontsize': 18, 'fontweight':'bold'})

    # plot and format the bars
 
    # Set custom color palette
    color1 = "steelblue"
    colors = [color1] #add more colors to the list if you have more bars to display
    customPalette = sns.set_palette(sns.color_palette(colors))
    sns.barplot(data=dist, x=feature, y='Values', hue='Type',palette=customPalette,alpha=0.6)

    ax.yaxis.grid(True,alpha=0.5)

    #ax.set_xlabel(feature, color='dimgrey',fontdict= { 'fontsize': 12, 'fontweight':'bold'})
    ax.xaxis.label.set_visible(False)
    #ax.set_xlabel(feature, color='dimgrey',fontdict= { 'fontsize': 12, 'fontweight':'bold'})
    ax.set_ylabel('% Dist', color='steelblue',fontdict= { 'fontsize': 12, 'fontweight':'bold'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    l1 = ax.legend(bbox_to_anchor=(0.0, 1.0), loc='upper center')

    #ax.xaxis.grid(True) # Show the vertical gridlines

    # create the secondary axex
    ax2 = ax.twinx()

    # plot and format the line
    sns.lineplot(data=rate, x=feature, y='Values', ax=ax2, color='indianred', label='event rate %', marker='o',alpha=0.6)
    ax2.set_ylabel('% Event Rate',color='indianred',fontdict= { 'fontsize': 12, 'fontweight':'bold'})
    #ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    l2 = ax2.legend(bbox_to_anchor=(0.0, 0.9), loc='upper left')
    
    # custom combined legend
#     plt.legend(l1.get_patches() + l2.get_lines(), 
#            [text.get_text() for text in l1.get_texts() + l2.get_texts()], 
#            bbox_to_anchor=(0.8, -0.3), ncol=3, fontsize=12)
    
    plt.legend(l1.get_patches() + l2.get_lines(), 
           [text.get_text() for text in l1.get_texts() + l2.get_texts()], 
           ncol=3, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.12))
    
    # remove l1 from the plot
    l1.remove()

    # Turn off secondary y axis grid
    ax2.grid(False)

    # annotate the line
    for _, x, _, y in rate.itertuples():
        ax2.text(x=x, y=y, s = '{:.1f}'.format(y), color='darkred', fontsize = 10)
        
    return result, IV

def exp_vs_act(df,score,target):
    
    df[score] = round(df[score],3)
    
    #print(df.info())
    orig_feature = score
    
    #CREATE THE SUMMARY TABLE
    if str(df[orig_feature].dtype) !='object':
        grp_range = 'grp_' + score
        df[grp_range] = pd.qcut(df[score], q=10, labels=None, retbins = False,  duplicates = 'raise')
        df['grp_num'] = (pd.qcut(df[score], q=10, labels=False, retbins = False,  duplicates = 'raise')+1).astype(str).str.zfill(2)
        df['grp'] = df['grp_num'].astype(str) + ' ' +df[grp_range].astype(str)
        score = 'grp'
        
    non_events = pd.DataFrame(df[df[target]==0].groupby([score],dropna=False)[target].count()).rename(columns={target: 'non_events'})
    non_events_total = df[df[target]==0].groupby([target],dropna=False)[target].count()
    events = pd.DataFrame(df[df[target]==1].groupby([score],dropna=False)[target].count()).rename(columns={target: 'events'})
    events_total = df[df[target]==1].groupby([target],dropna=False)[target].count()
    grand_total = df[target].count()
    result = pd.merge(non_events, events, how="left",on=score)
    
    if str(df[orig_feature].dtype) !='object':
        pred_events = round(pd.DataFrame(df.groupby([score],dropna=False)[orig_feature].sum()),1).rename(columns={orig_feature: 'pred_events'})
        result = pd.merge(result, pred_events, how="left",on=score)
        
    result['total'] = result['non_events'] + result['events']
    result['act_event_rate'] = round(result['events']/result['total'],3)
    if str(df[orig_feature].dtype) !='object':
        result['pred_event_rate'] = round(result['pred_events']/result['total'],3)
    result['% event dist'] = result['events']/(events_total.values)
    result['% non-event dist'] = result['non_events']/(non_events_total.values)
    result['% total dist'] = round(result['total']/grand_total*100,1)
    cond = ((result['% non-event dist']==0) | (result['% event dist'] ==0)) # Dont calculate WOE and IV if the bin has missing values of events and/or non-events
    result.loc[~cond,"WoE"] = np.log(result.loc[~cond,"% non-event dist"]/result.loc[~cond,"% event dist"])
    result.loc[~cond,"IV"] = (result['% non-event dist']-result['% event dist'])*result['WoE']

    # Calculate IV
    IV = round(result.IV.sum(),6)
    #print('Information Value of',"'"+score+"'",'is', IV)
    
    #round off the decimal places for better display
    result['% event dist'] = round(result['% event dist']*100,1)
    result['% non-event dist'] = round(result['% non-event dist']*100,1)
    result['WoE'] = round(result['WoE'],2)
    result['IV'] = round(result['IV'],2) 
    
    # convert index (score) to column
    result.reset_index(inplace=True)
    
    # Replace nan with Missing
    result[score]=result[score].replace(np.nan, 'Missing', regex=True)
    
    ax = result.plot(x='grp', y=['act_event_rate','pred_event_rate'],marker="*")
    ax.set_xticks(range(len(result)))
    p = ax.set_xticklabels([item for item in result.grp.tolist()],rotation=45)
    return result

