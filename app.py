# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:34:36 2020

@author: alfid
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV




import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('webpic.jfif')


title = '''

<style>
h1 {
  color: black;
  text-shadow: 2px 2px 5px red;
}
<h1>Financial Analysis</h1>
</style>
'''
st.markdown(title, unsafe_allow_html=True)


st.write(""" 
         

                             # *Financial Analysis*
    
    
    ------------------------

   Our client has a **credit card company**. They have brought us a dataset that includes some demographics and recent financial data (the past six months) for a sample of **30,000** of their account holders. This data is at the credit account level; in other words, there is one row for each account. Rows are labeled by whether in the next month after the six month historical data period, an account owner has defaulted, or in other words, failed to make the minimum payment.
   
   **`Note: Click on the    " > "    from the top_left of this page to view the process.`**
   
   
""")

st.sidebar.header('Work Flow')
st.sidebar.subheader('Click the checkbox to view the process')
df1 = pd.read_excel('default_of_credit_card_clients.xls')
df = pd.read_csv('cleaned_data.csv')
if st.sidebar.checkbox('1: Data Exploration'):
    st.title('Data Exploration')
    st.write('''
             Now that we\'ve understood the business problem and have an idea of what is supposed to be in the data, we can compare these impressions to what we actually see in the data. data exploration is to not only look through the data both directly and using numerical and graphical summaries, but also to think critically about whether the data make sense and match what we have been told about them.
    
    Below showing the dataset given by our client, the data should be cleaned yet''')
    st.write(df1.head())
    st.write('''
             we had performed a basic check on whether our dataset contains what we expected and verified whether there are the correct number of samples.
             The data are supposed to have observations for 30,000 credit accounts. While there are 30,000 rows, we had also checked whether there are 30,000 unique account IDs.
    
    The below showing IDs are duplicated in the dataset''')
    id_counts = df1['ID'].value_counts()
    dupe_mask = id_counts == 2
    dupe_ids = id_counts.index[dupe_mask]
    dupe_ids = list(dupe_ids)
    st.write(df1.loc[df1['ID'].isin(dupe_ids[0:3]),:].head(10))
    st.write("""
             We spent nearly all of this lesson identifying and correcting issues with our dataset. In our data exploration, we discovered an issue that could have undermined our project: the data we had received was not internally consistent. Most of the months of the payment status features were plagued by a data reporting issue, included nonsensical values, and were not representative of the most recent month of data, or the data that would be available to the model going forward. We only uncovered this issue by taking a careful look at all of the features.
             
    
    After checking out our dataset and modiying some features from the dataset like **PAY_1 ,PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, EDUCATION** and **MARRIAGE** we stored the cleaned data into the csv file and named it as **cleaned_data.csv**. 
             
             The cleaned data will look like
             
             
             """)
    st.write(df.head())
if st.sidebar.checkbox('2:Exploring Remaining Financial Features in the Dataset'):
    st.title("Exploring Remaining Financial Features in the Dataset")
    st.write('In the previous section we cleaned some features like **PAY_1 , EDUCATION** and **MARRIAGE** so in this part we have examined some remaining features like **BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6** and **PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6**')
    bill_feats = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_feats = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    st.write('**Exploring the BILL_AMT features**')
    st.write(df[bill_feats].describe())
    st.write('showing the histogram of the **BILL_AMT** features')
    st.write(df[bill_feats].hist(bins = 20, layout = (2,3)))
    st.pyplot()
    st.write('**Exploring the PAY_AMT features**')
    st.write(df[pay_amt_feats].describe())
    st.write('After cleaning the zero values from **PAY_AMT** features the histogram will look like')
    zero_pay_amt = df[pay_amt_feats] == 0
    st.write(df[pay_amt_feats][~zero_pay_amt].apply(np.log10).hist(bins = 20, layout= (2,3)))
    st.pyplot()

if  st.sidebar.checkbox('3: Performing Logistic Regression'):
    st.title('**Performing Logistic Regression**')
    st.write('*Performing Logistic Regression with a New Feature and Creating a Precision-Recall Curve*')
    st.write(' In this part we have trained a logistic regression using a feature besides  **LIMIT_BAL**. Then we have graphically assessed the tradeoff between precision and recall, as well as calculated the area underneath a precision-recall curve.')
    
    X_train , X_test, y_train, y_test = train_test_split(df['LIMIT_BAL'].values.reshape(-1,1), df['default payment next month'].values, test_size = 0.2, random_state = 24)
    lg = LogisticRegression()
    lg.C = 0.1
    trained = lg.fit(X_train ,y_train)
    pred = trained.predict_proba(X_test)
    pos_prob =pred[:,1]
    fpr, tpr, threshold = roc_curve(y_test,pos_prob)
    st.write(""" 
             
             **Performing Logistic Regression** on the cleaned data to calculated the **Predicted-Probability** and then to calculated the **roc_auc_curve** and **precision_recall_curve**.
             
             
             The **roc_auc_score** for the training data is shown below.
             
             
    """)
    train_pred = trained.predict_proba(X_train)
    st.write(roc_auc_score(y_train, train_pred[:,1]))
    st.write('The **roc_auc_score** for the test data is shown below')
    st.write(roc_auc_score(y_test, pos_prob))
    st.write('The **ROC Curve** is shown below')
    plt.plot(fpr, tpr, '*-')
    plt.plot([0,1],[0,1],'--')
    plt.legend(['Logistic Regression', 'Random Chance'])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    st.pyplot()
    precision, recall, _ = precision_recall_curve(y_test, pos_prob)
    st.write('The **precision_recall_curve** is shown below')
    plt.plot(recall, precision, 'r^-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('precision_recall curve')
    st.pyplot()
if st.sidebar.checkbox('4: Fitting a Logistic Regression Model'):
    st.title('**Fitting a Logistic Regression Model**')
    st.write("*Fitting a Logistic Regression Model and Directly Using the Coefficients*")
    st.write(''' In this part we have trained a logistic regression model on the two most important features **PAY_1, LIMIT_BAL**we discovered in univariate feature exploration.
             Then we began an in-depth description of how logistic regression works, exploring such topics as the **Sigmoid function**, **log odds**, and the **linear decision boundary**.
    ''')
    st.write('Here we have used **Sigmod function** to manually calculate the **roc_auc_score** using the co-efficients and intercepts')
    r'''
    
    ### **Sigmoid function :**    **$$f(x) = \frac{1} {1 + e^{-X}}$$**
    '''
    def sigmoid(X):
        f = 1 / (1 + np.exp(-X))
        return f
    X = df.loc[:,['PAY_1', 'LIMIT_BAL']]
    y = df['default payment next month']
    lg = LogisticRegression()
    trainx , testx, trainy, testy = train_test_split(X,y, random_state = 24, test_size = 0.2)
    lr_fit = lg.fit(trainx,trainy)
    lr_fit.predict(testx)
    X['intercept_col'] = 1
    st.write('Then, to pull out the coefficients and intercept from the trained model and manually calculate predicted probabilities. we have added a column of 1s to our features, to multiply by the intercept.')
    st.write(X)
    coef1 = lr_fit.coef_[0][0]
    coef2 = lr_fit.coef_[0][1]
    intercept = lr_fit.intercept_
    fun = intercept * X['intercept_col'] + coef1 * X['PAY_1'] + coef2 * X['LIMIT_BAL']
    manually_predicted_probabilities = sigmoid(fun)
    st.write('**roc_auc_score** from the *Logistic Regression\'s* predicted probabilities')
    st.write(roc_auc_score(testy, lr_fit.predict(testx)))
    st.write('**roc_auc_score** from the manally calculation using *Sigmoid function*')
    st.write(roc_auc_score(testy, manually_predicted_probabilities[:5333]))
             
    
    st.write('So here we can conclude the manually calculated *roc_auc_score* and the *roc_auc_score* calculated from the Logistic Regression\'s proctions are similar.')
if st.sidebar.checkbox('5: Cross Validation and Feature Engineering'):
    st.title('**Cross Validation and Feature Engineering**')
    st.write('In this part we have applied the knowledge of cross-validation and regularization. We have performed basic feature engineering. In order to estimate parameters for the regularized logistic regression model for the case study data, which is larger in size than the synthetic data that we\'ve worked with.')
    features_list = ['LIMIT_BAL','EDUCATION','MARRIAGE','AGE','PAY_1','BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    x = df.loc[:,features_list]
    Y = df['default payment next month']
    featx, respx, featy, respy = train_test_split(x, Y, test_size = 0.2, random_state = 24)
    scaler = MinMaxScaler().fit(x,Y)
    scalar1 = MinMaxScaler().fit(x)
    xx = scalar1.transform(x)
    l = LogisticRegression(solver = 'saga', penalty = 'l1', max_iter = 1000)
    pipeline = Pipeline(steps=[('scaler', scaler), ('model', l)])
    pipeline.set_params(model__C = 2)
    C_val_exponents = np.linspace(2,-3,6)
    C_vals = np.float(10)**C_val_exponents
    k_folds = StratifiedKFold(n_splits = 4, random_state = 1)
    def cross_val_C_search_pipe(k_folds, C_vals, model, X, y):
        n_folds = k_folds.n_splits
        cv_train_roc_auc = np.empty((n_folds, len(C_vals)))
        cv_test_roc_auc = np.empty((n_folds, len(C_vals)))
        cv_test_roc = [[]]*len(C_vals)
        for c_val_counter in range(len(C_vals)):
            #Set the C value for the model object
            model.C = C_vals[c_val_counter]
            #Count folds for each value of C
            fold_counter = 0
            #Get training and testing indices for each fold
            for train_index, test_index in k_folds.split(X, y):
                #Subset the features and response, for training and testing data for
                #this fold
                X_cv_train, X_cv_test = X[train_index], X[test_index]
                y_cv_train, y_cv_test = y[train_index], y[test_index]
                #Fit the model on the training data
                model.fit(X_cv_train, y_cv_train)
                #Get the training ROC AUC
                y_cv_train_predict_proba = model.predict_proba(X_cv_train)
                cv_train_roc_auc[fold_counter, c_val_counter] = roc_auc_score(y_cv_train, y_cv_train_predict_proba[:,1])
                #Get the testing ROC AUC
                y_cv_test_predict_proba = model.predict_proba(X_cv_test)
                cv_test_roc_auc[fold_counter, c_val_counter] = roc_auc_score(y_cv_test, y_cv_test_predict_proba[:,1])
                #Testing ROC curves for each fold
                this_fold_roc = roc_curve(y_cv_test, y_cv_test_predict_proba[:,1])
                cv_test_roc[c_val_counter].append(this_fold_roc)
                #Increment the fold counter
                fold_counter += 1
                #Indicate progress
            #print('Done with C = {}'.format(model.C))
        return cv_train_roc_auc, cv_test_roc_auc, cv_test_roc



    cv_train_roc_auc, cv_test_roc_auc, cv_test_roc = cross_val_C_search_pipe(k_folds, C_vals, pipeline, xx, Y)
    st.write(''' **Cross validation scores for each fold**
             
    
    We have used 4 folds to train and test our Logistic Regression model and calculated the roc_auc_curve the below figure describes the process.
    
    ''')
    for this_fold in range(4):
        plt.plot(C_val_exponents, cv_train_roc_auc[this_fold], '-o', label='Training fold {}'.format(this_fold+1))
        plt.plot(C_val_exponents, cv_test_roc_auc[this_fold], '-x', label='Testing fold {}'.format(this_fold+1))
    plt.ylabel('ROC AUC')
    plt.xlabel('log$_{10}$(C)')
    plt.legend(loc = [1.1, 0.2])
    plt.title('Cross validation scores for each fold')
    st.pyplot()
    st.write('Then we have calculated the **Cross validation scores averaged over all folds** and it looks like.')
    plt.plot(C_val_exponents, np.mean(cv_train_roc_auc, axis=0), '-o',
        label='Average training score')
    plt.plot(C_val_exponents, np.mean(cv_test_roc_auc, axis=0), '-x',
            label='Average testing score')
    plt.ylabel('ROC AUC')
    plt.xlabel('log$_{10}$(C)')
    plt.legend()
    plt.title('Cross validation scores averaged over all folds')
    st.pyplot()
if st.sidebar.checkbox('6: Cross Validation Grid Search with Random Forest'):
    st.title('Cross Validation Grid Search with Random Forest')
    st.write('In this Part we have conducted a grid search over the **number of trees** in the forest **(n_estimators)** and the **maximum depth** of a tree **(max_depth)** for a random forest model on the case study data. We have then created a visualization showing the average testing score for the grid of hyperparameters that we have searched over.')
    features_response = df.columns.tolist()
    items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                       'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                       'others', 'university']
    features_response = [item for item in features_response if item not in items_to_remove]
    X_tran, X_tst, y_tran, y_tst = train_test_split(
    df[features_response[:-1]].values,
    df['default payment next month'].values,
    test_size=0.2, random_state=24)
    rf = RandomForestClassifier(
        n_estimators=10, criterion='gini', max_depth=3,
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1,
        random_state=4, verbose=0, warm_start=False, class_weight=None)
    params = {
        'max_depth': [3,6,9,12],
        'n_estimators': [10,50,100,200]
        }
    cv = GridSearchCV(rf, param_grid=params, scoring='roc_auc',
                        n_jobs=None, iid=False, refit=True, cv=4, verbose=2,
                        pre_dispatch=None, error_score=np.nan, return_train_score=True)
    cv.fit(X_tran, y_tran)
    cv_result = pd.DataFrame(cv.cv_results_)
    st.write('Here we have taken **Random Forest Classifier** for the cross validation and for the **Grid Search** we had set some range of hyperparameters that is **n_estimators** and **max_depth** to check the best combination of hyperparameters using the Grid Search. After fitting the model we had stored the result into the dataframe which is shown below')
    st.write(cv_result)
    st.write('Below figure will descibe how the depth of the tree will affect the ROC AUC.')
    ax = plt.axes()
    ax.errorbar(cv_result['param_max_depth'],
                cv_result['mean_train_score'],
                yerr=cv_result['std_train_score'],
                label='Mean $\pm$ 1 SD training scores')
    ax.errorbar(cv_result['param_max_depth'],
                cv_result['mean_test_score'],
                yerr=cv_result['std_test_score'],
                label='Mean $\pm$ 1 SD testing scores')
    ax.legend()
    plt.xlabel('max_depth')
    plt.ylabel('ROC AUC')
    st.pyplot()
    st.write('The best parameters are shown below')
    st.write(cv.best_params_)
    mean_test_score = cv_result['mean_test_score'].values.reshape(4,4)
    xx, yy = np.meshgrid(range(5), range(5))
    st.write('pcolormesh visualization of the mean testing score for each combination of hyperparameters')
    ax = plt.axes()
    pcolor_ex = ax.pcolormesh(xx, yy, mean_test_score, cmap=plt.cm.jet)
    plt.colorbar(pcolor_ex, label='Color scale')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    st.pyplot()
    feat_imp_df = pd.DataFrame({'Features': features_response[:-1], 'Importance': cv.best_estimator_.feature_importances_})
    st.write('Below dataframe will show all the feature based on the importance ')
    st.write(feat_imp_df.sort_values('Importance', ascending = False))    
if st.sidebar.checkbox('7: Deriving Financial Insights'):
    st.title('**Deriving Financial Insights**')
    st.write(''' 
             
             We ask the case study client about the points outlined above and learn the following: for credit accounts that are at a high risk of default, the client is designing a new program to provide individualized counseling for the account holder, to encourage them to pay their bill on time or provide alternative payment options if that will not be possible. Credit counseling is performed by trained customer service representatives who work in a call center. **The cost per attempted counseling session is NT$7,500** and the **expected success rate of a session is 70%**, meaning that on average 70% of the recipients of phone calls offering counseling will pay their bill on time, or make alternative arrangements that are acceptable to the creditor. The potential benefits of successful counseling are that the amount of an account's monthly bill will be realized as **savings**, if it was going to default but instead didn't, as a result of the counseling. Currently, the monthly bills for accounts that default are reported as **losses**.
             
             
             As we proceed to the financial analysis, we see that the decision that the model will help the client make, on an account by account basis, is a yes/no decision: whether to offer counseling to the holder of a given account. Therefore, our analysis should focus on **finding an appropriate threshold of predicted probability**, by which we may divide our accounts in to two groups: **higher risk** accounts that will receive counseling and **lower risk** ones that won't.
             
    ''')
    thresholds = np.linspace(0, 1, 101)
    features_response = df.columns.tolist()
    items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                       'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                       'others', 'university']
    features_response = [item for item in features_response if item not in items_to_remove]
    missing_pay_1_mask = df1['PAY_1'] == 'Not available'
    df_missing_pay_1 = df.loc[missing_pay_1_mask,:].copy()
    df_fill_pay_1_model = df_missing_pay_1.copy()
    X_trn, X_ts, y_trn, y_ts = train_test_split(
    df[features_response[:-1]].values,
    df['default payment next month'].values,
    test_size=0.2, random_state=24)
    df_fill_pay_1_model['PAY_1'] = np.zeros_like(df_fill_pay_1_model['PAY_1'].values)
    X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
    train_test_split(
        df_fill_pay_1_model[features_response[:-1]].values,
        df_fill_pay_1_model['default payment next month'].values,
    test_size=0.2, random_state=24)
    X_train_all = np.concatenate((X_trn, X_fill_pay_1_train), axis=0)
    X_test_all = np.concatenate((X_ts, X_fill_pay_1_test), axis=0)
    y_train_all = np.concatenate((y_trn, y_fill_pay_1_train), axis=0)
    y_test_all = np.concatenate((y_ts, y_fill_pay_1_test), axis=0)
    savings_per_default = np.mean(X_test_all[:, 5])
    cost_per_counseling = 7500
    effectiveness = 0.70
    n_pos_pred = np.empty_like(thresholds)
    cost_of_all_counselings = np.empty_like(thresholds)
    n_true_pos = np.empty_like(thresholds)
    savings_of_all_counselings = np.empty_like(thresholds)
    random = RandomForestClassifier\
    (n_estimators=200, criterion='gini', max_depth=9,
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1,
    random_state=4, verbose=1, warm_start=False, class_weight=None)
    random.fit(X_train_all, y_train_all)
    y_test_all_predict_proba = random.predict_proba(X_test_all)
    counter = 0
    for threshold in thresholds:
        pos_pred = y_test_all_predict_proba[:,1]>threshold
        n_pos_pred[counter] = sum(pos_pred)
        cost_of_all_counselings[counter] = n_pos_pred[counter] * cost_per_counseling
        true_pos = pos_pred & y_test_all.astype(bool)
        n_true_pos[counter] = sum(true_pos)
        savings_of_all_counselings[counter] = n_true_pos[counter] * savings_per_default * effectiveness
        
        counter += 1
    net_savings = savings_of_all_counselings - cost_of_all_counselings
    st.write(''' 
             The average of the most recent months' bill across all accounts is **NT$51,602**. So, using the assumption that this is the opportunity for savings of a prevented default for each account, the net savings after a cost of **NT$7,500** for credit counseling will be **NT$44,102**. This indicates a potential for net savings in the credit counseling program.
             
             
             The issue is that not all accounts will default. For an account that wouldn't default, a counseling session represents a wasted **NT$7,500**. Our analysis needs to balance the costs of counseling with the risk of default.
             
             
             Now, we're in a position to visualize how much money we might help our client save by providing counseling to the appropriate account holders. Let's visualize this.
    
    
    ''')
    plt.plot(thresholds, net_savings)
    plt.xlabel('Threshold')
    plt.ylabel('Net savings (NT$)')
    plt.xticks(np.linspace(0,1,11))
    plt.grid(True)
    st.pyplot()
    st.write('The plot indicates that the choice of threshold is important. While it will be possible to create net savings at many different values of the threshold, it looks like the highest net savings will be generated by setting the threshold somewhere in the range of about **0.2 to 0.25**.')
    max_savings_ix = np.argmax(net_savings)
    st.write('Displaying the threshold that results in the greatest net savings:')
    st.write(thresholds[max_savings_ix])
    st.write('Displaying the greatest possible net savings:')
    st.write(net_savings[max_savings_ix])
    st.write('We see that the greatest net savings occurs at a threshold of **0.2** of predicted probability of default. The amount of net savings realized at this threshold is over **NT$15 million**, for the testing dataset of accounts.')    
    st.write('As the threshold increases, we are raising the bar for how risky a client must be, in order for us to contact them and offer counseling. Increasing the threshold from **0.2 to 0.25** means we would be only contacting riskier clients whose probability is **> 0.25**. This means contacting fewer clients, reducing the up-front cost of the program.')
    cost_of_all_default = sum(y_test_all) * savings_per_default
    st.write('Using the testing set, calculated the cost of all defaults if there were no counseling program')
    st.write(cost_of_all_default)
    st.write('calculated by what percent can the cost of defaults be decreased by the counseling program')
    st.write(net_savings[max_savings_ix] / len(y_test_all))
    st.write('ploted the net savings per account against the cost of counseling per account for each threshold.')
    plt.plot(cost_of_all_counselings / len(y_test_all), net_savings / len(y_test_all))
    plt.xlabel('Cost of Counseling per Account')
    plt.ylabel('Net Savings per Account (NT$)')
    st.pyplot()
    st.write('ploted the fraction of accounts predicted as positive (this is called the "flag rate") at each threshold.')
    plt.plot(thresholds, n_pos_pred / len(y_test_all))
    plt.xlabel('thresholds')
    plt.ylabel('flag rate')
    st.pyplot()
    st.write(' ploted a precision-recall curve for the testing data.')
    precsion, recal, _ = precision_recall_curve(y_test_all, y_test_all_predict_proba[:,1])
    plt.plot(recal, precsion)
    plt.title('precision-recall curve for testing data')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    st.pyplot()
    st.write('Finally, ploted precision and recall separately on the y-axis against threshold on the x-axis.')
    plt.plot(thresholds, np.divide(n_true_pos,n_pos_pred), label="Precision")
    plt.plot(thresholds, n_true_pos/sum(y_test_all), label="Recall")
    plt.xlabel("Thresholds")
    plt.legend()
    st.pyplot()
    st.write('We have now completed modeling activities and also created a financial analysis to indicate to the client how they can use the model. While we have created the essential intellectual contributions that are the data scientist\'s responsibility, it is necessary to agree with the client on the form in which all these contributions will be delivered.')


if st.sidebar.checkbox('About & Contacts'):
    st.subheader(' **About** ')
    st.write('''
             
             Hi , I Mohammed Alfeed Peeranwale developed this webapp as a task of my datascience internship in Technocolabs, in the internship my project contened to exploring the data, build the model then conduct a financial analysis. After completion of all the tasks the final task is to deploy all work into the webapp.
             
             
    **Created on : *18-09-2020***
             
             ''')
    st.subheader(' **Contacts** ')
    st.write(''' 
             
             **Name : *Mohammed Alfeed Peeranwale*** 
     
        
        **E-mail : *alfidpiranwale@gmail.com***
             
             
        **Job-role : *Data Science Intern***
        
        
        
        **Company : *Technocolabs***
        
        
    
    ''')
    st.balloons()

    
         

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    