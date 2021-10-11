import streamlit as st
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.model_selection
from joblib import dump, load
from sklearn import svm
import pickle
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#import plotly.express as px
#import plotly.graph_objects as go 

st.title('__Genetic Variant Classifications - Predicting whether a variant will have conflicting clinical classifications.__')
st.header('__This app is created by Gulsah Altun, Springboard ML Engineering Thermo Fisher Student, 2021__')

st.sidebar.header('Welcome to my first app that I deployed on streamlit!')

st.sidebar.header('Select one of the pre-trained models to run:')

option_1 = st.sidebar.button('XGboost')
option_2 = st.sidebar.button('Random Forest')
option_3 = st.sidebar.button('SVM')
option_4 = st.sidebar.button('Logistic regression')

uploaded_file = st.sidebar.file_uploader(label = "Upload your variants file here:", type =['csv'])

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')



global df1


st.header("<---Please upload your variants file on the left side and select an ML model in order to run the app")


  
if uploaded_file is not None: 
    try: 
        df1 = pd.read_csv(uploaded_file)
    except Exception as e: 
        print(e) 
    print(uploaded_file)
    print("upload completed")


    try:
        st.markdown("""
        #Input data:
        """)
        st.write(df1)
    except Exception as e:
        print(e)


    pd.DataFrame([[i, len(df1[i].unique())] for i in df1.columns], columns=['Columns', 'Unique']).set_index('Columns')

    X_test = df1.select_dtypes(exclude=object) 




    if option_1:
        clf_loaded = load('BestModelXGBOOST3.joblib') 

    if option_2:
        clf_loaded = load('BestModelRANDOMFOREST.joblib')
        
    if option_3:
        clf_loaded = load('ModelSVM.joblib')
    
    if option_4:
        clf_loaded = load('Model_logreg.joblib')
    
    if option_1 or option_2 or option_3 or option_4:
        st.markdown("""
        #Results:
        """)
        
    
        st.markdown("""
        #Probability of each variant being a conflicting variant:
        """)
        st.write(clf_loaded.predict_proba(X_test)[:,1] )
        df4=pd.DataFrame(clf_loaded.predict_proba(X_test)[:,1])

        st.markdown("""
        #Probability of each variant being a non-conflicting variant:
        """)
    
        st.write(clf_loaded.predict_proba(X_test)[:,0]) 
        df5=pd.DataFrame(clf_loaded.predict_proba(X_test)[:,0])

    # Save results as a .csv file
        file_name = "results_file_probability_of_being_nonconflicting_variant.csv"
        file_path = f"./{file_name}"
        #
        df5.to_csv(file_path)   
        # Create Download Button
        file_bytes = open(file_path, 'rb')
        st.download_button(label='Click to download results file listing the probabilities of each variant being a conflicting variant',
                       data=file_bytes, 
                       file_name=file_name,
                       key='download_df')
        file_bytes.close()
        
        
        
        
 
