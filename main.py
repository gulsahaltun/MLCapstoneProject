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


st.title('__Genetic Variant Classifications - Predicting whether a variant will have conflicting clinical classifications.__')
st.header('__This app is created by Gulsah Altun, Springboard ML Engineering Thermo Fisher Student, 2021__')

st.sidebar.header('Welcome to my first app that I deployed on streamlit!')

st.sidebar.header('Select one of the pre-trained models to run:')

option_11 = st.sidebar.checkbox('XGboost')
option_22 = st.sidebar.checkbox('Random Forest')
option_33 = st.sidebar.checkbox('SVM')
option_44 = st.sidebar.checkbox('Logistic regression')

uploaded_file = st.sidebar.file_uploader(label = "Upload your variants file here:", type =['csv'])

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


global df1 
global df2


st.header("<---Please upload your variants file on the left side and select an ML model in order to run the app")


  
if uploaded_file is not None: 
    try: 
        df2 = pd.read_csv(uploaded_file)
    except Exception as e: 
        print(e) 
    print(uploaded_file)
    print("upload completed")
    st.write(df2.head())

    try:
        st.write(df2)
    except Exception as e:
        print(e)
        #st.sidebar.write("please upload a file containing variants")
    df2.CLASS.value_counts()

    pd.DataFrame([[i, len(df2[i].unique())] for i in df2.columns], columns=['Columns', 'Unique']).set_index('Columns')


    st.markdown("""
    #Numerical data:
    """)
    numerics = df2.select_dtypes(exclude=object) 

    st.write(numerics.head())


    st.markdown("""
    #Numerical data:
    """)
    cleaned = numerics.drop(["SSR", "DISTANCE","MOTIF_POS","MOTIF_SCORE_CHANGE","BLOSUM62"],axis=1)
    final = cleaned.drop(["CLNDISDBINCL","INTRON","HIGH_INF_POS","cDNA_position","CDS_position","CLNDNINCL","CLNSIGINCL","MOTIF_NAME","CHROM","Protein_position"],axis=1)
    
    st.write(final.head())
    
    X_test = final.drop(['CLASS'],axis=1).values   # independant features
    y_test = final['CLASS'].values


    if option_11:
        st.markdown("""
        #auc score for xgboost:
        """)
    
        clf_XGBloaded = load('BestModel.joblib') 
        st.write(roc_auc_score(y_test, clf_XGBloaded.predict_proba(X_test)[:,1]))
    
        st.markdown("""
        #Probability of each variant being a conflicting variant:
        """)
        st.write(clf_XGBloaded.predict_proba(X_test)[:,1] )
        df4=pd.DataFrame(clf_XGBloaded.predict_proba(X_test)[:,1])

        st.markdown("""
        #Probability of each variant being a non-conflicting variant:
        """)
    
        st.write(clf_XGBloaded.predict_proba(X_test)[:,0]) 
        df5=pd.DataFrame(clf_XGBloaded.predict_proba(X_test)[:,0])

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
 
