

# MICROSOFT SPRINGBOARD CAPSTONE PROJECT - Genetic Variant Classification

## Background

[ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) is a public archive of interpretations of clinically relevant variants where many researchers can share their variant interpretations. 
There has been over a million submissions so far from many researchers and clinical labs. 
Sometimes, some of these submissions lead to some variants to have conflicting classifications such as when one submission says a variant is 
benign and another says it is pathogenic. Here is some statistics about the [submitted records](https://www.ncbi.nlm.nih.gov/clinvar/submitters/).



## Problem Description

This problem of conflicting variants had been published and explained on [kaggle](https://www.kaggle.com/kevinarvai/clinvar-conflicting) as below:
        
ClinVar Genetic Variant Classification: Prediction of whether a variant will have conflicting clinical 
classifications. CLINVAR is a public resource containing annotations about human genetic variants. 
These variants are (usually manually) classified by clinical laboratories on a categorical spectrum ranging
from benign, likely benign, uncertain significance, likely pathogenic, and pathogenic. 
Variants that have conflicting classifications (from laboratory to laboratory) can cause confusion when
clinicians or researchers try to interpret whether the variant has an impact on the disease of a given patient. 


![alt text](https://github.com/gulsahaltun/MLCapstoneProject/blob/master/clinvar-class-fig.png)

(image courtesy of [kaggle](https://www.kaggle.com/kevinarvai/clinvar-conflicting))


## Getting Started


## Objective

The objective of this capstone was to predict whether a CLINVAR variant will have conflicting classifications.  A binary classification problem, where each record in the dataset is a genetic variant with many biological 
and clinical features. 


## Data set:

* The dataset used in this project has been downloaded from the kaggle project website which is a CSV file (clinvar_conflicting.csv) that contains the genomic locations and many other features of each clinvar variant.
In this training dataset each clinvar variant has been labeled as a conflicting or a nonconflicting variant.  There were 48747 nonconflicting variants ​and  16429 conflicting variants which is a good dataset to work with. 

 
* This CSV file was prepared from the raw ClinVar vcf file hosted on the CLINVAR ftp site for hg19 (GRCh37) human reference genome: ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/ 

* For more details on the dataset please visit the [kaggle](https://www.kaggle.com/kevinarvai/clinvar-conflicting) website which included more information and statistics. 




 ## Feature engineering 
 
The original dataset contained 46 features. However, after analyzing the dataset and all of its features and utilizing various statistical tools and domain knowledge, the number of features was reduced to 8. 
  
The final training dataset contained 65176 unique ClinVar variants when grouped by by genomic position on each chromosome along with their reference and alternative alleles and 8 features.

These features are the chromosome CHROM, genomic position () POS	various frequency scores such as AF_ ESP	AF_ EXAC	AF_ TGP	CADD_ PHRED	CADD_RAW and strand and class.

These allele frequencies of a variant that can be found in various websites. Allele frequencies are retrieved from GO-ESP,  ExAC, TGP and CADD websites. 
 
Here is an overview of the first 5 rows: 


 

CHROM	POS	AF_ ESP	AF_ EXAC	AF_ TGP	CLASS	STRAND	CADD_ PHRED	CADD_RAW

0	1	1168180	0.0771	0.10020	0.1066	0	1.0	1.053	-0.208682

1	1	1470752	0.0000	0.00000	0.0000	0	-1.0	31.000	6.517838

2	1	1737942	0.0000	0.00001	0.0000	1	-1.0	28.100	6.061752

3	1	2160305	0.0000	0.00000	0.0000	0	1.0	22.500	3.114491

4	1	2160305	0.0000	0.00000	0.0000	0	1.0	24.700	4.76622

....
 
 
 
 

### Dependencies

* These are the prerequisites and libraries needed when running the program.
* streamlit==0.88.0
* scipy==1.7.1
* xgboost==1.4.2
* pandas==1.1.3
* seaborn==0.11.0
* numpy==1.19.2
* matplotlib==3.3.2
* joblib==1.0.1
* scikit_learn==1.0

 
 


## Algorithms I tried 


I used 4 different algorithms and created models, these were: 

* XGBoost
* Random forest
* SVM
* Logistic regression

  
While using all of the models above, I tried different parameters to tune the models and saved the best estimator as the final model. 



Here are some of the examples of parameters I used where I utilized a random grid search:



* Parameter grid for XGBoost

    'min_ child_ weight': [1, 5, 10],
    
    'gamma': [0.5, 1, 1.5, 2, 5],
    
    'subsample': [0.6, 0.8, 1.0],
    
    'colsample_ bytree': [0.6, 0.8, 1.0],
    
    'max_ depth': [3, 4, 5]
    
 



* Parameter grid for RANDOM FOREST:
 
    'n_ estimators': [10, 25, 100],
    
    'max_ features': ['auto', 'sqrt', 'log2'],
    
    'max_ depth' : [1,2,3,4,5,6,7,8,9,10],
    
    'criterion' :['gini', 'entropy']
    
 







I used joblib library to export the models which I later used
to upload while deployment. So, a total of 4 joblib files were exported. 


### Deployment

* I had uploaded and deployed all of these 4 models using streamlit and wrote code where I created an app. In this app, a user can input their input test file with variants and get prediction results 
on if a variant is a conflicting variant or not. I had hosted the streamlit app that I had build in the following link: 

STREAMLIT LINK:
https://share.streamlit.io/gulsahaltun/mlcapstoneproject/main.py


 

 
### How to test the program on streamlit site

* I had provided a test file that contains 10 test clinvar variants of unknown status. The test program, the test file can be downloaded from the repo and uploaded to the streamlit app below. 
Once the test data is uploaded, then select the model that you want to use and the results will be provided both on the website as well as in a downloadable file format. The predictions in the 
downloaded file format will be in the same order as the order of variants in the input file. 
 
 
 


## RESULTS SUMMARY 

During this project, I had done a lot of Exploratory Data Analysis (EDA) and ended up training many models with 
different parameters as I was tuning the models. I used 4  models with their final
parameters and best models exported and used by the streamlit app. The best results were almost always achived by XGBoost with the highest ROC being 0.76. 
Random forest was 0.74 whereas Logistic regression and SVM didn't perform well and results were random. 







## QUICK LINKS: 

## Capstone project deployed on Streamlit:
[You can reach the Streamlit application here.](https://share.streamlit.io/gulsahaltun/mlcapstoneproject/main.py)


## Capstone project write-up and jupyter notebook:
[You can access the jupyter notebook here.](https://github.com/gulsahaltun/MLCapstoneProject/blob/master/CapstoneNotebook.ipynb)





## References


Here are some of the recent articles and publications about the conflicting variants in Clinvar:

* https://www.precisiononcologynews.com/cancer/brca2-variants-unknown-significance-reclassified-through-functional-data-additions#.YUOR5Z5KiVY

* https://www.nature.com/articles/s41598-019-57335-5

* https://www.healthcareitnews.com/news/teens-precision-medicine-analytics-website-highlights-value-data-democratization

* http://variantexplorer.org/

* https://www.ncbi.nlm.nih.gov/clinvar/docs/faq/

* https://f1000researchdata.s3.amazonaws.com/manuscripts/15752/5ca368c4-e377-47f8-9cc5-28053692872e_14470_-_robert_butler.pdf?doi=10.12688/f1000research.14470.1&numberOfBrowsableCollections=26&numberOfBrowsableInstitutionalCollections=4&numberOfBrowsableGateways=29

* https://www.genomeweb.com/molecular-diagnostics/clingen-implementing-strategies-resolve-variant-classification-conflicts?utm_source=TrendMD&utm_medium=TrendMD&utm_campaign=0&trendmd-shared=0#.YUOT455KiVY

* https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-019-0688-9

* https://www.sciencedirect.com/science/article/pii/S0002929718300879

* https://www.genomeweb.com/molecular-diagnostics/more-million-records-clinvar-value-grows-variant-classification-resource#.YUOTkJ5KiVY

* https://www.genomeweb.com/molecular-diagnostics/clingen-implementing-strategies-resolve-variant-classification-conflicts?utm_source=TrendMD&utm_medium=TrendMD&utm_campaign=0&trendmd-shared=0#.YUOT455KiVY

* https://www.genomeweb.com/clinical-genomics/tackling-vus-challenge-are-public-databases-solution-or-liability-labs?utm_source=TrendMD&utm_medium=TrendMD&utm_campaign=0&trendmd-shared=0#.YUOT7J5KiVY

* https://ascopubs.org/doi/10.1200/JCO.2016.68.4316 https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-019-0688-9





