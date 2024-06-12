# name2gender
This a python code for gender prediction based on full name
# Introduction
We utilize a pre-trained BERT model to predict an author's gender based on their full name. While Gender-Guesser is an open Python module that infers gender using a pre-existing author database, it often misidentifies authors not included in that database. Our approach significantly improves the accuracy of gender predictions by addressing these misidentification issues.
# Data and Methodology
It is generally challenging to infer the gender of individuals from China, Korea (both Southern and Northern), Japan, and Vietnam based solely on their names. To address this, our methodology involves two key steps:
1.	**Exclusion of CKJV Names:** We use a database containing 1,112,902 authors' full names from 173 countries to train a model (*name2CKJV*) that is used to identify authors from the four aforementioned countries.
2.	**Gender Prediction:** We utilize a dataset comprising gender information for 23,157,265 scholars to train a gender prediction model. This model can category gender into three groups: “female”, “male”, or “androgynous” (*name2gender*).
By refining our approach to exclude names from countries where gender is less discernible from names alone, our model enhances the precision and applicability of gender prediction.

# Result
**Table 1. The performance of *name2CKJV***
| Training times | Accuracy | Precision | Recall | F1 |
|-----|-----|-----|-----|-----|
| 1 | 0.9930 | 0.9174 | 0.9251 | 0.9213 |
| 2 | 0.9928 | 0.9151 | 0.9240 | 0.9195 |
| 3 | 0.9928 | 0.9151 | 0.9246 | 0.9198 |
| 4 | 0.9928 | 0.9144 | 0.9257 | 0.9200 |
| 5 | 0.9927 | 0.9132 | 0.9245 | 0.9188 |



**Table 2. The performance of *name2gender***
| Training times | Accuracy | Precision | Recall | F1 |
|-----|-----|-----|-----|-----|
| 1 | 0.9340 | 0.7793 | 0.7114| 0.7345 |
| 2 | 0.9339 | 0.7781 | 0.7118 | 0.7346 |
| 3 | 0.9339 | 0.7774 | 0.7129 | 0.7353 |




