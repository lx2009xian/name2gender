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
**Table 1. The AUC of *name2CKJV***
| Training times | Accuracy | Precision | Recall | F1 |
|-----|-----|-----|-----|-----|
| 1 | 行1 | 行1 | 行1 | 行1 |
| 2 | 行2 | 行2 | 行2 | 行2 |
| 3 | 行3 | 行3 | 行3 | 行3 |
| 4 | 行4 | 行4 | 行4 | 行4 |
| 5 | 行5 | 行5 | 行5 | 行5 |
| 6 | 行6 | 行6 | 行6 | 行6 |
| 7 | 行7 | 行7 | 行7 | 行7 |
| 8 | 行8 | 行8 | 行8 | 行8 |
| 9 | 行9 | 行9 | 行9 | 行9 |
| 10| 行10| 行10| 行10| 行10|


**Table 2. The AUC of *name2gender***
| Training times | Accuracy | Precision | Recall | F1 |
|-----|-----|-----|-----|-----|
| 1 | 行1 | 行1 | 行1 | 行1 |
| 2 | 行2 | 行2 | 行2 | 行2 |
| 3 | 行3 | 行3 | 行3 | 行3 |




