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
| Column 1 | Column 2 | Column 3 | Column 4 |
|----------|----------|----------|----------|
| Row1-1   | Row1-2   | Row1-3   | Row1-4   |
| Row2-1   | Row2-2   | Row2-3   | Row2-4   |
| Row3-1   | Row3-2   | Row3-3   | Row3-4   |
| Row4-1   | Row4-2   | Row4-3   | Row4-4   |
| Row5-1   | Row5-2   | Row5-3   | Row5-4   |
| Row6-1   | Row6-2   | Row6-3   | Row6-4   |
| Row7-1   | Row7-2   | Row7-3   | Row7-4   |
| Row8-1   | Row8-2   | Row8-3   | Row8-4   |
| Row9-1   | Row9-2   | Row9-3   | Row9-4   |
| Row10-1  | Row10-2  | Row10-3  | Row10-4  |


**Table 2. The AUC of *name2gender***
| Column 1 | Column 2 | Column 3 | Column 4 |
|----------|----------|----------|----------|
| Row1-1   | Row1-2   | Row1-3   | Row1-4   |
| Row2-1   | Row2-2   | Row2-3   | Row2-4   |
| Row3-1   | Row3-2   | Row3-3   | Row3-4   |
| Row4-1   | Row4-2   | Row4-3   | Row4-4   |
| Row5-1   | Row5-2   | Row5-3   | Row5-4   |
| Row6-1   | Row6-2   | Row6-3   | Row6-4   |
| Row7-1   | Row7-2   | Row7-3   | Row7-4   |
| Row8-1   | Row8-2   | Row8-3   | Row8-4   |
| Row9-1   | Row9-2   | Row9-3   | Row9-4   |
| Row10-1  | Row10-2  | Row10-3  | Row10-4  |


