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
| Syntax      | Description | Test Text     | Syntax      | Description | Test Text   | Description | Test Text     | Syntax      | Description | Test Text|  
| :---        |    :----:   |          ---: | :---        |    :----:   |          ---: |:---        |    :----:   |          ---: |          ---: |


| Accuracy      | Title       | Here's this   |Accuracy      | Title       | Here's this   | Accuracy      | Title       | Here's this   | Here's this   |


| Precision   | Text        | And more      | Precision   | Text        | And more      | Precision   | Text        | And more      | And more      |
| Recall      | Text        | And more      | Recall      | Text        | And more      | Recall      | Text        | And more      | Recall      | Text        | And more      |
| F1   | Text        | And more      | Text        | And more      | Text        | And more      | Text        | And more      | Text        | And more     |


**Table 2. The AUC of *name2gender***
| Syntax      | Description | Test Text     |
| :---        |    :----:   |          ---: |
| Accuracy      | Title       | Here's this   |
| Precision   | Text        | And more      |
| Recall      | Text        | And more      |
| F1   | Text        | And more      |
