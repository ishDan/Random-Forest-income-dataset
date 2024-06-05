# Adult Income Dataset Analysis

## Problem Statement
The objective of this project is to analyze the Adult Income dataset to understand the factors influencing income levels and build a predictive model to classify individuals into income categories (<=50K or >50K).

### Summary

1. **Data Exploration:** The dataset contains information about individuals including demographic features such as age, education, occupation, and marital status, as well as financial attributes like capital gain and loss. Initial exploration involved data visualization and summary statistics to understand the distribution and relationships between variables.

2. **Data Preprocessing:** Categorical variables were one-hot encoded, missing values were handled, and unnecessary columns were dropped. Additionally, the dataset was split into training and testing sets for model evaluation.

3. **Model Building:** Initially, a random forest classifier was trained with default parameters. Hyperparameter tuning was performed using GridSearchCV to find the best set of parameters for improved model performance.

4. **Model Evaluation:** Model performance was evaluated using accuracy score on the test set before and after hyperparameter tuning.

### Most Influential Features
The most influential features identified by the random forest classifier include:
- **Age:** Older individuals tend to have higher incomes.
- **Education Level:** Higher education levels are associated with higher incomes.
- **Marital Status:** Married individuals, particularly those in a civil union, tend to have higher incomes.
- **Occupation:** Certain occupations such as executives and professionals are associated with higher incomes.
- **Capital Gain:** Individuals with higher capital gains typically have higher incomes.

### Conclusion
The analysis of the Adult Income dataset revealed several insights into the factors affecting income levels. The random forest classifier, after hyperparameter tuning, achieved a high accuracy score on the test set, indicating its effectiveness in predicting income categories. Further analysis and refinement of the model could lead to even better performance.

### About the dataset

**Description:**

- **age:** continuous.
- **workclass:** Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- **fnlwgt:** continuous.
- **education:** Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- **education-num:** continuous.
- **marital-status:** Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- **occupation:** Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- **relationship:** Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- **race:** White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- **sex:** Female, Male.
- **capital-gain:** continuous.
- **capital-loss:** continuous.
- **hours-per-week:** continuous.
- **native-country:** United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
- **class:** >50K, <=50K

Kaggle link: [Adult Income Dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset/data)
