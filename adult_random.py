import pandas as pd
import numpy as np

# Load the dataset

data = pd.read_csv('adult.csv')

data.isna().sum()

head(data)

data.info()

data.describe()


data['education'].value_counts()


data['age'].aggregate(['min','max','mean'])


data.occupation.value_counts()


data.workclass.value_counts()


import matplotlib.pyplot as plt

import plotly.graph_objects as go



# Create subplots
fig = go.Figure()

# Add histogram for income > 50K
fig.add_trace(go.Histogram(x=data[data['income'] == '>50K']['age'], 
                            marker=dict(color='green', opacity=0.7),
                            name='Income more than 50,000'))

# Add histogram for income <= 50K
fig.add_trace(go.Histogram(x=data[data['income'] == '<=50K']['age'], 
                            marker=dict(color='yellow', opacity=0.7),
                            name='Income less than 50,000'))

# Update layout
fig.update_layout(barmode='overlay', 
                  xaxis=dict(title='Age'), 
                  yaxis=dict(title='Frequency'),
                  title='Distribution of Age by Income Level')

# Show plot
fig.show()


import plotly.express as px

data_less_than_50k = data[data['income'] == '<=50K']
data_more_than_50k = data[data['income'] == '>50K']

fig = go.Figure()

fig.add_trace(go.Bar(
    x=data_less_than_50k['marital-status'],
    y=data_less_than_50k['hours-per-week'],
    name='Income <=50K',
    marker=dict(color='blue')  # Color for <=50K bars
))

# Add bar trace for income >50K
fig.add_trace(go.Bar(
    x=data_more_than_50k['marital-status'],
    y=data_more_than_50k['hours-per-week'],
    name='Income >50K',
    marker=dict(color='green')  # Color for >50K bars
))

# Update the layout
fig.update_layout(
    title="Hours per Week by Marital Status and Income Level",
    xaxis=dict(title='Marital Status'),
    yaxis=dict(title='Hours per Week'),
    barmode='group'
)

# Update traces to customize appearance
fig.update_traces(marker_line_width=0)

# Show the figure
fig.show()

fig = go.Figure()

fig.add_trace(go.Bar(
    x=data_less_than_50k['gender'],
    y=data_less_than_50k['hours-per-week'],
    name='Income <=50K',
    marker=dict(color='purple')  # Color for <=50K bars
))

# Add bar trace for income >50K
fig.add_trace(go.Bar(
    x=data_more_than_50k['gender'],
    y=data_more_than_50k['hours-per-week'],
    name='Income >50K',
    marker=dict(color='black')  # Color for >50K bars
))

# Update the layout
fig.update_layout(
    title="Hours per Week by Gender and Income Level",
    xaxis=dict(title='gender'),
    yaxis=dict(title='Hours per Week'),
    barmode='group'
)

# Update traces to customize appearance
fig.update_traces(marker_line_width=0)

# Show the figure
fig.show()

fig = go.Figure()

fig.add_trace(go.Bar(
    x=data_less_than_50k['education'],
    y=data_less_than_50k['hours-per-week'],
    name='Income <=50K',
    marker=dict(color='brown')  # Color for <=50K bars
))

# Add bar trace for income >50K
fig.add_trace(go.Bar(
    x=data_more_than_50k['education'],
    y=data_more_than_50k['hours-per-week'],
    name='Income >50K',
    marker=dict(color='grey')  # Color for >50K bars
))

# Update the layout
fig.update_layout(
    title="Hours per Week by education and Income Level",
    xaxis=dict(title='education'),
    yaxis=dict(title='Hours per Week'),
    barmode='group'
)

# Update traces to customize appearance
fig.update_traces(marker_line_width=0)

# Show the figure
fig.show()

fig = go.Figure()

fig.add_trace(go.Bar(
    x=data_less_than_50k['occupation'],
    y=data_less_than_50k['hours-per-week'],
    name='Income <=50K',
    marker=dict(color='black')  # Color for <=50K bars
))

# Add bar trace for income >50K
fig.add_trace(go.Bar(
    x=data_more_than_50k['occupation'],
    y=data_more_than_50k['hours-per-week'],
    name='Income >50K',
    marker=dict(color='gold')  # Color for >50K bars
))

# Update the layout
fig.update_layout(
    title="Hours per Week by occupation and Income Level",
    xaxis=dict(title='occupation'),
    yaxis=dict(title='Hours per Week'),
    barmode='group'
)

# Update traces to customize appearance
fig.update_traces(marker_line_width=0)

# Show the figure
fig.show()

fig = go.Figure()

fig.add_trace(go.Bar(
    x=data_less_than_50k['relationship'],
    y=data_less_than_50k['hours-per-week'],
    name='Income <=50K',
    marker=dict(color='black')  # Color for <=50K bars
))

# Add bar trace for income >50K
fig.add_trace(go.Bar(
    x=data_more_than_50k['relationship'],
    y=data_more_than_50k['hours-per-week'],
    name='Income >50K',
    marker=dict(color='gold')  # Color for >50K bars
))

# Update the layout
fig.update_layout(
    title="Hours per Week by relationship and Income Level",
    xaxis=dict(title='relationship'),
    yaxis=dict(title='Hours per Week'),
    barmode='group'
)

# Update traces to customize appearance
fig.update_traces(marker_line_width=0)

# Show the figure
fig.show()

data.columns

data.head(5)

data = data.drop('education', axis=1) 

data = pd.concat([data.drop('occupation', axis=1), pd.get_dummies(data.occupation).add_prefix('occupation_')], axis=1)
data = pd.concat([data.drop('workclass', axis=1), pd.get_dummies(data.workclass).add_prefix('workclass_')], axis=1)
data = pd.concat([data.drop('marital-status', axis=1), pd.get_dummies(data['marital-status']).add_prefix('marital-status_')], axis=1)
data = pd.concat([data.drop('relationship', axis=1), pd.get_dummies(data.relationship).add_prefix('relationship_')], axis=1)
data = pd.concat([data.drop('race', axis=1), pd.get_dummies(data.race).add_prefix('race_')], axis=1)
data = pd.concat([data.drop('native-country', axis=1), pd.get_dummies(data['native-country']).add_prefix('native-country_')], axis=1)
data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

data.head(2)

import seaborn as sns

plt.figure(figsize=(18,15))
sns.heatmap(data.corr(),annot=False,cmap='coolwarm')

sort_correlations = (data.corr()['income'].abs()).sort_values()
num_drop = int(0.8 * len(data.columns)) # selects the first 80% of the sorted correlations
droped_columns = sort_correlations.iloc[:num_drop].index
data_droped = data.drop(droped_columns , axis =1)

plt.figure(figsize = (15,12))
sns.heatmap(data_droped.corr() , cmap='coolwarm',annot = True)

data.drop('fnlwgt', axis = 1, inplace=True)
data = data.replace('?', np.nan)

data.head(5)

from sklearn.model_selection import train_test_split

train , test = train_test_split(data,test_size=0.2)

print(f'The shape of the training data is: {train.shape}')
print(f'The shape of the test data is: {test.shape}')

x_train = data.drop('income', axis = 1)
y_train = data['income']

x_test = data.drop('income', axis = 1)
y_test = data['income']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

random_model = RandomForestClassifier()
random_model.fit(x_train , y_train)

random_model.score(x_test,y_test)

random_model.feature_names_in_

param_estimators = {
    'n_estimators':[50,100,150],
    'max_depth':[5,10,20,None],
    'min_samples_split':[2,4],
    'max_features':['sqrt','log2']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_estimators,verbose=10)
grid_search

grid_search.fit(x_train,y_train)

grid_search.best_estimator_

best_forest=grid_search.best_estimator_
best_forest.score(x_test,y_test)



from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Extract one of the decision trees from the random forest
tree = best_forest.estimators_[0]

# Plot the simplified decision tree with a maximum depth of 3
plt.figure(figsize=(40,30))
plot_tree(tree, feature_names=x_train.columns, class_names=['<=50K', '>50K'], filled=True, rounded=True, max_depth=3)
plt.show()

