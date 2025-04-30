import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

test_ids = test_data['Id']
train_data.dropna(subset=['SalePrice'], inplace=True)

X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']

X_test_final = test_data.copy()

num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

num_transformer = SimpleImputer(strategy='mean')
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

model = RandomForestRegressor(n_estimators=100, random_state=0)

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', model)
])

pipeline.fit(X, y)

predictions = pipeline.predict(X_test_final)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission file created!")
