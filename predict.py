# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data from the uploaded Excel file
df = pd.read_excel('/content/Dataset .xlsx')  # Adjust path if needed

# Select features for prediction
features = ['Price range', 'Has Table booking', 'Has Online delivery', 'Cuisines', 'Average Cost for two', 'City']
target = 'Aggregate rating'

# Prepare the data
X = df[features].copy()  # Avoid modifying the original DataFrame
y = df[target]

# Convert 'Yes'/'No' to binary
for col in ['Has Table booking', 'Has Online delivery']:
    X.loc[:, col] = X[col].map({'Yes': 1, 'No': 0})

# Define preprocessing for numeric and categorical data
numeric_features = ['Price range', 'Average Cost for two']
categorical_features = ['Cuisines', 'City']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessing and Random Forest Regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Example prediction for a new restaurant
new_restaurant = pd.DataFrame({
    'Price range': [3],
    'Has Table booking': [1],  # 1 for Yes, 0 for No
    'Has Online delivery': [0],  # 1 for Yes, 0 for No
    'Cuisines': ['Italian, Mediterranean'],
    'Average Cost for two': [2000],
    'City': ['New York']
})

# Ensure new data goes through the same preprocessing pipeline
new_restaurant_transformed = model.named_steps['preprocessor'].transform(new_restaurant)
predicted_rating = model.named_steps['regressor'].predict(new_restaurant_transformed)

print(f"Predicted Aggregate Rating for the new restaurant: {predicted_rating[0]:.2f}")
