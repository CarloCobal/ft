import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load your data from a CSV file into a pandas DataFrame
df = pd.read_csv('APY.csv')

# Strip extra spaces from column names
df.columns = df.columns.str.strip()

# Now the rest of your code should work

# Print column names
print(df.columns)

# Reset the index
df = df.reset_index()

# Create the 'question' column before one-hot encoding
df['question'] = 'What is the yield for ' + df['Crop'] + ' in ' + df['Crop_Year'].astype(str) + ' during ' + df['Season'] + '?'

# One-hot encode categorical variables

# One-hot encode the categorical features
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(df[['State', 'District', 'Crop', 'Season']])

# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['State', 'District', 'Crop', 'Season']))

# Combine encoded features with numerical features
df = pd.concat([df[['Crop_Year', 'Area', 'Production', 'Yield', 'question']], encoded_df], axis=1)

# Handle missing values (if any)
df = df.dropna()

# Split data into features (X) and target (y)
X = df.drop('Yield', axis=1)  # assuming we're predicting 'Yield'
y = df['Yield']

# Normalize features
scaler = StandardScaler()
X_numeric = df.drop(['Yield', 'question'], axis=1)  # exclude 'Yield' and 'question' columns
X_numeric = scaler.fit_transform(X_numeric)

# Combine normalized features with 'question' column
X_numeric_df = pd.DataFrame(X_numeric, columns=df.drop(['Yield', 'question'], axis=1).columns)
X_numeric_df.reset_index(drop=True, inplace=True)
df['question'].reset_index(drop=True, inplace=True)

X = pd.concat([X_numeric_df, df['question']], axis=1)

# Split data into features (X) and target (y)
X = df.drop(['Yield', 'question'], axis=1)  # exclude 'Yield' and 'question' columns
y = df['Yield']

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Combine normalized features with 'question' column
X_df = pd.DataFrame(X, columns=df.drop(['Yield', 'question'], axis=1).columns)
X_df.reset_index(drop=True, inplace=True)
df['question'].reset_index(drop=True, inplace=True)

X_final = pd.concat([X_df, df['question']], axis=1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)  # use X_df here, not X_final

# Initialize the model
model = RandomForestRegressor()

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Now 'predictions' is an array containing the model's yield predictions for the test set
