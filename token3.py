# token3.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datasets import Dataset
from transformers import AutoTokenizer

# Load your data from a CSV file into a pandas DataFrame
encoded_df = pd.read_csv('APY.csv')

encoded_df.columns = encoded_df.columns.str.strip()

# Print column names
print(encoded_df.columns)

# Reset the index
encoded_df = encoded_df.reset_index()
encoded_df['question'] = encoded_df['question'].astype(str).fillna('')
# Create the 'question' column before one-hot encoding
encoded_df['question'] = 'What is the yield for ' + encoded_df['Crop'] + ' in ' + encoded_df['Crop_Year'].astype(str) + ' during ' + encoded_df['Season'] + '?'
print(encoded_df['question'].head())


# One-hot encode categorical variables

# One-hot encode the categorical features
# encoder = OneHotEncoder(sparse=False)
# encoded_features = encoder.fit_transform(df[['State', 'District', 'Crop', 'Season']])

# Create a DataFrame with the encoded features
# encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['State', 'District', 'Crop', 'Season']))

# Combine encoded features with numerical features
# df = pd.concat([df[['Crop_Year', 'Area', 'Production', 'Yield', 'question']], encoded_df], axis=1)

# # Handle missing values (if any)
# df = df.dropna()

# # Split data into features (X) and target (y)
# X = df.drop('Yield', axis=1)  # assuming we're predicting 'Yield'
# y = df['Yield']

# # Normalize features
# scaler = StandardScaler()
# X_numeric = df.drop(['Yield', 'question'], axis=1)  # exclude 'Yield' and 'question' columns
# X_numeric = scaler.fit_transform(X_numeric)

# # Combine normalized features with 'question' column
# X_numeric_df = pd.DataFrame(X_numeric, columns=df.drop(['Yield', 'question'], axis=1).columns)
# X_numeric_df.reset_index(drop=True, inplace=True)
# df['question'].reset_index(drop=True, inplace=True)

# X = pd.concat([X_numeric_df, df['question']], axis=1)

# # Split data into features (X) and target (y)
# X = df.drop(['Yield', 'question'], axis=1)  # exclude 'Yield' and 'question' columns
# y = df['Yield']

# # Normalize features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Combine normalized features with 'question' column
# X_df = pd.DataFrame(X, columns=df.drop(['Yield', 'question'], axis=1).columns)
# X_df.reset_index(drop=True, inplace=True)
# df['question'].reset_index(drop=True, inplace=True)

# X_final = pd.concat([X_df, df['question']], axis=1)

# Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)  # use X_df here, not X_final

# print(encoded_df)

encoded_df.to_csv('processed_data.csv', index=False)
# ... your existing code ...

print(encoded_df)

encoded_df.to_csv('processed_data.csv', index=False)

# After saving the processed data to a CSV file, add the following code:

# Load your processed data
encoded_df = pd.read_csv('processed_data.csv')

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b')

# Tokenize the 'question' column
tokenized_data = tokenizer(encoded_df['question'].tolist(), truncation=True, max_length=512)

# Convert the tokenized data to a Hugging Face Dataset
dataset = Dataset.from_dict(tokenized_data)

# Save the tokenized dataset to disk
dataset.save_to_disk("./tokenized_datasets")
