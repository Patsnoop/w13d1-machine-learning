import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
file_path = Path("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = pd.read_csv(file_path)

# Explore the dataset
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Check the shape of the data
print("\nShape of the dataset:")
print(df.shape)

# Identify the target and feature variables
# The target variable is usually identified from the context of the problem
target_variable = "Attrition"

# Feature variables are all columns except the target variable
feature_variables = df.columns[df.columns != target_variable]

print("\nTarget variable:")
print(target_variable)

print("\nFeature variables:")
print(feature_variables)

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Dropping rows with missing values (if any)
df = df.dropna()

# Identify categorical features
categorical_features = df.select_dtypes(include=['object']).columns
print("\nCategorical features:")
print(categorical_features)

# One-hot encoding for categorical features
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Define target and feature variables
target_variable = "Attrition_Yes"  # After encoding, 'Attrition' becomes 'Attrition_Yes' due to get_dummies
X = df_encoded.drop(columns=[target_variable])
y = df_encoded[target_variable]

# Example: Creating a new feature for total working years minus current role tenure
if 'TotalWorkingYears' in df_encoded.columns and 'YearsInCurrentRole' in df_encoded.columns:
    df_encoded['RemainingWorkingYears'] = df_encoded['TotalWorkingYears'] - df_encoded['YearsInCurrentRole']

# Scale numerical features
# Identify numerical features
numerical_features = df_encoded.select_dtypes(include=['int64', 'float64']).columns

# Split the dataset
# Define target and feature variables
target_variable = "Attrition_Yes"  # After encoding
X = df_encoded.drop(columns=[target_variable])
y = df_encoded[target_variable]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the splits
print("\nShape of training and testing sets:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Evaluate the Logistic Regression model
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_log_reg):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_log_reg):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_log_reg):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))

# Step 4: Train and evaluate Decision Tree
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print("\nDecision Tree:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_tree):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_tree):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_tree):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

# Step 5: Train and evaluate Random Forest
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
y_pred_forest = forest_clf.predict(X_test)
print("\nRandom Forest:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_forest):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_forest):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_forest):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_forest):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_forest))
