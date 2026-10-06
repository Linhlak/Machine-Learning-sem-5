import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

# ==========================================================
# 1. Load the Mushroom Edibility Dataset
# ==========================================================
print("--- Step 1: Loading Dataset ---")
mushroom_df = pd.read_csv('/11_mushroom_edibility.csv')
mushroom_df.columns = mushroom_df.columns.str.strip()
print(f"Mushroom dataset shape: {mushroom_df.shape}")
display(mushroom_df.head())

# ==========================================================
# 2. Preprocessing
# ==========================================================
print("\n--- Step 2: Preprocessing and Encoding ---")
# Convert categorical string labels into numerical values
label_encoders = {}
for col in mushroom_df.columns:
    le = LabelEncoder()
    mushroom_df[col] = le.fit_transform(mushroom_df[col].astype(str))
    label_encoders[col] = le

# Identify target column (assuming 'class' or similar represents edibility, e.g., edible/poisonous)
target_candidates = [col for col in mushroom_df.columns if 'class' in col.lower() or 'edib' in col.lower()]
target_col = target_candidates[0] if target_candidates else mushroom_df.columns[0]

X_nb = mushroom_df.drop(columns=[target_col])
y_nb = mushroom_df[target_col]

# Split into training and testing sets
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
    X_nb, y_nb, test_size=0.2, random_state=42
)
print(f"Features shape: {X_nb.shape}, Target column: '{target_col}'")

# ==========================================================
# 3. Gaussian Naive Bayes (Continuous/Normal Distribution assumption)
# ==========================================================
print("\n--- Step 3: Gaussian Naive Bayes ---")
gnb = GaussianNB()
gnb.fit(X_train_nb, y_train_nb)
y_pred_gnb = gnb.predict(X_test_nb)
print(f"Gaussian NB Accuracy: {accuracy_score(y_test_nb, y_pred_gnb):.2%}")

# ==========================================================
# 4. Multinomial Naive Bayes (Discrete count/frequency assumption)
# ==========================================================
print("\n--- Step 4: Multinomial Naive Bayes ---")
mnb = MultinomialNB()
mnb.fit(X_train_nb, y_train_nb)
y_pred_mnb = mnb.predict(X_test_nb)
print(f"Multinomial NB Accuracy: {accuracy_score(y_test_nb, y_pred_mnb):.2%}")

# ==========================================================
# 5. Bernoulli Naive Bayes (Binary/Boolean feature assumption)
# ==========================================================
print("\n--- Step 5: Bernoulli Naive Bayes ---")
bnb = BernoulliNB(binarize=0.5)  # Thresholding features to binarize them
bnb.fit(X_train_nb, y_train_nb)
y_pred_bnb = bnb.predict(X_test_nb)
print(f"Bernoulli NB Accuracy: {accuracy_score(y_test_nb, y_pred_bnb):.2%}")

# ==========================================================
# 6. Comparative Classification Reports
# ==========================================================
print("\n--- Step 6: Detailed Classification Report (Gaussian NB) ---")
print(classification_report(y_test_nb, y_pred_gnb))
