import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

# ==========================================================
# 1. Load and Explore the Dataset
# ==========================================================
print("--- Step 1: Loading Dataset ---")
df = pd.read_csv('/content/07_loan_approval.csv')
df.columns = df.columns.str.strip()
print(f"Dataset loaded with shape: {df.shape}\n")

# ==========================================================
# 2. Preprocess Data and Split into Train/Test Sets
# ==========================================================
print("--- Step 2: Preprocessing and Splitting ---")
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

target_col = 'loan_status' if 'loan_status' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}\n")

# ==========================================================
# 3. Train the Decision Tree Classifier
# ==========================================================
print("--- Step 3: Training Decision Tree ---")
# max_depth is set to 3 to keep the tree easily interpretable
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)
print("Model trained successfully.\n")

# ==========================================================
# 4. Make Predictions and Evaluate
# ==========================================================
print("--- Step 4: Model Evaluation ---")
y_pred = clf.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ==========================================================
# 5. Visualize the Decision Tree Structure
# ==========================================================
print("--- Step 5: Visualizing Tree Structure ---")
fig, ax = plt.subplots(figsize=(16, 10))
plot_tree(clf,
          feature_names=X.columns.tolist(),
          class_names=['Rejected', 'Approved'] if len(np.unique(y)) == 2 else [str(c) for c in clf.classes_],
          filled=True,
          rounded=True,
          fontsize=10,
          ax=ax)
plt.title("Decision Tree Structure (Root, Decision Nodes, Branches, and Leaf Nodes)")
plt.show()
