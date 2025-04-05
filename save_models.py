import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Dummy data (5 samples, 41 features like NSL-KDD)
X_dummy = np.zeros((5, 41))
y_dummy = np.array([0, 1, 0, 1, 0])

# Folder to store models
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_dummy, y_dummy)
joblib.dump(rf, f"{model_dir}/random_forest.pkl")

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_dummy, y_dummy)
joblib.dump(dt, f"{model_dir}/decision_tree.pkl")

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_dummy, y_dummy)
joblib.dump(xgb, f"{model_dir}/xgboost.pkl")

print("✅ Models saved successfully to /models folder.")
