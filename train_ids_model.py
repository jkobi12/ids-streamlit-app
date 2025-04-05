# train_ids_model.py

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load dataset
train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

cat_cols = ['protocol_type', 'service', 'flag']

df = pd.read_csv(train_url, names=columns)
df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
df.drop('difficulty', axis=1, inplace=True)

X = df.drop('label', axis=1)
y = df['label'].apply(lambda x: 1 if x == 'attack' else 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model configurations
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0),
    "SVM": SVC(kernel='rbf', probability=True)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n🧠 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"🔍 Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred, target_names=["Normal", "Attack"]))

# Save best model (Random Forest) and scaler
joblib.dump(models["Random Forest"], "ids_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ Saved Random Forest model as 'ids_model.pkl' and scaler as 'scaler.pkl'")
