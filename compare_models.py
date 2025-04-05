import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# UI setup
st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("📊 Compare IDS Models")

# Load training data
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

df = pd.read_csv(train_url, names=columns)
df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
cat_cols = ['protocol_type', 'service', 'flag']
df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
df.drop('difficulty', axis=1, inplace=True)

X = df.drop('label', axis=1)
y = df['label'].apply(lambda x: 1 if x == 'attack' else 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
st.subheader("🧠 Training & Comparing Models")
comparison_models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
}

results = []
for name, clf in comparison_models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1 Score": f1_score(y_val, y_pred)
    })

# Display results
results_df = pd.DataFrame(results)
st.dataframe(results_df)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=results_df.melt(id_vars='Model'), x='Model', y='value', hue='variable')
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0.9, 1.01)
st.pyplot(fig)
