import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import os
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Threat explanation dictionary
threat_explanations = {
    "dos": "This is likely a Denial-of-Service attack — it attempts to make a system or service unavailable by flooding it with traffic.",
    "probe": "This might be a Probe — scanning the network to gather information or find vulnerabilities.",
    "r2l": "Remote to Local (R2L) — the attacker tries to gain unauthorized access from a remote machine.",
    "u2r": "User to Root (U2R) — someone is trying to escalate privileges from a regular user to an administrator/root.",
    "unknown": "The system flagged this as suspicious. Further investigation is advised."
}

# Define available models
MODEL_PATHS = {
    "Random Forest": "models/random_forest.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "XGBoost": "models/xgboost.pkl"
}
cat_cols = ['protocol_type', 'service', 'flag']

# UI setup
st.set_page_config(page_title="Intrusion Detection System", layout="wide")
st.title("🛡️ Intrusion Detection System (IDS)")

# === TABS ===
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "📂 Logs", "📊 Compare Models", "💬 Chat Assistant"])

# === TAB 1: Predict ===
with tab1:
    st.markdown("Upload a CSV file with 41 NSL-KDD features to detect intrusions.")
    selected_model_name = st.selectbox("⚙️ Select Model for Prediction", list(MODEL_PATHS.keys()))
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    simulate_live = st.checkbox("Simulate Live Intrusion Detection")

    # Load model and scaler
    model_path = MODEL_PATHS[selected_model_name]
    model = joblib.load(model_path)
    scaler = joblib.load(".venv/Scripts/scaler.pkl")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.drop('difficulty', axis=1, inplace=True, errors='ignore')
            df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
            X_full = scaler.transform(df.drop(columns=['label'], errors='ignore'))

            if simulate_live:
                st.subheader("🔴 Live Prediction Stream")
                pred_placeholder = st.empty()
                chart_placeholder = st.empty()
                progress_bar = st.progress(0)
                summary = {'Normal': 0, 'Attack': 0}
                live_log = []

                for i in range(len(X_full)):
                    row = X_full[i].reshape(1, -1)
                    pred = model.predict(row)[0]
                    label = 'Attack' if pred == 1 else 'Normal'
                    summary[label] += 1

                    attack_type = df.iloc[i].get('service', 'unknown')
                    explanation = threat_explanations.get(str(attack_type).lower(), threat_explanations['unknown'])

                    pred_placeholder.markdown(f"**Row {i+1} ➔ Prediction: `{label}`**")
                    if label == 'Attack':
                        pred_placeholder.info(f"💬 Explanation: {explanation}")

                    live_log.append({
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "row_number": i+1,
                        "prediction": label,
                        "model": selected_model_name,
                        "file": uploaded_file.name
                    })

                    fig, ax = plt.subplots()
                    sns.barplot(x=list(summary.keys()), y=list(summary.values()), ax=ax)
                    ax.set_title("Live Detection Count")
                    chart_placeholder.pyplot(fig)

                    progress_bar.progress((i + 1) / len(X_full))
                    time.sleep(0.3)

                log_df = pd.DataFrame(live_log)
                log_file = f"live_prediction_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                log_df.to_csv(log_file, index=False)
                st.success(f"Live log saved as {log_file}")

                csv = log_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Log CSV", csv, file_name=log_file, mime='text/csv')

            else:
                preds = model.predict(X_full)
                df['Prediction'] = ['Attack' if p == 1 else 'Normal' for p in preds]
                st.success("✅ Prediction complete!")
                st.dataframe(df[['Prediction']])

                summary = df['Prediction'].value_counts()
                st.subheader("📊 Prediction Summary")
                fig, ax = plt.subplots()
                sns.barplot(x=summary.index, y=summary.values, ax=ax)
                ax.set_title("Normal vs Attack")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# === TAB 2: Logs ===
with tab2:
    log_files = [f for f in os.listdir('.') if f.startswith("live_prediction_log_") and f.endswith(".csv")]
    if log_files:
        st.subheader("🕓 View Historical Logs")
        selected_log = st.selectbox("Choose a log file", log_files)
        if selected_log:
            log_df = pd.read_csv(selected_log)
            st.markdown(f"**Showing Log:** `{selected_log}`")
            st.dataframe(log_df)

            hist_summary = log_df['prediction'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=hist_summary.index, y=hist_summary.values, ax=ax)
            ax.set_title("Historical Detection Count")
            st.pyplot(fig)

# === TAB 3: Compare Models ===
with tab3:
    st.subheader("📊 Model Performance Comparison")

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
    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
    df.drop('difficulty', axis=1, inplace=True)

    X = df.drop('label', axis=1)
    y = df['label'].apply(lambda x: 1 if x == 'attack' else 0)

    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results_df.melt(id_vars='Model'), x='Model', y='value', hue='variable')
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0.9, 1.01)
    st.pyplot(fig)

# === TAB 4: Chat Assistant ===
with tab4:
    st.subheader("💬 Chat with IDS Assistant (Powered by ChatGPT)")
    st.markdown("Ask anything about how this IDS system works, security concepts, or predictions.")

    if prompt := st.chat_input("Ask your question here..."):
        st.chat_message("user").write(prompt)
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
            st.chat_message("assistant").write(answer)
        except Exception as e:
            st.error(f"ChatGPT error: {e}")

