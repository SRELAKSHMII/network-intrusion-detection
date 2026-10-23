import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "ids_fusion_model.pkl"

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Network Intrusion Detection System")
st.caption("CICIDS2017 • Multi-model, feature-group-fusion prototype")
st.warning(
    "Research prototype: this app classifies uploaded network-flow CSV records. "
    "It does not capture live network traffic. The current model was trained on "
    "BENIGN and PortScan classes only."
)


@st.cache_resource
def load_bundle(path_string: str):
    return joblib.load(path_string)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Add `ids_fusion_model.pkl` to the same GitHub "
        "repository folder as `app.py`, then redeploy the app."
    )
    st.stop()

try:
    bundle = load_bundle(str(MODEL_PATH))
except Exception as exc:
    st.error(
        "The model bundle could not be loaded. Confirm that the file was saved "
        "from your training notebook and that the deployed scikit-learn version "
        "matches the version used when training."
    )
    st.exception(exc)
    st.stop()

required_keys = {
    "models", "imputers", "feature_groups", "feature_columns", "label_encoder"
}
missing_keys = required_keys - set(bundle.keys())
if missing_keys:
    st.error(f"The model bundle is missing required items: {sorted(missing_keys)}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a network-flow CSV file",
    type=["csv"],
    help="Use a CSV with the same CICIDS2017 feature columns used during training.",
)

if uploaded_file is None:
    st.info("Upload a compatible CSV file to start detection.")
    st.markdown(
        "**Expected input:** numeric CICIDS2017 network-flow features. "
        "If the file contains a `Label` column, it will be ignored for prediction."
    )
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file, low_memory=False)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

raw_df.columns = raw_df.columns.astype(str).str.strip()
if raw_df.empty:
    st.error("The uploaded CSV has no records.")
    st.stop()

# Labels must never be used as predictive features.
input_df = raw_df.drop(columns=["Label"], errors="ignore").copy()
feature_columns = list(bundle["feature_columns"])
missing_features = [c for c in feature_columns if c not in input_df.columns]

if missing_features:
    st.error(
        f"This CSV is missing {len(missing_features)} feature column(s) required "
        "by the trained model. Please upload data with the same feature schema "
        "used during training. Examples of missing columns: "
        + ", ".join(missing_features[:10])
    )
    st.stop()

X_new = input_df.reindex(columns=feature_columns).copy()
X_new = X_new.apply(pd.to_numeric, errors="coerce")
X_new = X_new.replace([np.inf, -np.inf], np.nan)

# Remove records with no numeric information at all; individual missing values
# are handled by the group's saved training imputer.
usable_mask = ~X_new.isna().all(axis=1)
if not usable_mask.all():
    st.warning(
        f"Skipped {(~usable_mask).sum()} row(s) where every required feature was missing or non-numeric."
    )
    X_new = X_new.loc[usable_mask].copy()

if X_new.empty:
    st.error("No usable rows remain after input validation.")
    st.stop()

label_encoder = bundle["label_encoder"]
number_of_classes = len(label_encoder.classes_)
combined_probabilities = []
model_errors = []

for group_name, model in bundle["models"].items():
    try:
        group_columns = bundle["feature_groups"][group_name]
        imputer = bundle["imputers"][group_name]
        X_group = imputer.transform(X_new[group_columns])
        raw_probabilities = model.predict_proba(X_group)

        aligned = np.zeros((len(X_new), number_of_classes), dtype=float)
        for index, class_label in enumerate(model.classes_):
            aligned[:, int(class_label)] = raw_probabilities[:, index]
        combined_probabilities.append(aligned)
    except Exception as exc:
        model_errors.append(f"{group_name}: {exc}")

if model_errors:
    st.error("One or more feature-group models failed during prediction.")
    for message in model_errors:
        st.write(message)
    st.stop()

if not combined_probabilities:
    st.error("No trained feature-group models were found in the model bundle.")
    st.stop()

probabilities = np.mean(combined_probabilities, axis=0)
predicted_ids = probabilities.argmax(axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_ids)
confidence = probabilities.max(axis=1)

results = pd.DataFrame({
    "Predicted_Label": predicted_labels,
    "Prediction_Confidence": np.round(confidence, 4),
})
results["Status"] = np.where(
    results["Predicted_Label"].str.casefold() == "benign",
    "NORMAL",
    "WARNING: POSSIBLE ATTACK",
)

st.subheader("Detection summary")
col1, col2, col3 = st.columns(3)
total = len(results)
attack_count = int((results["Status"] != "NORMAL").sum())
normal_count = total - attack_count
col1.metric("Records analysed", f"{total:,}")
col2.metric("Potential attack records", f"{attack_count:,}")
col3.metric("Normal records", f"{normal_count:,}")

if attack_count:
    st.error(f"Potentially suspicious records detected: {attack_count:,}")
else:
    st.success("No records were classified as attacks in this upload.")

st.subheader("Predicted class distribution")
distribution = results["Predicted_Label"].value_counts().rename_axis("Class").to_frame("Records")
st.bar_chart(distribution)

st.subheader("Prediction results")
st.dataframe(results, use_container_width=True, hide_index=True)

csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download detection report (CSV)",
    data=csv_bytes,
    file_name="intrusion_detection_report.csv",
    mime="text/csv",
)

with st.expander("About this prototype"):
    st.write(
        "This app averages class probabilities from separate models trained on "
        "feature groups drawn from the same structured network-flow dataset. "
        "This is feature-group fusion, not integration of independent modalities "
        "such as packet payloads or host logs. A displayed probability is a model "
        "score and is not a guarantee of correctness. Validate results on suitable "
        "unseen traffic before using this for operational security decisions."
    )
