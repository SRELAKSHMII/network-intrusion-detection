# Network Intrusion Detection System (NIDS)

A Streamlit demonstration app for binary network traffic classification using CICIDS2017 and a multi-model, feature-group-fusion approach.

## Current scope

- Predicts the classes available in the trained artifact (currently `BENIGN` and `PortScan`).
- Combines probability outputs from models trained on Timing, Packet Statistics, and Network Flow feature groups.
- Accepts CSV uploads containing the same feature columns used during model training.
- Displays predicted labels, simple warning status, class distribution, and a downloadable CSV report.

**Limitations:** This is a research prototype. It analyses uploaded records and does not capture live packets. The feature groups come from the same structured dataset, so this is feature-group fusion rather than fusion of genuinely independent modalities. Prediction confidence is not a guarantee of correctness.

## Files

- `app.py` — Streamlit dashboard
- `requirements.txt` — Python dependencies
- `ids_fusion_model.pkl` — trained model bundle exported from the Colab notebook (you must add this file)

Do not upload the raw CICIDS2017 dataset to this public repository. Only upload the model artifact if you are allowed to share it and the file is within GitHub's file-size limit.

## Run locally

1. Put `ids_fusion_model.pkl` beside `app.py`.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Upload a CSV that contains the same feature columns used to train the saved models.

## Deploy publicly

1. Push `app.py`, `requirements.txt`, `README.md`, and `ids_fusion_model.pkl` to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, set the main file path to `app.py`, then deploy.
4. Share the resulting `*.streamlit.app` URL.

## Model compatibility

The pickle/joblib artifact was created with the Python and scikit-learn environment used in Colab. If loading fails in the cloud, check the scikit-learn version used to train the artifact and pin that exact version in `requirements.txt`, then redeploy.
