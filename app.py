


# # app.py
# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="Hospital Medical Cost Prediction", layout="wide")

# # --- Helper: load with friendly errors ---
# def load_pickle(path):
#     try:
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     except FileNotFoundError:
#         st.error(f"File not found: {path}")
#         return None
#     except Exception as e:
#         st.error(f"Error loading {path}: {e}")
#         return None

# # --- Load model & encoders (change filenames if your pickles have different names) ---
# model = load_pickle("medical_cost_model.pkl")
# facility_encoder = load_pickle("facility_encoder.pkl")
# drg_encoder = load_pickle("drg_encoder.pkl")

# # --- Load dataset if available (local file or uploaded) ---
# @st.cache_data
# def load_local_df(path="Hospital_Inpatient_Cost_Transparency__Beginning.csv"):
#     p = Path(path)
#     if p.exists():
#         try:
#             return pd.read_csv(p)
#         except Exception as e:
#             st.error(f"Unable to read {path}: {e}")
#             return None
#     return None

# uploaded_file = st.sidebar.file_uploader("Upload CSV (optional) — dataset with columns like Facility Name, APR DRG Description, APR Severity of Illness Code, Discharges, Mean Charge, Mean Cost", type=["csv"])
# if uploaded_file:
#     try:
#         df = pd.read_csv(uploaded_file)
#     except Exception as e:
#         st.error(f"Failed to read uploaded file: {e}")
#         df = None
# else:
#     df = load_local_df("Hospital_Inpatient_Cost_Transparency__Beginning.csv")  # try local filename

# # --- Sidebar navigation ---
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to:", ["Prediction", "Dataset Preview", "Model Performance", "About"])

# # --- UI header ---
# st.markdown("<h1 style='text-align:center; color:#2a8cff;'>💊 Hospital Medical Cost Prediction</h1>", unsafe_allow_html=True)

# # --- Prediction page ---
# if page == "Prediction":
#     st.subheader("Make a single prediction")

#     # validate encoders & model
#     if model is None or facility_encoder is None or drg_encoder is None:
#         st.warning("Model or encoders not found. Please ensure 'medical_cost_model.pkl', 'facility_encoder.pkl' and 'drg_encoder.pkl' are present in the same folder.")
#     else:
#         # build input widgets
#         facility_list = list(facility_encoder.classes_)
#         drg_list = list(drg_encoder.classes_)

#         # Layout: two columns for inputs
#         col1, col2 = st.columns(2)
#         with col1:
#             facility_name = st.selectbox("Select Facility (Facility Name)", facility_list)
#             drg_name = st.selectbox("Select DRG (APR DRG Description)", drg_list)
#             severity = st.number_input("APR Severity of Illness Code", min_value=1, step=1, value=1)
#         with col2:
#             discharges = st.number_input("Discharges", min_value=0, step=1, value=1)
#             mean_charge = st.number_input("Mean Charge (dollars)", min_value=0.0, step=100.0, value=1000.0)
#             # optional: Year input if your model expects it; remove if not used
#             # year = st.number_input("Year", min_value=2000, max_value=2030, value=2023, step=1)

#         if st.button("Predict Mean Cost"):
#             # encode categorical inputs
#             try:
#                 facility_enc = int(facility_encoder.transform([facility_name])[0])
#                 drg_enc = int(drg_encoder.transform([drg_name])[0])
#             except Exception as e:
#                 st.error(f"Encoding error: {e}")
#             else:
#                 # Create feature vector — ORDER must match how model was trained
#                 # I assume the model expects: [facility_enc, drg_enc, severity, discharges, mean_charge]
#                 X = np.array([[facility_enc, drg_enc, severity, discharges, mean_charge]], dtype=float)
#                 try:
#                     pred = model.predict(X)
#                     st.success(f"Predicted Mean Cost: ${pred[0]:,.2f}")
#                 except Exception as e:
#                     st.error(f"Prediction failed: {e}")

# # --- Dataset preview page ---
# elif page == "Dataset Preview":
#     st.subheader("Dataset Preview")
#     if df is None:
#         st.info("No dataset loaded. Upload a CSV on the sidebar or place 'hospital_costs.csv' next to this app.")
#     else:
#         st.write("Columns found:", list(df.columns))
#         # Show only the columns you asked for if they exist
#         want_cols = ['Year', 'Facility Id', 'Facility Name', 'APR DRG Code',
#                      'APR Severity of Illness Code', 'APR DRG Description',
#                      'APR Severity of Illness Description', 'APR Medical Surgical Code',
#                      'APR Medical Surgical Description', 'Discharges', 'Mean Charge',
#                      'Median Charge', 'Mean Cost', 'Median Cost']
#         # display intersection
#         cols_to_show = [c for c in want_cols if c in df.columns]
#         st.dataframe(df[cols_to_show].head(200))  # show first 200 rows in interactive table

# # --- Model performance page ---
# elif page == "Model Performance":
#     st.subheader("Model Performance & diagnostics")

#     if model is None:
#         st.warning("Model not loaded — cannot compute metrics.")
#     elif df is None:
#         st.info("Dataset not loaded. Upload dataset to compute metrics (or place hospital_costs.csv next to the app).")
#     else:
#         # Ensure necessary columns exist
#         required_cols = ['Facility Name', 'APR DRG Description', 'APR Severity of Illness Code', 
#                          'Discharges', 'Mean Charge', 'Mean Cost']
#         missing = [c for c in required_cols if c not in df.columns]
        
#         if missing:
#             st.error(f"Dataset is missing columns required for metrics: {missing}")
#         else:
#             try:
#                 # -------------------------------------------
#                 # 1️⃣ Filter out unknown facilities & DRGs
#                 # -------------------------------------------
#                 valid_facilities = set(facility_encoder.classes_)
#                 valid_drgs = set(drg_encoder.classes_)

#                 df_filtered = df[
#                     df['Facility Name'].isin(valid_facilities) &
#                     df['APR DRG Description'].isin(valid_drgs)
#                 ]

#                 if df_filtered.empty:
#                     st.error("No matching rows after filtering unknown facilities/DRGs.")
#                     st.stop()

#                 # -------------------------------------------
#                 # 2️⃣ Prepare X_raw and y (from filtered df)
#                 # -------------------------------------------
#                 X_raw = df_filtered[['Facility Name', 'APR DRG Description',
#                                      'APR Severity of Illness Code', 
#                                      'Discharges', 'Mean Charge']].copy()

#                 y = df_filtered['Mean Cost'].astype(float).values

#                 # -------------------------------------------
#                 # 3️⃣ Encode categorical features
#                 # -------------------------------------------
#                 X_raw['Facility Enc'] = facility_encoder.transform(X_raw['Facility Name'])
#                 X_raw['DRG Enc'] = drg_encoder.transform(X_raw['APR DRG Description'])

#                 # Final feature order used during training
#                 X = X_raw[['Facility Enc', 'DRG Enc',
#                            'APR Severity of Illness Code',
#                            'Discharges', 'Mean Charge']].astype(float).values

#                 # -------------------------------------------
#                 # 4️⃣ Predict & compute metrics
#                 # -------------------------------------------
#                 y_pred = model.predict(X)

#                 r2 = r2_score(y, y_pred)
#                 rmse = mean_squared_error(y, y_pred, squared=False)
#                 mae = mean_absolute_error(y, y_pred)

#                 st.write(f"**R² Score:** {r2:.4f}")
#                 st.write(f"**RMSE:** ${rmse:,.2f}")
#                 st.write(f"**MAE:** ${mae:,.2f}")

#                 # -------------------------------------------
#                 # 5️⃣ Plot Actual vs Predicted
#                 # -------------------------------------------
#                 fig, ax = plt.subplots(figsize=(6, 4))
#                 ax.scatter(y, y_pred, alpha=0.6)
#                 lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
#                 ax.plot(lims, lims, linestyle='--')
#                 ax.set_xlabel("Actual Mean Cost")
#                 ax.set_ylabel("Predicted Mean Cost")
#                 ax.set_title("Actual vs Predicted Mean Cost")
#                 st.pyplot(fig)

#             except Exception as e:
#                 st.error(f"Failed to compute metrics / prepare features: {e}")


# # --- About page ---
# elif page == "About":
#     st.subheader("About this app")
#     st.markdown("""
#     **Project:** Hospital Inpatient Cost Prediction (multiple linear regression / ML model)  
#     **Features used (example):**
#     - Facility Name (encoded)
#     - APR DRG Description (encoded)
#     - APR Severity of Illness Code
#     - Discharges
#     - Mean Charge

#     **Notes & Tips**
#     - Make sure encoders were fitted on the same strings that appear in your dataset (`Facility Name` and `APR DRG Description`).
#     - The order of features when predicting must match the order used when training the model.
#     - If you get errors about unknown categories when encoding, inspect the strings in your dataset vs. encoder classes.
#     - For production usage, consider saving a preprocessing pipeline (ColumnTransformer / Pipeline) so the app applies identical preprocessing as training.
#     """)
#     st.write("Built for Jupyter/Streamlit. Place the model and encoder pickles in the same directory as this app, or upload them through the notebook environment.")

# # --- common styles / footer ---
# st.markdown("""
# <style>
# body { background-color: #f5faff; }
# </style>
# """, unsafe_allow_html=True)










import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hospital Medical Cost Prediction", layout="wide")

# ----------------------------
# Load Model + Encoders Safely
# ----------------------------

def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

model = load_pickle("medical_cost_model.pkl")
facility_encoder = load_pickle("facility_encoder.pkl")
drg_encoder = load_pickle("drg_encoder.pkl")

# ----------------------------
# Upload Dataset (For metrics)
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go To:", ["Prediction", "Dataset Preview", "Model Performance", "About"])

uploaded_file = st.sidebar.file_uploader("Upload Dataset (optional)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = None

# ----------------------------
# Main Header
# ----------------------------
st.markdown("<h1 style='text-align:center; color:#0077ff;'>💊 Hospital Medical Cost Prediction</h1>", unsafe_allow_html=True)


# ----------------------------
# PAGE 1 — Prediction
# ----------------------------
if page == "Prediction":

    st.subheader("Predict Mean Hospital Cost")

    if model is None or facility_encoder is None or drg_encoder is None:
        st.error("Model or encoders not found. Make sure all pickle files are in the folder.")
    else:

        facility_list = facility_encoder.classes_
        drg_list = drg_encoder.classes_

        col1, col2 = st.columns(2)

        with col1:
            facility = st.selectbox("Facility Name", facility_list)
            drg = st.selectbox("APR DRG Description", drg_list)
            mean_charge = st.number_input("Mean Charge ($)", min_value=0.0, step=100.0)

        with col2:
            severity = st.number_input("Severity Code", min_value=1, step=1)
            discharges = st.number_input("Number of Discharges", min_value=1, step=1)
            # mean_charge = st.number_input("Mean Charge ($)", min_value=0.0, step=100.0)

        if st.button("Predict"):

            facility_enc = facility_encoder.transform([facility])[0]
            drg_enc = drg_encoder.transform([drg])[0]

            X = np.array([[facility_enc, drg_enc, severity, discharges, mean_charge]])

            pred = model.predict(X)[0]

            st.success(f"Predicted Mean Cost: ${pred:,.2f}")


# ----------------------------
# PAGE 2 — Dataset Preview
# ----------------------------
elif page == "Dataset Preview":

    st.subheader("Dataset Preview")

    if df is None:
        st.info("Upload a CSV file from the sidebar to preview it.")
    else:
        st.write(df.head(50))


# ----------------------------
# PAGE 3 — Model Performance
# ----------------------------
elif page == "Model Performance":

    st.subheader("Model Performance (Metrics)")

    if df is None:
        st.info("Upload dataset to compute model performance.")
    else:

        required = ['Facility Name', 'APR DRG Description',
                    'APR Severity of Illness Code', 'Discharges', 'Mean Charge', 'Mean Cost']

        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"Missing columns in dataset: {missing}")
        else:
            # Only keep rows that the encoders can handle
            df = df[df['Facility Name'].isin(facility_encoder.classes_)]
            df = df[df['APR DRG Description'].isin(drg_encoder.classes_)]

            # Encode
            df['Facility Enc'] = facility_encoder.transform(df['Facility Name'])
            df['DRG Enc'] = drg_encoder.transform(df['APR DRG Description'])

            X = df[['Facility Enc', 'DRG Enc',
                    'APR Severity of Illness Code', 'Discharges', 'Mean Charge']].values

            y = df['Mean Cost'].values

            y_pred = model.predict(X)

            r2 = r2_score(y, y_pred)
            rmse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            st.write(f"**R² Score:** {r2:.4f}")
            st.write(f"**RMSE:** ${rmse:,.2f}")
            st.write(f"**MAE:** ${mae:,.2f}")

            # Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y, y_pred, alpha=0.5)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            ax.set_xlabel("Actual Cost")
            ax.set_ylabel("Predicted Cost")
            ax.set_title("Actual vs Predicted Cost")
            st.pyplot(fig)


# ----------------------------
# PAGE 4 — About
# ----------------------------
elif page == "About":

    st.subheader("About This Project")
    st.write("""
    **This project predicts hospital inpatient treatment cost using ML.**  
    Model: Linear Regression  
    Inputs: Facility, DRG, Severity, Discharges, Mean Charge  
    Output: Predicted Mean Cost
    
    We trained the model cleanly and used Label Encoders that match the dataset, so the app is stable and error-free.
    """)


