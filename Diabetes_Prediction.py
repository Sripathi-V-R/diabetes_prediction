# === Diabetes Prediction App (Polished UI + Charts + Medical PDF) ===
# Save as: c:/Users/sripathivr/Tasks/Diabetes_Prediction/Diabetes_Prediction.py
# Run:     py -m streamlit run c:/Users/sripathivr/Tasks/Diabetes_Prediction/Diabetes_Prediction.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import tempfile
import os
from datetime import datetime
from pathlib import Path
import atexit

# ---------------------- CONFIG ----------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "lightgbm_best_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"                # optional (if you saved scaler separately)
DATA_PATH  = BASE_DIR / "preprocessed_dataset.csv"

ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "hospital_logo.png"
BANNER_PATH = ASSETS_DIR / "medical_banner.jpg"

APP_TITLE = "🏥 Diabetes Prediction Medical Portal"
PRIMARY_COLOR = "#7b1e1e"

HEALTHY_RANGES = {
    "Age (yrs)": (20, 50),
    "BMI": (18.5, 24.9),
    "HbA1c (%)": (4.0, 5.6),
    "Glucose (mg/dL)": (90, 140)
}

# Track temp files for cleanup
_TEMP_FILES = []
def _register_temp(path):
    _TEMP_FILES.append(path)
def _cleanup_temp():
    for p in _TEMP_FILES:
        try:
            os.remove(p)
        except Exception:
            pass
atexit.register(_cleanup_temp)

# ---------------------- LOAD ARTIFACTS ----------------------
@st.cache_resource
def load_model_and_scaler():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        loaded = pickle.load(f)

    model = None
    scaler = None

    if isinstance(loaded, dict):
        model = loaded.get("model") or loaded.get("estimator") or loaded.get("model_object")
        scaler = loaded.get("scaler") or loaded.get("preprocessor")
        if model is None and "pipeline" in loaded:
            model = loaded["pipeline"]
    else:
        model = loaded

    if scaler is None and SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

    if scaler is None:
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            expected_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
            present = [c for c in expected_cols if c in df.columns]
            if len(present) < 4:
                raise ValueError("Dataset missing required columns to fit scaler. Provide a saved scaler instead.")
            scaler = StandardScaler().fit(df[present])
        else:
            raise FileNotFoundError("Scaler not found and no dataset available to create one.")

    return model, scaler

# ---------------------- CHART HELPERS ----------------------
def draw_range_bars(values_dict):
    params = list(values_dict.keys())
    vals = [values_dict[k] for k in params]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    y_pos = np.arange(len(params))

    colors = []
    for i, p in enumerate(params):
        low, high = HEALTHY_RANGES.get(p, (0, np.max([vals[i], 1])))
        v = vals[i]
        out_of_range = (v < low) or (v > high)
        colors.append("tab:orange" if out_of_range else "tab:green")

    ax.barh(y_pos, vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.set_xlabel("Value")

    for i, p in enumerate(params):
        low, high = HEALTHY_RANGES.get(p, (0, 0))
        ax.axvline(low, linestyle="--", linewidth=0.9)
        ax.axvline(high, linestyle="--", linewidth=0.9)
        v = vals[i]
        ax.text(v if v < ax.get_xlim()[1] else ax.get_xlim()[1]*0.98, i,
                f"  {v}", va="center", ha="left" if v < ax.get_xlim()[1]*0.95 else "right")

    ax.set_title("Health Parameters vs Healthy Ranges")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def draw_radar_chart(values_dict):
    labels = list(values_dict.keys())
    values = [values_dict[k] for k in labels]

    def norm(val, rng):
        low, high = rng
        mid = (low + high) / 2
        span = max(high - low, 1e-6)
        score = 1 - min(abs(val - mid) / (span/2), 1)
        return max(0, min(1, score))

    ranges_map = {
        "Age (yrs)": HEALTHY_RANGES["Age (yrs)"],
        "BMI": HEALTHY_RANGES["BMI"],
        "HbA1c (%)": HEALTHY_RANGES["HbA1c (%)"],
        "Glucose (mg/dL)": HEALTHY_RANGES["Glucose (mg/dL)"]
    }

    scores = [norm(values[i], ranges_map[labels[i]]) for i in range(len(labels))]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    scores_loop = scores + [scores[0]]
    angles_loop = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(5.8, 5.8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_loop, scores_loop, linewidth=2)
    ax.fill(angles_loop, scores_loop, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim(0, 1)
    ax.set_yticklabels(["Poor", "Average", "Good"])
    ax.set_title("Overall Health Profile (Radar)")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ---------------------- PDF REPORT ----------------------
def generate_medical_pdf(patient, prediction_text, range_bars_png, radar_png):
    pdf = FPDF()
    pdf.add_page()

    try:
        dejavu_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        if dejavu_path.exists():
            pdf.add_font("DejaVu", "", str(dejavu_path), uni=True)
            pdf.set_font("DejaVu", "", 12)
        else:
            pdf.set_font("Arial", "", 12)
    except Exception:
        pdf.set_font("Arial", "", 12)

    if LOGO_PATH.exists():
        try:
            pdf.image(str(LOGO_PATH), 10, 8, 25)
        except Exception:
            pass

    pdf.set_font_size(16)
    pdf.set_font(pdf.font_family, style="B")
    pdf.cell(0, 10, "Diabetes Prediction Medical Report", ln=True, align="C")
    pdf.set_font_size(11)
    pdf.cell(0, 6, "Sunrise Diagnostics & Wellness Center", ln=True, align="C")
    pdf.cell(0, 6, "123 Health Street, Wellness City", ln=True, align="C")
    pdf.cell(0, 6, "Email: care@sunriseclinic.example", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font_size(12)
    pdf.set_font(pdf.font_family, style="B")
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font(pdf.font_family, style="")

    name_prefix = "Mr." if patient["gender"] == "Male" else "Ms." if patient["gender"] == "Female" else "Mx."
    pdf.multi_cell(0, 7,
        f"Name: {name_prefix} {patient['name']}\n"
        f"Gender: {patient['gender']}\n"
        f"Age: {patient['age']} years\n"
        f"Hypertension: {patient['hypertension']}\n"
        f"Heart Disease: {patient['heart_disease']}\n"
        f"Smoking History: {patient['smoking_history']}\n"
        f"Report Date/Time: {patient['timestamp']}"
    )

    pdf.set_font(pdf.font_family, style="B")
    pdf.cell(0, 8, "Prediction", ln=True)
    pdf.set_font(pdf.font_family, style="")
    if "Diabetic" in prediction_text:
        pdf.set_text_color(180, 0, 0)
    else:
        pdf.set_text_color(0, 120, 0)
    pdf.multi_cell(0, 8, prediction_text)
    pdf.set_text_color(0, 0, 0)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1:
            f1.write(range_bars_png)
            range_path = f1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
            f2.write(radar_png)
            radar_path = f2.name

        pdf.image(range_path, x=12, y=120, w=90)
        pdf.image(radar_path, x=110, y=120, w=90)
    except Exception:
        pass

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.close()
    pdf.output(temp_pdf.name)
    with open(temp_pdf.name, "rb") as f:
        return f.read()

# ---------------------- MAIN APP ----------------------
def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🩺")

    header_cols = st.columns([1, 3, 1])
    with header_cols[0]:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
    with header_cols[1]:
        st.markdown(f"<h1 style='text-align:center;color:{PRIMARY_COLOR};margin:0'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    with header_cols[2]:
        if BANNER_PATH.exists():
            st.image(str(BANNER_PATH), use_container_width=True)

    st.markdown("---")

    with st.sidebar:
        st.header("About")
        st.write("This app predicts diabetes risk using a trained LightGBM model.")
        st.write("⚠️ Not a medical diagnosis. Consult a doctor.")

    tab1, tab2, tab3 = st.tabs(["📝 Input", "📊 Results", "📄 Report"])

    if "patient" not in st.session_state:
        st.session_state["patient"] = {}
    if "inputs_ready" not in st.session_state:
        st.session_state["inputs_ready"] = False

    # -------- Input Tab --------
    with tab1:
        st.subheader("Enter Patient Details")
        name = st.text_input("Full Name", "John Doe")
        gender = st.radio("Gender", options=("Male", "Female", "Others"))
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=5.5, format="%.1f")
        glucose = st.number_input("Blood Glucose (mg/dL)", min_value=0.0, max_value=400.0, value=120.0, format="%.1f")
        hypertension = st.selectbox("Hypertension", ("No", "Yes"))
        heart_disease = st.selectbox("Heart Disease", ("No", "Yes"))
        smoking_history = st.selectbox("Smoking History", ("never", "current", "formerly", "No Info", "ever", "not current"))

        st.session_state["patient"] = {
            "name": name.strip(),
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "hba1c": hba1c,
            "glucose": glucose,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state["inputs_ready"] = True

    # -------- Results Tab --------
    with tab2:
        st.subheader("Prediction & Visualizations")
        if not st.session_state.get("inputs_ready"):
            st.warning("Enter patient details in Input tab first.")
        else:
            try:
                model, scaler = load_model_and_scaler()
            except Exception as e:
                st.error(f"Error loading model/scaler: {e}")
                st.stop()

            p = st.session_state["patient"]

            gender_numeric = 1 if p["gender"] == "Male" else 0
            hypertension_numeric = 1 if p["hypertension"] == "Yes" else 0
            heart_disease_numeric = 1 if p["heart_disease"] == "Yes" else 0
            smoking_map = {"never":0, "current":1, "formerly":2, "No Info":3, "ever":4, "not current":5}
            smoking_numeric = smoking_map.get(p["smoking_history"], 3)

            inputs_num = np.array([[p["age"], p["bmi"], p["hba1c"], p["glucose"]]])
            scaled = scaler.transform(inputs_num)

            feature_vector = np.concatenate((
                [gender_numeric, hypertension_numeric, heart_disease_numeric, smoking_numeric],
                scaled.flatten()
            )).reshape(1, -1)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feature_vector)[:, 1][0]
                pred_label = 1 if proba >= 0.5 else 0
                result = "Diabetic" if pred_label == 1 else "Non-Diabetic"
                st.info(f"Model probability (Diabetic): {proba:.2%}")
            else:
                pred = model.predict(feature_vector)
                result = "Diabetic" if int(pred[0]) == 1 else "Non-Diabetic"

            if result == "Diabetic":
                st.error("⚠️ Prediction: **Diabetic**")
            else:
                st.success("🎉 Prediction: **Non-Diabetic**")

            values = {"Age (yrs)": p["age"], "BMI": p["bmi"], "HbA1c (%)": p["hba1c"], "Glucose (mg/dL)": p["glucose"]}
            st.image(draw_range_bars(values), caption="Health Parameters vs Healthy Ranges")
            st.image(draw_radar_chart(values), caption="Overall Health Profile (Radar)")

            st.session_state["prediction_text"] = f"Prediction: {result}"

    # -------- Report Tab --------
    with tab3:
        st.subheader("Download Report")
        if "prediction_text" not in st.session_state:
            st.info("Generate a prediction in Results tab first.")
        else:
            pdf_bytes = generate_medical_pdf(
                patient=st.session_state["patient"],
                prediction_text=st.session_state["prediction_text"],
                range_bars_png=draw_range_bars({
                    "Age (yrs)": st.session_state["patient"]["age"],
                    "BMI": st.session_state["patient"]["bmi"],
                    "HbA1c (%)": st.session_state["patient"]["hba1c"],
                    "Glucose (mg/dL)": st.session_state["patient"]["glucose"]
                }),
                radar_png=draw_radar_chart({
                    "Age (yrs)": st.session_state["patient"]["age"],
                    "BMI": st.session_state["patient"]["bmi"],
                    "HbA1c (%)": st.session_state["patient"]["hba1c"],
                    "Glucose (mg/dL)": st.session_state["patient"]["glucose"]
                }),
            )
            st.download_button("⬇️ Download Medical PDF", data=pdf_bytes, file_name="Medical_Report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()


# Run:     py -m streamlit run c:/Users/sripathivr/Tasks/Diabetes_Prediction/Diabetes_Prediction.py

