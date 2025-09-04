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
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "lightgbm_best_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"  # optional, recommended to save scaler during training
DATA_PATH = BASE_DIR / "preprocessed_dataset.csv"

ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "hospital_logo.png"
BANNER_PATH = ASSETS_DIR / "medical_banner.jpg"

APP_TITLE = "🏥 Diabetes Prediction Medical Portal"
PRIMARY_COLOR = "#7b1e1e"

HEALTHY_RANGES = {
    "Age (yrs)": (20, 50),
    "BMI": (18.5, 24.9),
    "HbA1c (%)": (4.0, 5.6),
    "Glucose (mg/dL)": (90, 140),
}

# ---------------------- TEMP FILE CLEANUP ----------------------
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
                raise ValueError(
                    "Dataset does not contain required columns to fit a scaler. Please provide a saved scaler."
                )
            scaler = StandardScaler().fit(df[present])
        else:
            raise FileNotFoundError(
                "Scaler not found and DATA_PATH not available to fit one. Please provide a saved scaler."
            )

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
        ax.text(
            v if v < ax.get_xlim()[1] else ax.get_xlim()[1] * 0.98,
            i,
            f"  {v}",
            va="center",
            ha="left" if v < ax.get_xlim()[1] * 0.95 else "right",
        )

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
        score = 1 - min(abs(val - mid) / (span / 2), 1)
        return max(0, min(1, score))

    ranges_map = {
        "Age (yrs)": HEALTHY_RANGES["Age (yrs)"],
        "BMI": HEALTHY_RANGES["BMI"],
        "HbA1c (%)": HEALTHY_RANGES["HbA1c (%)"],
        "Glucose (mg/dL)": HEALTHY_RANGES["Glucose (mg/dL)"],
    }

    scores = [norm(values[i], ranges_map[labels[i]]) for i in range(len(labels))]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
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

    # ✅ Load Unicode-capable font (DejaVuSans)
    # Works on Streamlit Cloud (Linux) at this path
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font not found at {font_path}. Install 'fonts-dejavu-core'.")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 12)

    # ---- Header ----
    if os.path.exists(LOGO_PATH):
        try:
            pdf.image(str(LOGO_PATH), 10, 8, 25)
        except Exception:
            pass

    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 10, "Diabetes Prediction Medical Report", ln=True, align="C")
    pdf.set_font("DejaVu", "", 11)
    pdf.cell(0, 6, "Sunrise Diagnostics & Wellness Center", ln=True, align="C")
    pdf.cell(0, 6, "123 Health Street, Wellness City  +91-98765-43210", ln=True, align="C")
    pdf.cell(0, 6, "Email: care@sunriseclinic.example", ln=True, align="C")
    pdf.ln(6)
    pdf.set_draw_color(120, 30, 30)
    pdf.set_line_width(0.8)
    pdf.line(10, 45, 200, 45)
    pdf.ln(6)

    # ---- Patient Info ----
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font("DejaVu", "", 12)

    name_prefix = "Mr." if patient["gender"] == "Male" else "Ms." if patient["gender"] == "Female" else "Mx."
    pdf.multi_cell(
        0,
        7,
        f"Name: {name_prefix} {patient['name']}\n"
        f"Gender: {patient['gender']}\n"
        f"Age: {patient['age']} years\n"
        f"Hypertension: {patient['hypertension']}\n"
        f"Heart Disease: {patient['heart_disease']}\n"
        f"Smoking History: {patient['smoking_history']}\n"
        f"Report Date/Time: {patient['timestamp']}",
    )
    pdf.ln(2)

    # ---- Measurements ----
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 8, "Measurements", ln=True)
    pdf.set_font("DejaVu", "", 12)
    col_w = [60, 60, 60]
    row_h = 8

    def row(a, b, c):
        pdf.cell(col_w[0], row_h, a, 1)
        pdf.cell(col_w[1], row_h, b, 1)
        pdf.cell(col_w[2], row_h, c, 1)
        pdf.ln(row_h)

    pdf.set_font("DejaVu", "B", 12)
    row("Parameter", "Value", "Healthy Range")
    pdf.set_font("DejaVu", "", 12)
    row("BMI", f"{patient['bmi']}", "18.5 - 24.9")
    row("HbA1c (%)", f"{patient['hba1c']}", "4.0 - 5.6")
    row("Glucose (mg/dL)", f"{patient['glucose']}", "90 - 140")
    pdf.ln(4)

    # ---- Prediction ----
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 8, "Prediction", ln=True)
    pdf.set_font("DejaVu", "", 12)

    if "Diabetic" in prediction_text:
        pdf.set_text_color(200, 0, 0)
    else:
        pdf.set_text_color(0, 150, 0)

    pdf.multi_cell(0, 8, prediction_text)
    pdf.set_text_color(0, 0, 0)

    # ---- Charts ----
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1:
            f1.write(range_bars_png)
            range_path = f1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
            f2.write(radar_png)
            radar_path = f2.name

        pdf.ln(2)
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 8, "Visual Summary", ln=True)
        y_img = pdf.get_y() + 2
        pdf.image(range_path, x=12, y=y_img, w=90)
        pdf.image(radar_path, x=110, y=y_img, w=90)
        pdf.ln(75)

    except Exception:
        pass

    # ---- Recommendation ----
    pdf.ln(2)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 8, "Doctor's Recommendation", ln=True)
    pdf.set_font("DejaVu", "", 12)
    recommendation = (
        "⚠️ The model predicts a risk of Diabetes. Please consult a physician."
        if "Diabetic" in prediction_text
        else "✅ No diabetes indicated. Maintain a healthy lifestyle."
    )
    pdf.multi_cell(0, 7, recommendation)

    # ---- Footer ----
    pdf.set_y(-25)
    pdf.set_font("DejaVu", "I", 9)
    pdf.cell(
        0,
        8,
        "This is a computer-generated report and should not replace professional medical advice.",
        0,
        1,
        "C",
    )

    # ---- Save ----
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.close()
    pdf.output(temp_pdf.name)
    with open(temp_pdf.name, "rb") as f:
        pdf_bytes = f.read()
    return pdf_bytes

# ---------------------- APP ----------------------
def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🩺")

    # ---- Header ----
    header_cols = st.columns([1, 3, 1])
    with header_cols[0]:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
    with header_cols[1]:
        st.markdown(
            f"<h1 style='text-align:center;color:{PRIMARY_COLOR};margin:0'>{APP_TITLE}</h1>",
            unsafe_allow_html=True,
        )
    with header_cols[2]:
        if BANNER_PATH.exists():
            st.image(str(BANNER_PATH), use_container_width=True)

    st.markdown("---")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("About")
        st.write("This app predicts diabetes risk using a trained LightGBM model.")
        st.write("**Note:** This is not a medical diagnosis. Consult a doctor.")
        st.markdown(
            "**Tips**\n- Use numeric inputs for accurate results.\n- Save scaler used in training and provide it (recommended)."
        )

    tab1, tab2, tab3 = st.tabs(["📝 Input", "📊 Results", "📄 Report"])

    if "patient" not in st.session_state:
        st.session_state["patient"] = {}
    if "inputs_ready" not in st.session_state:
        st.session_state["inputs_ready"] = False

    # -------- Input Tab --------
    def safe_float_input(label, default):
        val = st.text_input(label, value=str(default))
        try:
            return float(val)
        except ValueError:
            st.warning(f" Please enter a valid number for {label}")
            return default

    with tab1:
        st.subheader("Enter Patient Details")
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Full Name", value=st.session_state["patient"].get("name", "John Doe"))
            gender = st.radio(
                "Gender",
                options=("Male", "Female", "Others"),
                horizontal=True,
                index=0 if st.session_state["patient"].get("gender", "Male") == "Male" else 1,
            )
            age = safe_float_input("Age (years)", st.session_state["patient"].get("age", 30))
            bmi = safe_float_input("BMI", st.session_state["patient"].get("bmi", 25.0))
            hba1c = safe_float_input("HbA1c Level (%)", st.session_state["patient"].get("hba1c", 5.5))
            glucose = safe_float_input("Blood Glucose (mg/dL)", st.session_state["patient"].get("glucose", 120.0))

        with c2:
            hypertension = st.selectbox(
                "Hypertension",
                ("No", "Yes"),
                index=0 if st.session_state["patient"].get("hypertension", "No") == "No" else 1,
            )
            heart_disease = st.selectbox(
                "Heart Disease",
                ("No", "Yes"),
                index=0 if st.session_state["patient"].get("heart_disease", "No") == "No" else 1,
            )
            smoking_history = st.selectbox(
                "Smoking History",
                ("never", "current", "formerly", "No Info", "ever", "not current"),
                index=0,
            )

        st.session_state["patient"] = {
            "name": name.strip(),
            "gender": gender,
            "age": float(age),
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": float(bmi),
            "hba1c": float(hba1c),
            "glucose": float(glucose),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        st.session_state["inputs_ready"] = True
        st.success("Inputs updated — go to the Results tab to compute prediction.")

    # -------- Results Tab --------
    with tab2:
        st.subheader("Prediction & Visualizations")
        if not st.session_state.get("inputs_ready"):
            st.warning("Please complete valid inputs in the Input tab first.")
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
            smoking_map = {"never": 0, "current": 1, "formerly": 2, "No Info": 3, "ever": 4, "not current": 5}
            smoking_numeric = smoking_map.get(p["smoking_history"], 3)

            inputs_num = np.array([[p["age"], p["bmi"], p["hba1c"], p["glucose"]]])
            try:
                scaled = scaler.transform(inputs_num)
            except Exception as e:
                st.error(f"Scaler transform failed: {e}")
                st.stop()

            feature_vector = np.concatenate(
                ([scaled[0][0], scaled[0][1], scaled[0][2], scaled[0][3]],
                 [gender_numeric, hypertension_numeric, heart_disease_numeric, smoking_numeric])
            ).reshape(1, -1)

            try:
                pred = model.predict(feature_vector)
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(feature_vector)[0][1]
                else:
                    prob = None
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            pred_text = f"Prediction: {'Diabetic' if pred[0]==1 else 'Not Diabetic'}"
            if prob is not None:
                pred_text += f" (Risk Probability: {prob:.2%})"

            st.markdown(
                f"<h3 style='color:{'red' if pred[0]==1 else 'green'}'>{pred_text}</h3>",
                unsafe_allow_html=True,
            )

            values_dict = {
                "Age (yrs)": p["age"],
                "BMI": p["bmi"],
                "HbA1c (%)": p["hba1c"],
                "Glucose (mg/dL)": p["glucose"],
            }
            rb_img = draw_range_bars(values_dict)
            radar_img = draw_radar_chart(values_dict)

            st.image(rb_img, caption="Health Ranges Comparison", use_container_width=True)
            st.image(radar_img, caption="Overall Health Radar", use_container_width=True)

            st.session_state["pred_text"] = pred_text
            st.session_state["rb_img"] = rb_img
            st.session_state["radar_img"] = radar_img

    # -------- Report Tab --------
    with tab3:
        st.subheader("Download PDF Medical Report")
        if "pred_text" not in st.session_state:
            st.warning("Please compute a prediction in Results tab first.")
        else:
            pdf_bytes = generate_medical_pdf(
                st.session_state["patient"],
                st.session_state["pred_text"],
                st.session_state["rb_img"],
                st.session_state["radar_img"],
            )
            st.download_button(
                "📥 Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"Diabetes_Report_{st.session_state['patient']['name'].replace(' ', '_')}.pdf",
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()

# Run:     py -m streamlit run c:/Users/sripathivr/Tasks/Diabetes_Prediction/Diabetes_Prediction.py






