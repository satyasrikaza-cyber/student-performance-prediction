from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_preprocessing import FEATURE_COLUMNS
from src.predict import (
    load_trained_model,
    performance_category,
    predict_grade,
    top_feature_importance,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METRICS_PATH = PROJECT_ROOT / "outputs" / "model_comparison.csv"
VISUALIZATIONS_DIR = PROJECT_ROOT / "outputs" / "visualizations"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon=":material/school:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f4f7fb; }
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dfe7f1;
        border-radius: 12px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model(model_path: str):
    return load_trained_model(Path(model_path))


@st.cache_data
def get_metrics(metrics_path: str) -> pd.DataFrame:
    return pd.read_csv(metrics_path)


def choose_code(label: str, options: dict[str, str]) -> str:
    """Show friendly labels and return the short code used in the dataset."""
    selected_label = st.selectbox(label, list(options))
    return options[selected_label]


st.title("Student Performance Prediction")
st.write(
    "Estimate a student's final mathematics grade (G3, from 0 to 20) using "
    "academic, personal, family, and social information from the UCI Student "
    "Performance Dataset."
)
st.info(
    "G1 and G2 are the student's first- and second-period grades. They are "
    "earlier assessments, not the final G3 grade. This is an educational "
    "demonstration, not a decision-making tool for students."
)

if not MODEL_PATH.is_file() or not METRICS_PATH.is_file():
    st.error(
        "The trained model or model metrics are missing. In the VS Code "
        "terminal, run `python -m src.train_model` from this project folder, "
        "then refresh this page."
    )
    st.stop()

try:
    model = get_model(str(MODEL_PATH))
    metrics = get_metrics(str(METRICS_PATH))
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not load the trained project files: {error}")
    st.stop()

st.header("Enter student details")
with st.form("student_details"):
    academic_col, personal_col, family_col = st.columns(3)

    with academic_col:
        st.subheader("Academic information")
        g1 = st.number_input("G1: first-period grade (0-20)", 0, 20, 10)
        g2 = st.number_input("G2: second-period grade (0-20)", 0, 20, 10)
        studytime = st.selectbox(
            "Weekly study time",
            [1, 2, 3, 4],
            format_func=lambda value: {
                1: "Less than 2 hours",
                2: "2 to 5 hours",
                3: "5 to 10 hours",
                4: "More than 10 hours",
            }[value],
        )
        failures = st.number_input(
            "Past class failures", min_value=0, max_value=4, value=0, step=1
        )
        absences = st.number_input(
            "School absences", min_value=0, max_value=93, value=0, step=1
        )
        school = choose_code("School", {"Gabriel Pereira (GP)": "GP", "Mousinho da Silveira (MS)": "MS"})
        paid = choose_code("Extra paid classes", {"No": "no", "Yes": "yes"})
        schoolsup = choose_code("Extra school support", {"No": "no", "Yes": "yes"})

    with personal_col:
        st.subheader("Personal information")
        sex = choose_code("Gender in dataset", {"Female": "F", "Male": "M"})
        age = st.number_input("Age", min_value=15, max_value=22, value=17, step=1)
        address = choose_code("Home area", {"Urban": "U", "Rural": "R"})
        traveltime = st.selectbox(
            "Travel time to school",
            [1, 2, 3, 4],
            format_func=lambda value: {
                1: "Less than 15 minutes",
                2: "15 to 30 minutes",
                3: "30 minutes to 1 hour",
                4: "More than 1 hour",
            }[value],
        )
        activities = choose_code("Extracurricular activities", {"No": "no", "Yes": "yes"})
        nursery = choose_code("Attended nursery school", {"Yes": "yes", "No": "no"})
        higher = choose_code("Wants higher education", {"Yes": "yes", "No": "no"})
        internet = choose_code("Internet access at home", {"Yes": "yes", "No": "no"})
        romantic = choose_code("In a romantic relationship", {"No": "no", "Yes": "yes"})

    with family_col:
        st.subheader("Family and social information")
        famsize = choose_code("Family size", {"3 or fewer": "LE3", "More than 3": "GT3"})
        pstatus = choose_code("Parents live together", {"Together": "T", "Apart": "A"})
        medu = st.selectbox(
            "Mother's education level (0-4)", [0, 1, 2, 3, 4], index=2
        )
        fedu = st.selectbox(
            "Father's education level (0-4)", [0, 1, 2, 3, 4], index=2
        )
        mjob = choose_code(
            "Mother's job",
            {"At home": "at_home", "Health care": "health", "Other": "other",
             "Services": "services", "Teacher": "teacher"},
        )
        fjob = choose_code(
            "Father's job",
            {"At home": "at_home", "Health care": "health", "Other": "other",
             "Services": "services", "Teacher": "teacher"},
        )
        reason = choose_code(
            "Reason for choosing school",
            {"Course preference": "course", "Close to home": "home",
             "School reputation": "reputation", "Other": "other"},
        )
        guardian = choose_code(
            "Primary guardian", {"Mother": "mother", "Father": "father", "Other": "other"}
        )
        famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
        famsup = choose_code("Family educational support", {"Yes": "yes", "No": "no"})
        freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
        goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
        dalc = st.slider("Workday alcohol consumption (1-5)", 1, 5, 1)
        walc = st.slider("Weekend alcohol consumption (1-5)", 1, 5, 1)
        health = st.slider("Current health (1-5)", 1, 5, 3)

    student = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": pstatus,
        "Medu": medu,
        "Fedu": fedu,
        "Mjob": mjob,
        "Fjob": fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": dalc,
        "Walc": walc,
        "health": health,
        "absences": absences,
        "G1": g1,
        "G2": g2,
    }
    predict_clicked = st.form_submit_button(
        "Predict final grade", type="primary", width="stretch"
    )

if predict_clicked:
    try:
        prediction = predict_grade(model, pd.DataFrame([student], columns=FEATURE_COLUMNS))
    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Prediction could not be made: {error}")
    else:
        category = performance_category(prediction)
        st.header("Prediction result")
        result_col, category_col = st.columns(2)
        result_col.metric("Predicted final grade (G3)", f"{prediction:.1f} / 20")
        category_col.metric("Performance category", category)

st.header("Model performance")
selected_rows = metrics.loc[metrics["selected"]]
if not selected_rows.empty:
    selected = selected_rows.iloc[0]
    st.caption(
        f"Selected model: {selected['model']} (chosen using the lowest "
        "5-fold cross-validation MAE on the training data). Test metrics are "
        "reported for comparison only."
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("MAE", f"{selected['test_mae']:.2f}")
    metric_cols[1].metric("MSE", f"{selected['test_mse']:.2f}")
    metric_cols[2].metric("RMSE", f"{selected['test_rmse']:.2f}")
    metric_cols[3].metric("R-squared", f"{selected['test_r2']:.3f}")
st.dataframe(metrics, hide_index=True, width="stretch")

st.header("Exploratory data visualizations")
chart_files = [
    ("Grade distribution", "grade_distribution.png"),
    ("Study time and grade", "study_time_vs_grade.png"),
    ("Absences and grade", "absences_vs_grade.png"),
    ("Previous grades and final grade", "previous_grades_vs_final_grade.png"),
    ("Gender and performance", "gender_vs_grade.png"),
    ("Parental education and performance", "parental_education_vs_grade.png"),
    ("Correlation heatmap", "correlation_heatmap.png"),
]
for start in range(0, len(chart_files), 2):
    chart_cols = st.columns(2)
    for column, (title, filename) in zip(chart_cols, chart_files[start : start + 2]):
        chart_path = VISUALIZATIONS_DIR / filename
        if chart_path.is_file():
            column.image(str(chart_path), caption=title, width="stretch")
        else:
            column.warning(f"Visualization not found: {filename}")

importance = top_feature_importance(model)
if not importance.empty:
    st.header("Most influential model features")
    st.caption(
        "These are model feature importances, not proof that a factor causes "
        "a change in grades."
    )
    st.bar_chart(importance.set_index("feature")["importance"])

st.caption(
    "Dataset: UCI Student Performance, mathematics subject. Predictions are "
    "estimates from historical data and should not be used to label or limit "
    "students."
)