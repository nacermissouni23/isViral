from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
import xgboost as xgb


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "xgb_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.pkl"
DATA_PATH = Path("data.csv")

CATEGORICAL_COLS = ["Platform", "Content_Type", "Category", "Day_of_Week", "Sentiment"]


@st.cache_resource
def load_model_and_features() -> tuple:
	if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
		missing = []
		if not MODEL_PATH.exists():
			missing.append(str(MODEL_PATH))
		if not FEATURES_PATH.exists():
			missing.append(str(FEATURES_PATH))
		raise FileNotFoundError(
			"Missing required artifact(s): " + ", ".join(missing)
		)

	with open(MODEL_PATH, "rb") as f:
		model = pickle.load(f)

	with open(FEATURES_PATH, "rb") as f:
		feature_columns = pickle.load(f)

	return model, feature_columns


@st.cache_data
def load_options() -> dict:
	if not DATA_PATH.exists():
		raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

	df = pd.read_csv(DATA_PATH)
	options = {}
	for col in CATEGORICAL_COLS:
		if col in df.columns:
			options[col] = sorted(df[col].dropna().astype(str).unique().tolist())
		else:
			options[col] = []
	return options


def preprocess_input(input_df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
	encoded = pd.get_dummies(input_df, columns=CATEGORICAL_COLS, drop_first=False)
	aligned = encoded.reindex(columns=feature_columns, fill_value=0)
	return aligned


def safe_options(values: list) -> list:
	return values if values else ["Unknown"]


def get_top_factors(model, processed_row: pd.DataFrame, top_n: int = 3) -> list:
	booster = model.get_booster()
	dmatrix = xgb.DMatrix(processed_row, feature_names=list(processed_row.columns))
	contribs = booster.predict(dmatrix, pred_contribs=True)[0]

	factors = []
	for idx, feature_name in enumerate(processed_row.columns):
		contribution = float(contribs[idx])
		if contribution != 0:
			factors.append((feature_name, contribution))

	factors.sort(key=lambda item: abs(item[1]), reverse=True)
	return factors[:top_n]


st.set_page_config(page_title="Viral Score Predictor", layout="centered")
st.title("Viral Score Predictor")
st.caption("Estimate post score using your trained XGBoost model")

try:
	model, feature_columns = load_model_and_features()
	options = load_options()
except Exception as exc:
	st.error(str(exc))
	st.stop()


with st.form("prediction_form"):
	st.subheader("Post Inputs")

	platform = st.selectbox("Platform", safe_options(options["Platform"]))
	content_type = st.selectbox("Content_Type", safe_options(options["Content_Type"]))
	category = st.selectbox("Category", safe_options(options["Category"]))
	day_of_week = st.selectbox("Day_of_Week", safe_options(options["Day_of_Week"]))
	sentiment = st.selectbox("Sentiment", safe_options(options["Sentiment"]))

	follower_count = st.number_input("Follower_Count", min_value=0, value=10000, step=100)
	hour_of_day = st.slider("Hour_of_Day", min_value=0, max_value=23, value=12)
	hashtag_count = st.number_input("Hashtag_Count", min_value=0, value=3, step=1)
	content_length = st.number_input("Content_Length", min_value=0, value=140, step=1)

	has_media = st.checkbox("Has_Media", value=True)
	is_verified = st.checkbox("Is_Verified", value=False)

	submitted = st.form_submit_button("Predict")


if submitted:
	raw_input = {
		"Platform": platform,
		"Content_Type": content_type,
		"Category": category,
		"Day_of_Week": day_of_week,
		"Sentiment": sentiment,
		"Follower_Count": float(follower_count),
		"Hour_of_Day": int(hour_of_day),
		"Hashtag_Count": float(hashtag_count),
		"Content_Length": float(content_length),
		"Has_Media": int(has_media),
		"Is_Verified": int(is_verified),
	}

	input_df = pd.DataFrame([raw_input])
	processed_df = preprocess_input(input_df, feature_columns)

	prediction = float(model.predict(processed_df)[0])
	prediction_clipped = max(0.0, min(1.0, prediction))
	prediction_percent = prediction_clipped * 100

	st.subheader("Prediction")
	st.metric("Predicted Score (0-1)", f"{prediction_clipped:.4f}")
	st.metric("Predicted Score (%)", f"{prediction_percent:.2f}%")

	st.subheader("Top Influencing Factors")
	top_factors = get_top_factors(model, processed_df, top_n=3)
	if not top_factors:
		st.write("No strong factors found for this input.")
	else:
		for feature_name, contribution in top_factors:
			direction = "increased" if contribution > 0 else "decreased"
			st.write(f"- {feature_name}: {direction} score by {abs(contribution):.4f}")
