import os
import io
from pathlib import Path
from datetime import datetime
import datetime as dt

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import requests
from sklearn.preprocessing import LabelEncoder

try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None


# ---------------- Utility Functions ----------------

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def ensure_datetime(df: pd.DataFrame, col_candidates: list[str]) -> pd.DataFrame:
    for c in col_candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def derive_time_columns(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    if datetime_col not in df.columns:
        return df
    df["date"] = df[datetime_col].dt.date
    df["year_week"] = df[datetime_col].dt.strftime("%G-W%V")
    df["hour"] = df[datetime_col].dt.hour
    return df


def infer_lat_lon_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lat_candidates = ["lat", "latitude", "Start_Lat", "Latitude"]
    lon_candidates = ["lon", "longitude", "Start_Lng", "Longitude", "lng"]
    lat = next((c for c in lat_candidates if c in df.columns), None)
    lon = next((c for c in lon_candidates if c in df.columns), None)
    return lat, lon


def compute_daily_weekly_hazards(df: pd.DataFrame, datetime_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or datetime_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    grp_daily = df.groupby("date").size().reset_index(name="incidents")
    grp_weekly = df.groupby("year_week").size().reset_index(name="incidents")
    return grp_daily, grp_weekly


def make_heatmap(df: pd.DataFrame, lat_col: str, lon_col: str, tiles: str = "cartodbpositron"):
    if folium is None or df.empty:
        return None
    center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=center, zoom_start=6, tiles=tiles)
    for _, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        if pd.notna(lat) and pd.notna(lon):
            folium.CircleMarker(location=[lat, lon], radius=2, color="#e74c3c", fill=True, fill_opacity=0.5).add_to(m)
    return m


def generate_recommendations(df: pd.DataFrame) -> list[str]:
    recs: list[str] = []
    if df.empty:
        recs.append("No data available. Ensure datasets are loaded.")
        return recs
    common_cols = set(c.lower() for c in df.columns)
    if "hour" in common_cols:
        recs.append("Increase patrols and signage during peak-risk hours identified in the chart.")
    if any(x in common_cols for x in ["precipitation", "weather_condition", "weather"]):
        recs.append("Deploy temporary warnings during adverse weather; adjust speed limits dynamically.")
    lat_col, lon_col = infer_lat_lon_columns(df)
    if lat_col and lon_col:
        recs.append("Install speed calming measures and better lighting in heatmap hotspots.")
    recs.append("Run targeted awareness campaigns for recurrent high-risk weekdays.")
    return recs


def sidebar_filters(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    st.sidebar.subheader("Filters")
    min_date = df[datetime_col].min().date() if df[datetime_col].notna().any() else dt.date(2016, 1, 1)
    max_date = df[datetime_col].max().date() if df[datetime_col].notna().any() else dt.date.today()
    start_date, end_date = st.sidebar.date_input("Date range", value=(min_date, max_date))
    if start_date and end_date:
        mask = (df[datetime_col].dt.date >= start_date) & (df[datetime_col].dt.date <= end_date)
        df = df.loc[mask]
    return df


# ---------------- Preprocessing for Model ----------------

def preprocess_for_model(df: pd.DataFrame, model_features: list[str] | None = None) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    if model_features:
        for f in model_features:
            if f not in df.columns:
                df[f] = np.nan
        df = df[model_features]

    # Replace NaN with None for JSON-safe export
    df = df.replace({np.nan: None})
    return df




# ---------------- Main App ----------------

def main() -> None:
    st.set_page_config(page_title="Traffic Accident Risk Dashboard", layout="wide")
    st.title("Traffic Accident Risk Dashboard")
    st.caption("Daily/weekly hazard forecasts, heatmaps, and safety recommendations")

    # Load candidate datasets
    root = os.getcwd()
    candidates = [
        "traffic-accidents-prediction-Ai-application/Main/final_cleaned_dataset.csv",
        "traffic-accidents-prediction-Ai-application/Main/final_resampled_dataset.csv",
        "traffic-accidents-prediction-Ai-application/Main/enriched_accident_data_with_risk_zones.csv",
        "final iteration/final_cleaned_dataset.csv",
        "accidents_Final_cleaned.csv",
        "accidents_Final.csv",
    ]
    dataframes: list[pd.DataFrame] = []
    for path in candidates:
        full = os.path.join(root, path)
        df = load_csv(full)
        if not df.empty:
            dataframes.append(df)

    df = dataframes[0] if dataframes else pd.DataFrame()

    # Infer datetime column
    datetime_candidates = [
        "Start_Time", "start_time", "timestamp", "Date_Time", "datetime", "date_time",
    ]
    df = ensure_datetime(df, datetime_candidates)
    datetime_col = next((c for c in datetime_candidates if c in df.columns), None)
    if datetime_col:
        df = derive_time_columns(df, datetime_col)
        filtered_df = sidebar_filters(df.copy(), datetime_col)
    else:
        filtered_df = df

    lat_col, lon_col = infer_lat_lon_columns(df)

    tab1, tab2, tab3, tab4 = st.tabs(["Forecasts", "Heatmap", "Recommendations", "Model Test"])

    # -------- Forecasts Tab --------
    with tab1:
        st.subheader("Daily/Weekly Hazard Forecasts")
        if df.empty or not datetime_col:
            st.info("Dataset not found or timestamp column missing.")
        else:
            daily, weekly = compute_daily_weekly_hazards(filtered_df, datetime_col)
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("Daily incidents")
                st.line_chart(daily.set_index("date")["incidents"]) if not daily.empty else st.warning("No daily data in range")
            with col_b:
                st.write("Weekly incidents")
                st.bar_chart(weekly.set_index("year_week")["incidents"]) if not weekly.empty else st.warning("No weekly data in range")

    # -------- Heatmap Tab --------
    with tab2:
        st.subheader("High-Risk Areas Heatmap")

        main_dir = os.path.join(os.getcwd(), "traffic-accidents-prediction-Ai-application", "Main")
        prebuilt_candidates = [
            os.path.join(main_dir, "traffic_accident_risk_map_with_zones.html"),
            os.path.join(main_dir, "traffic_accident_risk_map.html"),
        ]
        if os.path.isdir(main_dir):
            for fname in os.listdir(main_dir):
                if fname.lower().startswith("model_risk_map_") and fname.lower().endswith(".html"):
                    prebuilt_candidates.append(os.path.join(main_dir, fname))
        available_prebuilt = [p for p in prebuilt_candidates if os.path.exists(p)]
        mode = "Live heatmap"
        if available_prebuilt:
            mode = st.radio("Map source", ["Live heatmap", "Prebuilt map"], horizontal=True)

        if mode == "Prebuilt map" and available_prebuilt:
            chosen = st.selectbox("Choose prebuilt map", options=available_prebuilt, index=0, format_func=lambda p: os.path.basename(p))
            with open(chosen, "r", encoding="utf-8") as f:
                html = f.read()
            components.html(html, height=640, scrolling=True)
        elif df.empty or not (lat_col and lon_col):
            st.info("Geospatial columns not found. Expected latitude/longitude.")
        else:
            map_obj = make_heatmap(filtered_df[[lat_col, lon_col]].dropna(), lat_col, lon_col)
            if map_obj is None or st_folium is None:
                st.warning("Folium not available. Install dependencies to view the map.")
            else:
                st_folium(map_obj, width=None, height=600)

    # -------- Recommendations Tab --------
    with tab3:
        st.subheader("Safety Recommendations")
        for rec in generate_recommendations(filtered_df):
            st.markdown(f"- {rec}")

    # -------- Model Test Tab --------
    with tab4:
        st.subheader("Model Manual Testing")
        st.write("Upload a CSV with feature columns expected by the backend model, run predictions, and generate a model-based heatmap highlighting high-risk points.")

        api_url2 = st.text_input("API base URL", value=os.getenv("RF_API_URL", "http://localhost:8000"), key="api_url_model_test")

        # Try fetching model info
        try:
            info = requests.get(f"{api_url2}/model-info", timeout=10).json()
            model_features = info.get("feature_names_in", [])
        except Exception as e:
            st.error(f"Failed to fetch model info: {e}")
            model_features = []

        uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
        if uploaded is not None:
            try:
                buf = io.StringIO(uploaded.getvalue().decode("utf-8"))
                df_up = pd.read_csv(buf)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                df_up = pd.DataFrame()

            if not df_up.empty:
                st.write("Preview:")
                st.dataframe(df_up.head(20))

                lat_cand, lon_cand = infer_lat_lon_columns(df_up)
                lat_sel = st.selectbox("Latitude column", options=[lat_cand] + [c for c in df_up.columns if c != lat_cand] if lat_cand else list(df_up.columns), index=0)
                lon_sel = st.selectbox("Longitude column", options=[lon_cand] + [c for c in df_up.columns if c != lon_cand] if lon_cand else list(df_up.columns), index=0)

                if st.button("Run batch prediction"):
                    try:
                        # 🔹 Preprocess uploaded data to match model expectations
                        df_proc = preprocess_for_model(df_up.copy(), model_features)

                        rows = df_proc.replace({np.nan: None}).to_dict(orient="records")
                        resp = requests.post(f"{api_url2}/predict-batch", json={"rows": rows}, timeout=60)
                        out = resp.json()
                        if resp.status_code != 200:
                            st.error(f"API error: {out}")
                        else:
                            preds = out.get("predictions", [])
                            df_pred = df_up.copy()
                            df_pred["model_prediction"] = preds
                            st.success("Predictions completed.")
                            st.dataframe(df_pred.head(50))

                            csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

                            try:
                                if folium is None:
                                    st.warning("Folium not installed; cannot create map.")
                                else:
                                    df_points = df_pred[[lat_sel, lon_sel, "model_prediction"]].copy()
                                    df_points["model_prediction"] = df_points["model_prediction"].astype(str)
                                    pos = df_points[df_points["model_prediction"].isin(["1", "high", "risk", "True", "true"])].dropna(subset=[lat_sel, lon_sel])
                                    if not pos.empty:
                                        center = [pos[lat_sel].astype(float).mean(), pos[lon_sel].astype(float).mean()]
                                        m = folium.Map(location=center, zoom_start=7, tiles="cartodbpositron")
                                        for _, r in pos.iterrows():
                                            folium.CircleMarker([float(r[lat_sel]), float(r[lon_sel])], radius=3, color="#d62728", fill=True, fill_opacity=0.6).add_to(m)
                                        main_dir_path = Path("traffic-accidents-prediction-Ai-application") / "Main"
                                        main_dir_path.mkdir(parents=True, exist_ok=True)
                                        fname = f"model_risk_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                                        out_path = main_dir_path / fname
                                        m.save(str(out_path))
                                        st.success(f"Map saved: {out_path}")
                                        components.html(m._repr_html_(), height=640)
                                    else:
                                        st.info("No high-risk predictions found to plot.")
                            except Exception as e:
                                st.warning(f"Failed to generate map: {e}")

                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")

    # -------- Sidebar --------
    st.sidebar.markdown("---")
    st.sidebar.caption("Data sources loaded from local CSVs if available.")

    st.sidebar.subheader("Model Inference (Backend API)")
    api_url = st.sidebar.text_input("API base URL", value=os.getenv("RF_API_URL", "http://localhost:8000"))
    if st.sidebar.button("Check API"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            st.sidebar.success(str(r.json()))
        except Exception as e:
            st.sidebar.error(f"Health check failed: {e}")

    with st.expander("Try a single prediction"):
        st.write("Provide minimal features expected by the model.")

        feature_kv = st.text_area("JSON features", value="{}", height=120)

        if st.button("Predict"):
            try:
                features_dict = eval(feature_kv) if feature_kv.strip() else {}

                # 🔹 Wrap in DataFrame and preprocess
                df_one = pd.DataFrame([features_dict])
                try:
                    info = requests.get(f"{api_url}/model-info", timeout=10).json()
                    model_features = info.get("feature_names_in", [])
                except Exception:
                    model_features = []

                df_proc = preprocess_for_model(df_one.copy(), model_features)

                # 🔹 Ensure JSON safe
                row = df_proc.replace({np.nan: None}).to_dict(orient="records")[0]

                payload = {"features": row}
                resp = requests.post(f"{api_url}/predict", json=payload, timeout=15)

                st.json(resp.json())
            except Exception as e:
                st.error(f"Prediction call failed: {e}")



if __name__ == "__main__":
    main()
