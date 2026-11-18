
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Campus Energy Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Campus Energy Analytics Dashboard")
st.caption(
    "Machine Learning Based Electricity Consumption "
    "Prediction and Energy Optimization System"
)

ZIP_PATH = "/content/archive (1).zip"
)

print("Dataset path:", ZIP_PATH)
print("File exists:", os.path.exists(ZIP_PATH))

# ---------------------------------
# LOAD DATASET
# ---------------------------------

@st.cache_data(show_spinner="Loading electricity dataset...")
def load_data():

    with zipfile.ZipFile(ZIP_PATH) as z:

        campus = pd.read_csv(
            z.open("campus_meta.csv")
        )

        chunks = []

        with z.open("building_consumption.csv") as f:

            for chunk in pd.read_csv(
                f,
                chunksize=500000
            ):

                chunk["timestamp"] = pd.to_datetime(
                    chunk["timestamp"],
                    errors="coerce"
                )

                chunk = chunk.dropna(
                    subset=["timestamp", "consumption"]
                )

                chunk = chunk[
                    chunk["consumption"] >= 0
                ]

                chunk["timestamp"] = (
                    chunk["timestamp"].dt.floor("h")
                )

                grouped = chunk.groupby(
                    ["campus_id", "timestamp"]
                )["consumption"].sum().reset_index()

                chunks.append(grouped)

    df = pd.concat(chunks, ignore_index=True)

    df = df.groupby(
        ["campus_id", "timestamp"]
    )["consumption"].sum().reset_index()

    return df, campus


try:
    df, campus = load_data()

except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.header("Dashboard Settings")

campus_options = dict(
    zip(campus["name"], campus["id"])
)

selected_name = st.sidebar.selectbox(
    "Select Campus",
    list(campus_options.keys())
)

selected_id = campus_options[selected_name]

cost_per_unit = st.sidebar.number_input(
    "Electricity cost per unit (₹)",
    min_value=0.0,
    value=8.0,
    step=0.5
)

saving_percent = st.sidebar.slider(
    "Target energy saving (%)",
    min_value=5,
    max_value=30,
    value=10
)

data = df[
    df["campus_id"] == selected_id
].copy()

data = data.sort_values("timestamp")

if data.empty:
    st.warning("No data available for this campus.")
    st.stop()

# ---------------------------------
# DATE FILTER
# ---------------------------------

st.sidebar.subheader("Date Filter")

min_date = data["timestamp"].min().date()
max_date = data["timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:

    start_date, end_date = date_range

    filtered = data[
        (data["timestamp"].dt.date >= start_date) &
        (data["timestamp"].dt.date <= end_date)
    ].copy()

else:
    filtered = data.copy()

if filtered.empty:
    st.warning("No records found for the selected date range.")
    st.stop()

# ---------------------------------
# KPI CARDS
# ---------------------------------

total_energy = filtered["consumption"].sum()

average_energy = filtered["consumption"].mean()

peak_energy = filtered["consumption"].max()

peak_time = filtered.loc[
    filtered["consumption"].idxmax(),
    "timestamp"
]

estimated_cost = total_energy * cost_per_unit

potential_saving = (
    total_energy * saving_percent / 100
)

estimated_saving = (
    potential_saving * cost_per_unit
)

st.subheader("Energy Consumption Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Consumption",
    f"{total_energy:,.2f}"
)

c2.metric(
    "Average Hourly Usage",
    f"{average_energy:,.2f}"
)

c3.metric(
    "Peak Consumption",
    f"{peak_energy:,.2f}"
)

c4.metric(
    "Estimated Electricity Cost",
    f"₹{estimated_cost:,.2f}"
)

st.caption(
    f"Highest usage recorded at: {peak_time}"
)

st.divider()

# ---------------------------------
# CONSUMPTION TREND
# ---------------------------------

st.subheader("📈 Electricity Consumption Trend")

fig = px.line(
    filtered,
    x="timestamp",
    y="consumption",
    title="Hourly Electricity Consumption",
    labels={
        "timestamp": "Date and Time",
        "consumption": "Consumption (dataset units)"
    }
)

fig.update_layout(
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# DAILY CONSUMPTION
# ---------------------------------

st.subheader("📅 Daily Energy Consumption")

daily = filtered.copy()

daily["date"] = daily["timestamp"].dt.date

daily_data = daily.groupby(
    "date"
)["consumption"].sum().reset_index()

fig2 = px.bar(
    daily_data,
    x="date",
    y="consumption",
    title="Daily Electricity Usage",
    labels={
        "date": "Date",
        "consumption": "Consumption"
    }
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------
# HOURLY USAGE ANALYSIS
# ---------------------------------

st.subheader("⏰ Peak Usage Time Analysis")

filtered["hour"] = filtered["timestamp"].dt.hour

hourly = filtered.groupby(
    "hour"
)["consumption"].mean().reset_index()

fig3 = px.bar(
    hourly,
    x="hour",
    y="consumption",
    title="Average Consumption by Hour",
    labels={
        "hour": "Hour of Day",
        "consumption": "Average Consumption"
    }
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------
# MACHINE LEARNING MODEL
# ---------------------------------

st.divider()

st.subheader("🤖 Machine Learning Prediction")

st.write(
    "Random Forest Regressor is used to predict "
    "the next 24 hours of electricity consumption."
)

# Use complete campus history for model training
model_data = data[
    ["timestamp", "consumption"]
].copy()

model_data = model_data.groupby(
    "timestamp"
)["consumption"].sum().reset_index()

model_data = (
    model_data.set_index("timestamp")
    .asfreq("h")
)

model_data["consumption"] = (
    model_data["consumption"].fillna(0)
)

model_data = model_data.reset_index()

# Feature engineering
def make_features(d):

    d = d.copy()

    d["hour"] = d["timestamp"].dt.hour
    d["day"] = d["timestamp"].dt.day
    d["month"] = d["timestamp"].dt.month

    d["day_of_week"] = (
        d["timestamp"].dt.dayofweek
    )

    d["is_weekend"] = (
        d["day_of_week"] >= 5
    ).astype(int)

    d["lag_1"] = d["consumption"].shift(1)
    d["lag_24"] = d["consumption"].shift(24)
    d["lag_168"] = d["consumption"].shift(168)

    d["rolling_mean_24"] = (
        d["consumption"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    return d

model_data = make_features(model_data).dropna()

features = [
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24"
]

if len(model_data) < 200:
    st.warning(
        "Not enough historical records to train the model."
    )
    st.stop()

X = model_data[features]
y = model_data["consumption"]

split = int(len(model_data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

@st.cache_resource
def train_model(X_train, y_train):

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    return rf

model = train_model(X_train, y_train)

# Model evaluation
test_predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    test_predictions
)

rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

r2 = r2_score(
    y_test,
    test_predictions
)

m1, m2, m3 = st.columns(3)

m1.metric("MAE", f"{mae:.3f}")
m2.metric("RMSE", f"{rmse:.3f}")
m3.metric("R² Score", f"{r2:.3f}")

# ---------------------------------
# NEXT 24 HOURS FORECAST
# ---------------------------------

history = model_data[
    ["timestamp", "consumption"]
].copy()

forecast = []

for i in range(24):

    next_time = (
        history["timestamp"].iloc[-1]
        + pd.Timedelta(hours=1)
    )

    next_row = pd.DataFrame({
        "timestamp": [next_time],
        "consumption": [np.nan]
    })

    combined = pd.concat(
        [history, next_row],
        ignore_index=True
    )

    temp = make_features(combined)

    prediction = model.predict(
        temp[features].iloc[[-1]]
    )[0]

    prediction = max(0, prediction)

    forecast.append({
        "timestamp": next_time,
        "predicted_consumption": prediction
    })

    history = pd.concat([
        history,
        pd.DataFrame({
            "timestamp": [next_time],
            "consumption": [prediction]
        })
    ], ignore_index=True)

forecast_df = pd.DataFrame(forecast)

st.subheader("🔮 Next 24 Hours Consumption Forecast")

fig4 = px.line(
    forecast_df,
    x="timestamp",
    y="predicted_consumption",
    markers=True,
    title="Predicted Electricity Consumption"
)

st.plotly_chart(fig4, use_container_width=True)

st.metric(
    "Predicted Next 24 Hours Consumption",
    f"{forecast_df['predicted_consumption'].sum():,.2f}"
)

# ---------------------------------
# ENERGY OPTIMIZATION
# ---------------------------------

st.divider()

st.subheader("💡 Smart Energy Optimization Recommendations")

peak_hours = (
    hourly.sort_values(
        "consumption",
        ascending=False
    ).head(3)
)

low_hours = (
    hourly.sort_values(
        "consumption"
    ).head(3)
)

st.write("### High Energy Usage Periods")

for _, row in peak_hours.iterrows():

    st.warning(
        f"High average usage at {int(row['hour']):02d}:00"
    )

st.write("### Recommended Low Usage Periods")

for _, row in low_hours.iterrows():

    st.success(
        f"Consider scheduling flexible electricity usage "
        f"around {int(row['hour']):02d}:00, "
        f"subject to campus operating requirements."
    )

st.write("### Estimated Energy Saving Opportunity")

st.metric(
    "Potential Energy Saving",
    f"{potential_saving:,.2f} dataset units"
)

st.metric(
    "Potential Cost Saving",
    f"₹{estimated_saving:,.2f}"
)

st.caption(
    "Savings are scenario estimates based on your selected "
    "saving percentage and assumed electricity tariff. "
    "They are not measured or guaranteed savings."
)

# ---------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------

st.divider()

st.subheader("📊 Actual vs Predicted Consumption")

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": test_predictions
})

comparison["Index"] = range(len(comparison))

comparison_long = comparison.melt(
    id_vars="Index",
    value_vars=["Actual", "Predicted"],
    var_name="Type",
    value_name="Consumption"
)

fig5 = px.line(
    comparison_long,
    x="Index",
    y="Consumption",
    color="Type",
    title="Model Performance Comparison"
)

st.plotly_chart(fig5, use_container_width=True)

# ---------------------------------
# DOWNLOAD REPORT
# ---------------------------------

st.subheader("📥 Download Energy Report")

csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Consumption CSV",
    data=csv,
    file_name="campus_energy_report.csv",
    mime="text/csv"
)

st.success("Campus Energy Dashboard Loaded Successfully!")
