from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Next Bill Prediction", page_icon="⚡", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "billings.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["billing_month"] = pd.to_datetime(df["billing_month"])
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df.sort_values(["customer_id", "billing_month"]).reset_index(drop=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
bundle = load_model()
model = bundle["model"]
features = bundle["features"]
threshold_multiplier = float(bundle.get("business_high_bill_threshold", 2.5))

st.sidebar.title("⚡ Bill Analytics")
page = st.sidebar.radio("Dashboard", ["📊 Dataset & EDA", "🔮 Bill Prediction"])
st.sidebar.caption("One CSV • One model • Two Streamlit pages")

if page == "📊 Dataset & EDA":
    st.title("📊 Dataset Overview & Exploratory Data Analysis")
    st.caption("A complete visual summary of the billing dataset used by the prediction model.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Customers", f"{df['customer_id'].nunique():,}")
    c3.metric("Avg Bill", f"₹{df['bill_amount'].mean():,.2f}")
    c4.metric("Avg Usage", f"{df['usage_units'].mean():,.1f}")
    c5.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Billing Trends", "Customer Analysis", "Relationships"])

    with tab1:
        left, right = st.columns([1.7, 1])
        with left:
            st.subheader("Dataset preview")
            st.dataframe(df.head(100), use_container_width=True, height=430)
        with right:
            st.subheader("Data quality")
            quality = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing": df.isna().sum().values,
                "Unique": [df[c].nunique() for c in df.columns],
            })
            st.dataframe(quality, use_container_width=True, height=430, hide_index=True)

        st.subheader("Key EDA Visualizations")

        # Graph 1: Monthly average bill trend
        monthly_eda = (
            df.groupby("billing_month", as_index=False)["bill_amount"]
            .mean()
            .rename(columns={"bill_amount": "average_bill"})
        )

        # Graph 2: Average bill by plan type
        plan_eda = (
            df.groupby("plan_type", as_index=False)["bill_amount"]
            .mean()
            .sort_values("bill_amount", ascending=False)
        )

        # Use a sample for the scatter plot so the dashboard stays fast
        sample_eda = df.sample(min(4000, len(df)), random_state=42)

        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            fig = px.line(
                monthly_eda,
                x="billing_month",
                y="average_bill",
                markers=True,
                title="Monthly Average Bill Trend"
            )
            fig.update_layout(
                xaxis_title="Billing Month",
                yaxis_title="Average Bill Amount"
            )
            st.plotly_chart(fig, use_container_width=True)

        with row1_col2:
            fig = px.histogram(
                df,
                x="bill_amount",
                nbins=40,
                title="Bill Amount Distribution"
            )
            fig.update_layout(
                xaxis_title="Bill Amount",
                yaxis_title="Number of Records"
            )
            st.plotly_chart(fig, use_container_width=True)

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            fig = px.bar(
                plan_eda,
                x="plan_type",
                y="bill_amount",
                text_auto=".2f",
                title="Average Bill by Plan Type"
            )
            fig.update_layout(
                xaxis_title="Plan Type",
                yaxis_title="Average Bill Amount"
            )
            st.plotly_chart(fig, use_container_width=True)

        with row2_col2:
            fig = px.scatter(
                sample_eda,
                x="usage_units",
                y="bill_amount",
                color="plan_type",
                opacity=0.55,
                title="Usage vs Bill Amount"
            )
            fig.update_layout(
                xaxis_title="Usage Units",
                yaxis_title="Bill Amount"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Categorical summary")
        cats = ["plan_type", "customer_segment", "region", "payment_status"]
        cols = st.columns(4)
        for i, col in enumerate(cats):
            temp = df[col].value_counts().reset_index()
            temp.columns = [col, "count"]
            fig = px.bar(temp, x=col, y="count", title=col.replace("_", " ").title())
            cols[i].plotly_chart(fig, use_container_width=True)

    with tab2:
        monthly = df.groupby("billing_month", as_index=False).agg(
            average_bill=("bill_amount", "mean"),
            median_bill=("bill_amount", "median"),
            average_usage=("usage_units", "mean")
        )
        fig = px.line(monthly, x="billing_month", y=["average_bill", "median_bill"], markers=True,
                      title="Average and Median Bill by Month")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="bill_amount", nbins=50, marginal="box", title="Bill Amount Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df, x="plan_type", y="bill_amount", points=False, title="Bill Amount by Plan Type")
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            plan_month = df.groupby(["billing_month", "plan_type"], as_index=False)["bill_amount"].mean()
            fig = px.line(plan_month, x="billing_month", y="bill_amount", color="plan_type", markers=True,
                          title="Monthly Average Bill by Plan")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            region_bill = df.groupby("region", as_index=False)["bill_amount"].mean().sort_values("bill_amount", ascending=False)
            fig = px.bar(region_bill, x="region", y="bill_amount", title="Average Bill by Region")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            seg = df.groupby("customer_segment", as_index=False).agg(
                average_bill=("bill_amount", "mean"), average_usage=("usage_units", "mean"), customers=("customer_id", "nunique")
            )
            fig = px.bar(seg, x="customer_segment", y="average_bill", text_auto=".2f",
                         title="Average Bill by Customer Segment")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            plan = df.groupby("plan_type", as_index=False).agg(
                average_bill=("bill_amount", "mean"), average_usage=("usage_units", "mean"), customers=("customer_id", "nunique")
            )
            fig = px.bar(plan, x="plan_type", y="average_bill", text_auto=".2f", title="Average Bill by Plan")
            st.plotly_chart(fig, use_container_width=True)

        customer_summary = df.groupby("customer_id", as_index=False).agg(
            average_bill=("bill_amount", "mean"),
            maximum_bill=("bill_amount", "max"),
            average_usage=("usage_units", "mean"),
            plan_type=("plan_type", "last"),
            region=("region", "last")
        )
        st.subheader("Highest-bill customers")
        st.dataframe(customer_summary.sort_values("maximum_bill", ascending=False).head(20), use_container_width=True, hide_index=True)

    with tab4:
        sample = df.sample(min(4000, len(df)), random_state=42)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(sample, x="usage_units", y="bill_amount", color="plan_type", opacity=0.55,
                             title="Usage vs Bill Amount")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(sample, x="temperature_avg", y="usage_units", color="customer_segment", opacity=0.55,
                             title="Temperature vs Usage")
            st.plotly_chart(fig, use_container_width=True)

        numeric_cols = ["bill_amount", "usage_units", "peak_usage_share", "unit_price", "fixed_charge",
                        "discount_amount", "tax_amount", "one_off_charges", "late_fee", "support_tickets",
                        "household_size", "temperature_avg"]
        corr = df[numeric_cols].corr().round(2)
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                        colorscale="RdBu", zmid=0, text=corr.values, texttemplate="%{text}"))
        fig.update_layout(title="Correlation Heatmap", height=700)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.title("🔮 Next Bill & High-Bill Prediction")
    st.caption("Select a customer, review the latest values, adjust the inputs if needed, and predict the next bill.")

    customer_id = st.selectbox("Customer ID", sorted(df["customer_id"].unique()))
    hist = df[df["customer_id"] == customer_id].sort_values("billing_month").copy()
    latest = hist.iloc[-1]
    recent = hist.tail(7).sort_values("billing_month", ascending=False).reset_index(drop=True)

    if len(recent) < 7:
        st.error("This customer needs at least 7 months of history for prediction.")
        st.stop()

    st.subheader("Recent customer history")
    history_view = hist.tail(7)[["billing_month", "bill_amount", "usage_units", "plan_type", "region"]].copy()
    history_view["billing_month"] = history_view["billing_month"].dt.strftime("%Y-%m")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.dataframe(history_view.sort_values("billing_month", ascending=False), use_container_width=True, hide_index=True)
    with c2:
        chart_df = hist.tail(7).set_index("billing_month")[["bill_amount"]]
        st.line_chart(chart_df, use_container_width=True)

    st.subheader("Prediction inputs")
    st.caption("Values are pre-filled from the customer's latest record. Change them to test another scenario.")

    a, b, c, d = st.columns(4)
    current_bill = a.number_input("Current bill (₹)", min_value=0.0, value=float(latest["bill_amount"]), step=10.0)
    current_usage = b.number_input("Current usage units", min_value=0.0, value=float(latest["usage_units"]), step=1.0)
    peak_usage = c.slider("Peak usage share", 0.0, 1.0, float(latest["peak_usage_share"]), 0.01)
    temperature = d.number_input("Average temperature", value=float(latest["temperature_avg"]), step=0.5)

    a, b, c, d = st.columns(4)
    unit_price = a.number_input("Unit price", min_value=0.0, value=float(latest["unit_price"]), step=0.1)
    fixed_charge = b.number_input("Fixed charge", min_value=0.0, value=float(latest["fixed_charge"]), step=10.0)
    support_tickets = c.number_input("Support tickets", min_value=0, max_value=20, value=int(latest["support_tickets"]), step=1)
    household_size = d.number_input("Household size", min_value=1, max_value=20, value=int(latest["household_size"]), step=1)

    opts = bundle.get("categorical_options", {})
    a, b, c, d = st.columns(4)
    plan_values = opts.get("plan_type", sorted(df.plan_type.unique().tolist()))
    seg_values = opts.get("customer_segment", sorted(df.customer_segment.unique().tolist()))
    region_values = opts.get("region", sorted(df.region.unique().tolist()))
    plan_type = a.selectbox("Plan type", plan_values, index=plan_values.index(latest["plan_type"]) if latest["plan_type"] in plan_values else 0)
    segment = b.selectbox("Customer segment", seg_values, index=seg_values.index(latest["customer_segment"]) if latest["customer_segment"] in seg_values else 0)
    region = c.selectbox("Region", region_values, index=region_values.index(latest["region"]) if latest["region"] in region_values else 0)
    tariff_change = d.selectbox("Tariff changed?", [0, 1], index=int(latest["tariff_change_flag"]))

    if st.button("Predict Next Bill", type="primary", use_container_width=True):
        bills = recent["bill_amount"].astype(float).tolist()
        usage = recent["usage_units"].astype(float).tolist()
        bills[0] = float(current_bill)
        usage[0] = float(current_usage)

        current_month = pd.Timestamp(latest["billing_month"])
        signup_date = pd.Timestamp(latest["signup_date"])
        month = current_month.month
        tenure = max(0, (current_month.year - signup_date.year) * 12 + current_month.month - signup_date.month)

        bill_mean_3 = float(np.mean(bills[:3]))
        bill_mean_6 = float(np.mean(bills[:6]))
        usage_mean_3 = float(np.mean(usage[:3]))
        usage_mean_6 = float(np.mean(usage[:6]))

        row = {
            "bill_amount": float(current_bill),
            "usage_units": float(current_usage),
            "peak_usage_share": float(peak_usage),
            "unit_price": float(unit_price),
            "fixed_charge": float(fixed_charge),
            "tariff_change_flag": int(tariff_change),
            "support_tickets": int(support_tickets),
            "household_size": int(household_size),
            "temperature_avg": float(temperature),
            "month": month,
            "sin_month": np.sin(2 * np.pi * month / 12),
            "cos_month": np.cos(2 * np.pi * month / 12),
            "tenure_months": tenure,
            "bill_lag_1": bills[1], "bill_lag_2": bills[2], "bill_lag_3": bills[3], "bill_lag_6": bills[6],
            "usage_lag_1": usage[1], "usage_lag_2": usage[2], "usage_lag_3": usage[3], "usage_lag_6": usage[6],
            "bill_roll_mean_3": bill_mean_3,
            "bill_roll_mean_6": bill_mean_6,
            "bill_roll_std_3": float(np.std(bills[:3], ddof=1)),
            "bill_roll_std_6": float(np.std(bills[:6], ddof=1)),
            "usage_roll_mean_3": usage_mean_3,
            "usage_roll_mean_6": usage_mean_6,
            "usage_roll_std_3": float(np.std(usage[:3], ddof=1)),
            "usage_roll_std_6": float(np.std(usage[:6], ddof=1)),
            "bill_vs_roll6": float(current_bill / bill_mean_6) if bill_mean_6 else 1.0,
            "usage_vs_roll6": float(current_usage / usage_mean_6) if usage_mean_6 else 1.0,
            "plan_type": plan_type,
            "customer_segment": segment,
            "region": region,
        }

        X = pd.DataFrame([row])[features]
        prediction = max(0.0, float(model.predict(X)[0]))
        baseline = bill_mean_6
        high_bill_limit = threshold_multiplier * baseline
        ratio = prediction / baseline if baseline else 0.0
        is_high = prediction > high_bill_limit

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Next Bill", f"₹{prediction:,.2f}")
        m2.metric("6-Month Average", f"₹{baseline:,.2f}")
        m3.metric("2.5× High-Bill Limit", f"₹{high_bill_limit:,.2f}")
        m4.metric("Prediction / Baseline", f"{ratio:.2f}×")

        if is_high:
            st.error("🚨 HIGH BILL ALERT — the predicted bill is above 2.5× the customer's recent 6-month average.")
        else:
            st.success("✅ NORMAL BILL — the predicted bill is below the 2.5× high-bill threshold.")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            number={"prefix": "₹", "valueformat": ",.0f"},
            delta={"reference": baseline, "relative": True},
            gauge={
                "axis": {"range": [0, max(high_bill_limit * 1.2, prediction * 1.2, 1)]},
                "threshold": {"line": {"width": 4}, "thickness": 0.8, "value": high_bill_limit},
                "steps": [
                    {"range": [0, high_bill_limit], "color": "rgba(0,180,120,0.15)"},
                    {"range": [high_bill_limit, max(high_bill_limit * 1.2, prediction * 1.2, 1)], "color": "rgba(220,50,50,0.15)"},
                ],
            },
            title={"text": "Predicted bill vs 2.5× threshold"}
        ))
        gauge.update_layout(height=360)
        st.plotly_chart(gauge, use_container_width=True)

        comparison = pd.DataFrame({
            "Value": ["6-month average", "Predicted next bill", "2.5× threshold"],
            "Amount": [baseline, prediction, high_bill_limit]
        })
        fig = px.bar(comparison, x="Value", y="Amount", text_auto=".2f", title="Prediction comparison")
        st.plotly_chart(fig, use_container_width=True)
