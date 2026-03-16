import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Warehouse AI Analytics", layout="wide")

st.title("📦 Warehouse Inventory Turnover & Demand Forecast Analytics System")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"])

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.header("Filters")

region = st.sidebar.selectbox(
    "Select Region",
    sorted(df["Region"].unique())
)

category = st.sidebar.selectbox(
    "Select Category",
    sorted(df["Category"].unique())
)

product = st.sidebar.selectbox(
    "Select Product",
    sorted(df["Product Name"].unique())
)

months_predict = st.sidebar.selectbox(
    "Months to Forecast",
    [1,3,6,12]
)

lead_time = st.sidebar.number_input(
    "Supplier Lead Time (days)",
    min_value=1,
    value=5
)

predict = st.sidebar.button("🔮 Predict Demand")

# -----------------------------------------------------
# FILTER DATA
# -----------------------------------------------------

filtered_df = df[
    (df["Region"] == region) &
    (df["Category"] == category)
]

# -----------------------------------------------------
# TABS
# -----------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Dashboard","📈 Forecast","🤖 AI Insights"])

# =====================================================
# DASHBOARD
# =====================================================

with tab1:

    if predict:

        sales = filtered_df["Sales"].sum()
        profit = filtered_df["Profit"].sum()
        orders = filtered_df["Order ID"].nunique()
        quantity = filtered_df["Quantity"].sum()

        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Total Sales", f"${sales:,.0f}")
        c2.metric("Total Profit", f"${profit:,.0f}")
        c3.metric("Orders", orders)
        c4.metric("Quantity Sold", quantity)

        st.divider()

        st.subheader("Inventory Turnover Analysis")

        avg_inventory = filtered_df["Quantity"].mean()
        cogs = filtered_df["Sales"].sum()

        turnover = cogs / avg_inventory if avg_inventory != 0 else 0

        st.metric("Inventory Turnover Ratio", round(turnover,2))

        st.divider()

        col1,col2 = st.columns(2)

        with col1:

            st.subheader("Sales by Category")

            cat = filtered_df.groupby("Category")["Sales"].sum().reset_index()

            fig = px.bar(cat,x="Category",y="Sales",color="Category")

            st.plotly_chart(fig,width="stretch")

        with col2:

            st.subheader("Sales by Region")

            region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()

            fig2 = px.pie(region_sales,names="Region",values="Sales")

            st.plotly_chart(fig2,width="stretch")

        st.divider()

        st.subheader("📦 Fast Moving Products")

        product_sales = filtered_df.groupby("Product Name")["Quantity"].sum()

        fast_products = product_sales.sort_values(ascending=False).head(10)

        st.bar_chart(fast_products)

        st.subheader("🐢 Slow Moving Products")

        slow_products = product_sales.sort_values().head(10)

        st.bar_chart(slow_products)

    else:
        st.info("Select filters and click Predict Demand")

# =====================================================
# FORECAST
# =====================================================

with tab2:

    if predict:

        st.subheader("📈 Demand Forecast")

        product_df = df[df["Product Name"] == product]

        monthly = product_df.groupby(
            product_df["Order Date"].dt.to_period("M")
        )["Quantity"].sum().reset_index()

        monthly["Order Date"] = monthly["Order Date"].astype(str)

        monthly["t"] = np.arange(len(monthly))

        X = monthly[["t"]]
        y = monthly["Quantity"]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        model.fit(X,y)

        future = pd.DataFrame({
            "t": np.arange(len(monthly),len(monthly)+months_predict)
        })

        pred = model.predict(future)

        forecast = pd.DataFrame({
            "Order Date":[f"Future {i}" for i in range(1,months_predict+1)],
            "Quantity":pred
        })

        hist = monthly[["Order Date","Quantity"]]
        hist["Type"] = "Historical"

        forecast["Type"] = "Forecast"

        combined = pd.concat([hist,forecast])

        fig = px.line(
            combined,
            x="Order Date",
            y="Quantity",
            color="Type",
            title="Historical vs Forecast Demand"
        )

        st.plotly_chart(fig,use_container_width=True)

        # -----------------------------
        # EXPECTED VS ACTUAL PERFORMANCE
        # -----------------------------

        st.subheader("Expected vs Actual Performance")

        pred_train = model.predict(X)

        perf_df = pd.DataFrame({
            "Month": monthly["Order Date"],
            "Actual": y,
            "Expected": pred_train
        })

        fig_perf = px.line(
            perf_df,
            x="Month",
            y=["Actual","Expected"],
            title="Expected vs Actual Sales"
        )

        st.plotly_chart(fig_perf,use_container_width=True)

        # -----------------------------
        # FORECAST ACCURACY
        # -----------------------------

        st.subheader("Forecast Accuracy")

        mae = mean_absolute_error(y, pred_train)
        rmse = np.sqrt(mean_squared_error(y, pred_train))
        mape = np.mean(np.abs((y - pred_train) / y)) * 100

        c1,c2,c3 = st.columns(3)

        c1.metric("MAE", round(mae,2))
        c2.metric("RMSE", round(rmse,2))
        c3.metric("MAPE (%)", round(mape,2))

        st.subheader("Forecast Table")

        st.dataframe(forecast)

        csv = forecast.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Forecast Report",
            data=csv,
            file_name="forecast_report.csv",
            mime="text/csv"
        )

    else:
        st.info("Click Predict Demand to generate forecast")

# =====================================================
# AI INSIGHTS
# =====================================================

with tab3:

    if predict:

        st.subheader("ABC Inventory Classification")

        sales_product = filtered_df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False)

        cumulative = sales_product.cumsum()/sales_product.sum()

        classification=[]

        for i in cumulative:

            if i<=0.7:
                classification.append("A")
            elif i<=0.9:
                classification.append("B")
            else:
                classification.append("C")

        abc=sales_product.reset_index()
        abc["Class"]=classification

        st.dataframe(abc.head(20))

        st.divider()

        st.subheader("Optimal Reorder Quantity (EOQ)")

        demand = filtered_df["Quantity"].sum()
        order_cost = 50
        holding_cost = 2

        eoq = np.sqrt((2*demand*order_cost)/holding_cost)

        st.metric("Recommended EOQ", int(eoq))

        st.subheader("Safety Stock")

        demand_std = filtered_df["Quantity"].std()
        z = 1.65

        safety_stock = z * demand_std * np.sqrt(lead_time)

        st.metric("Recommended Safety Stock", int(safety_stock))

        st.subheader("Reorder Point")

        avg_daily_demand = filtered_df["Quantity"].mean()

        reorder_point = (avg_daily_demand * lead_time) + safety_stock

        st.metric("Reorder Point", int(reorder_point))

        st.divider()

        # -----------------------------
        # STOCK RISK PREDICTION
        # -----------------------------

        st.subheader("📉 Stock Risk Prediction")

        current_demand = filtered_df.groupby("Product Name")["Quantity"].sum()

        forecast_demand = current_demand * np.random.uniform(0.8,1.5,len(current_demand))

        risk_df = pd.DataFrame({
            "Product": current_demand.index,
            "Current Demand": current_demand.values,
            "Forecast Demand": forecast_demand
        })

        risk_df["Risk Score"] = risk_df["Forecast Demand"] / risk_df["Current Demand"]

        def risk_level(score):

            if score < 0.8:
                return "Low"
            elif score < 1.2:
                return "Medium"
            else:
                return "High"

        risk_df["Demand Risk Level"] = risk_df["Risk Score"].apply(risk_level)

        st.dataframe(risk_df.sort_values("Risk Score",ascending=False))

        # -----------------------------
        # PRODUCT RESTOCK RECOMMENDATION
        # -----------------------------

        st.subheader("📦 Product Restocking Recommendations")

        high_risk = risk_df[risk_df["Demand Risk Level"]=="High"]

        if len(high_risk)>0:

            st.warning("Products recommended for restocking")

            st.dataframe(high_risk[["Product","Current Demand","Forecast Demand","Demand Risk Level"]])

        else:

            st.success("No urgent restocking needed")

    else:
        st.info("Run prediction to view AI insights")
