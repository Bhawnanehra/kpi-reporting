import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Sales & KPI Dashboard", layout="wide")


@st.cache_data
def load_data(path="sales_data.csv"):
    df = pd.read_csv(path)
    # normalize column names (trim)
    df.columns = [c.strip() for c in df.columns]
    # parse dates
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    return df


df = load_data("sales_data.csv")

# quick column checks
required = ["Order ID", "Order Date", "Customer ID", "Sales"]
for c in required:
    if c not in df.columns:
        st.error(
            f"Required column missing: {c}. Please ensure your CSV has this column."
        )
        st.stop()

df = df.dropna(subset=["Order Date"])
df["Order_Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
df["Order_Year"] = df["Order Date"].dt.year

st.sidebar.header("Filters & Settings")
years = sorted(df["Order_Year"].unique())
sel_years = st.sidebar.multiselect("Year", years, default=years)
regions = sorted(df["Region"].dropna().unique()) if "Region" in df.columns else []
sel_regions = st.sidebar.multiselect("Region", regions, default=regions)
category_filter = None
if "Category" in df.columns:
    categories = sorted(df["Category"].dropna().unique())
    category_filter = st.sidebar.multiselect("Category", categories, default=categories)

# margin handling
has_cost = "Cost" in df.columns
if has_cost:
    st.sidebar.write("Cost column detected — exact margin will be used.")
else:
    assumed_margin_pct = st.sidebar.slider(
        "Assume gross margin % (if Cost not provided)", 0.0, 100.0, 30.0
    )

# apply filters
mask = df["Order_Year"].isin(sel_years)
if sel_regions:
    mask &= df["Region"].isin(sel_regions)
if category_filter is not None:
    mask &= df["Category"].isin(category_filter)
dff = df[mask].copy()

dff["Revenue"] = dff["Sales"].astype(float)

if has_cost:
    dff["Margin"] = dff["Revenue"] - dff["Cost"].astype(float)
else:
    dff["Margin"] = dff["Revenue"] * (assumed_margin_pct / 100.0)

total_revenue = dff["Revenue"].sum()
total_margin = dff["Margin"].sum()
unique_orders = dff["Order ID"].nunique()
aov = total_revenue / unique_orders if unique_orders else 0.0

# retention: % of customers with >1 distinct orders in the filtered window
cust_orders = dff.groupby("Customer ID")["Order ID"].nunique()
total_customers = cust_orders.shape[0]
returning_customers = (cust_orders > 1).sum()
retention_rate = (
    (returning_customers / total_customers * 100) if total_customers else 0.0
)

st.title("Sales & KPI Dashboard")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", f"$ {total_revenue:,.2f}")
k2.metric("Total Margin (est.)", f"$ {total_margin:,.2f}")
k3.metric("AOV (Average Order Value)", f"$ {aov:,.2f}")
k4.metric("Retention (customers w/ >1 order)", f"{retention_rate:.2f}%")

st.markdown("---")

st.subheader("Revenue Growth (monthly)")
rev_ts = dff.groupby("Order_Month", as_index=False).agg({"Revenue": "sum"})
fig_rev = px.line(
    rev_ts,
    x="Order_Month",
    y="Revenue",
    markers=True,
    labels={"Order_Month": "Month", "Revenue": "Revenue"},
    title="Monthly Revenue",
)
st.plotly_chart(fig_rev, use_container_width=True)

cols = st.columns(2)
with cols[0]:
    if "Region" in dff.columns:
        st.subheader("Revenue by Region")
        by_region = (
            dff.groupby("Region", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=False)
        )
        fig_r = px.bar(by_region, x="Region", y="Revenue", title="Revenue by Region")
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.info("No 'Region' column found for regional breakdown.")

with cols[1]:
    if "Category" in dff.columns:
        st.subheader("Revenue by Category")
        by_cat = (
            dff.groupby("Category", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=False)
        )
        fig_c = px.bar(by_cat, x="Category", y="Revenue", title="Revenue by Category")
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("No 'Category' column found for category breakdown.")

if "Product Name" in dff.columns:
    st.subheader("Top Products by Revenue")
    top_products = (
        dff.groupby("Product Name", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )
    fig_p = px.bar(
        top_products,
        x="Revenue",
        y="Product Name",
        orientation="h",
        title="Top 10 Products",
    )
    st.plotly_chart(fig_p, use_container_width=True)

st.subheader("AOV (Average Order Value) Over Time")
aov_ts = (
    dff.groupby("Order_Month")
    .agg({"Revenue": "sum", "Order ID": "nunique"})
    .reset_index()
)
aov_ts["AOV"] = aov_ts["Revenue"] / aov_ts["Order ID"]
fig_aov = px.line(aov_ts, x="Order_Month", y="AOV", markers=True, title="AOV by Month")
st.plotly_chart(fig_aov, use_container_width=True)

st.subheader("Customer Retention Snapshot")
ret_df = pd.DataFrame(
    {
        "Type": ["Returning Customers (>1 order)", "One-time Customers"],
        "Count": [returning_customers, total_customers - returning_customers],
    }
)
fig_ret = px.pie(
    ret_df, names="Type", values="Count", title="Returning vs One-time Customers"
)
st.plotly_chart(fig_ret, use_container_width=True)


def cohort_analysis(df_orders):
    dfc = df_orders.copy()
    # ensure order date exists and normalized to month period start
    dfc["OrderPeriod"] = (
        pd.to_datetime(dfc["Order Date"]).dt.to_period("M").dt.to_timestamp()
    )
    # cohort month = customer's first order month
    dfc["CohortMonth"] = dfc.groupby("Customer ID")["OrderPeriod"].transform("min")

    # months since cohort
    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + (d1.month - d2.month)

    dfc["CohortIndex"] = dfc.apply(
        lambda row: diff_month(row["OrderPeriod"], row["CohortMonth"]), axis=1
    )

    # unique customers per cohort-month / cohort-index
    cohort_data = (
        dfc.groupby(["CohortMonth", "CohortIndex"])["Customer ID"]
        .nunique()
        .reset_index(name="Customers")
    )

    # pivot to counts matrix (rows = cohort month, cols = months since cohort)
    cohort_counts = (
        cohort_data.pivot(
            index="CohortMonth", columns="CohortIndex", values="Customers"
        )
        .fillna(0)
        .astype(int)
    )

    # cohort size = month 0 counts (first column); handle if no column 0
    if 0 in cohort_counts.columns:
        cohort_size = cohort_counts[0]
    else:
        # fallback: compute sizes from cohort_data where CohortIndex==0
        cohort_size = cohort_data[cohort_data["CohortIndex"] == 0].set_index(
            "CohortMonth"
        )["Customers"]

    # retention rates (divide each row by cohort_size)
    cohort_rates = cohort_counts.div(cohort_size, axis=0).round(3)

    return cohort_counts, cohort_rates


st.subheader("Cohort Retention (Monthly)")

try:
    counts_matrix, rates_matrix = cohort_analysis(dff)

    if counts_matrix.shape[0] == 0:
        st.info("Not enough data for cohort analysis in filtered selection.")
    else:
        # choose view
        view = st.radio("Display", ("Retention %", "Absolute counts"), horizontal=True)

        # prepare axis labels safely (convert index to YYYY-MM)
        def pretty_index_labels(idx):
            try:
                return pd.to_datetime(idx).strftime("%Y-%m")
            except Exception:
                return idx.astype(str)

        y_labels = pretty_index_labels(counts_matrix.index)
        x_labels = counts_matrix.columns.astype(str)

        if view == "Retention %":
            plot_matrix = rates_matrix
            title = "Cohort Retention (%)"
            # values are floats 0..1; show as percent in tooltip via hovertemplate if desired
            z = (plot_matrix.values * 100).round(
                2
            )  # convert to percent for readability
            colorbar_title = "Retention (%)"
        else:
            plot_matrix = counts_matrix
            title = "Cohort Customers (Absolute Counts)"
            z = plot_matrix.values
            colorbar_title = "Customers"

        # draw heatmap
        fig_cohort = px.imshow(
            z,
            x=x_labels,
            y=y_labels,
            aspect="auto",
            labels=dict(
                x="Months Since Cohort", y="Cohort Month", color=colorbar_title
            ),
            title=title,
        )
        # improve hover info
        fig_cohort.update_traces(
            hovertemplate="Cohort=%{y}<br>Months since cohort=%{x}<br>Value=%{z}<extra></extra>"
        )

        st.plotly_chart(fig_cohort, use_container_width=True)

        # prepare exportable dataframe (index as YYYY-MM strings)
        export_df = plot_matrix.copy()
        try:
            export_df.index = pd.to_datetime(export_df.index).strftime("%Y-%m")
        except Exception:
            export_df.index = export_df.index.astype(str)

        # show small table and provide download
        with st.expander("Show cohort matrix (preview)"):
            st.dataframe(export_df)

        csv_bytes = export_df.to_csv().encode("utf-8")
        st.download_button(
            label="Download cohort matrix CSV",
            data=csv_bytes,
            file_name="cohort_matrix.csv",
            mime="text/csv",
        )

except Exception as e:
    st.error(f"Cohort calculation failed: {e}")


st.markdown("---")
with st.expander("Show sample rows from filtered dataset"):
    st.dataframe(dff.head(200))
