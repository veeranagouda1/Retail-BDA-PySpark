import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Store Big Data Analytics",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Retail Store Big Data Analytics")
st.markdown(
    "A retail sales analytics dashboard using **Apache Spark (PySpark)**, "
    "Streamlit, Pandas, and Plotly."
)

REQUIRED_COLUMNS = [
    "Date", "Product", "Category", "Store", "Customer",
    "Quantity", "Price", "Payment", "Sales"
]

TEXT_COLUMNS = [
    "Product", "Category", "Store", "Customer", "Payment"
]

CSV_PATH = Path(__file__).resolve().parent / "sample_retail_sales.csv"

HDFS_PATH = (
    "hdfs://localhost:9000/retail/input/retail_sales_big.csv"
)


# ============================================================
# 2. SPARK SESSION
# ============================================================

@st.cache_resource
def get_spark_session():
    """Reuse the same Spark session across Streamlit reruns."""
    spark = (
        SparkSession.builder
        .config(
            "spark.hadoop.fs.defaultFS",
            "hdfs://localhost:9000"
        )
        .appName("RetailStoreBigDataAnalytics")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ============================================================
# 3. SAMPLE DATA GENERATOR
# ============================================================

def generate_sample_data(number_of_rows=500):
    random.seed(42)

    products = {
        "Rice": ("Groceries", 60.0),
        "Milk": ("Dairy", 30.0),
        "Bread": ("Bakery", 40.0),
        "Soap": ("Personal Care", 35.0),
        "Shampoo": ("Personal Care", 120.0),
        "Biscuits": ("Snacks", 20.0),
        "Cooking Oil": ("Groceries", 150.0),
        "Juice": ("Beverages", 50.0),
        "Coffee": ("Beverages", 90.0),
        "Eggs": ("Dairy", 8.0),
        "Chocolate": ("Snacks", 25.0),
        "Cake": ("Bakery", 250.0),
    }

    rows = []

    for _ in range(number_of_rows):
        product = random.choice(list(products.keys()))
        category, price = products[product]
        quantity = random.randint(1, 5)

        rows.append({
            "Date": (
                date.today()
                - timedelta(days=random.randint(0, 89))
            ).isoformat(),
            "Product": product,
            "Category": category,
            "Store": random.choice(
                ["Store A", "Store B", "Store C"]
            ),
            "Customer": (
                f"Customer {random.randint(1, 100):03d}"
            ),
            "Quantity": quantity,
            "Price": float(price),
            "Payment": random.choice(["Cash", "UPI", "Card"]),
            "Sales": float(quantity * price),
        })

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


# ============================================================
# 4. LOAD LOCAL DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_sample_csv(path, modified_time):
    """Cache the bundled CSV until the file changes."""
    return pd.read_csv(path)


def load_input_data(uploaded_file):
    if uploaded_file is not None:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)

    if CSV_PATH.exists():
        return load_sample_csv(
            str(CSV_PATH),
            CSV_PATH.stat().st_mtime,
        )

    return generate_sample_data()


# ============================================================
# 5. DATA CLEANING AND SPARK DATAFRAME CREATION
# ============================================================

def prepare_spark_dataframe(spark, input_df, source_is_hdfs=False):
    """
    Clean HDFS Spark data natively or prepare local Pandas data.
    """

    # --------------------------------------------------------
    # HDFS: Clean directly in Spark
    # --------------------------------------------------------

    if source_is_hdfs:

        df = input_df.toDF(
            *[str(column).strip() for column in input_df.columns]
        )

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing required CSV columns: "
                + ", ".join(missing)
            )

        df = df.select(*REQUIRED_COLUMNS)

        raw_date = F.trim(F.col("Date").cast("string"))

        parsed_date = F.coalesce(
        F.to_date(
            F.to_timestamp(raw_date, "yyyy-MM-dd HH:mm:ss")
        ),
        F.to_date(raw_date, "yyyy-MM-dd"),
        F.to_date(raw_date, "M/d/yyyy"),
        F.to_date(raw_date, "M/d/yy")
)

        df = (
            df.withColumn("Date", parsed_date)
            .withColumn(
                "Quantity",
                F.col("Quantity").cast("double")
            )
            .withColumn(
                "Price",
                F.col("Price").cast("double")
            )
            .withColumn(
                "Sales",
                F.col("Sales").cast("double")
            )
        )

        for column in TEXT_COLUMNS:
            df = df.withColumn(
                column,
                F.when(
                    F.col(column).isNull()
                    | (
                        F.trim(
                            F.col(column).cast("string")
                        ) == ""
                    ),
                    F.lit("Unknown"),
                ).otherwise(
                    F.trim(
                        F.col(column).cast("string")
                    )
                )
            )

        df = df.filter(
            F.col("Date").isNotNull()
            & F.col("Quantity").isNotNull()
            & (F.col("Quantity") > 0)
            & (
                F.col("Quantity")
                == F.floor(F.col("Quantity"))
            )
            & F.col("Price").isNotNull()
            & (F.col("Price") >= 0)
            & F.col("Sales").isNotNull()
            & (F.col("Sales") >= 0)
        )

        return df.withColumn(
            "Quantity",
            F.col("Quantity").cast("int")
        )

    # --------------------------------------------------------
    # Local CSV / Uploaded CSV / Generated sample data
    # --------------------------------------------------------

    df = input_df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(missing)
        )

    df = df[REQUIRED_COLUMNS].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    for column in ["Quantity", "Price", "Sales"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=["Date", "Quantity", "Price", "Sales"]
    ).copy()

    df = df[
        (df["Quantity"] > 0)
        & (df["Quantity"] % 1 == 0)
        & (df["Price"] >= 0)
        & (df["Sales"] >= 0)
    ].copy()

    for column in TEXT_COLUMNS:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    records = [
        {
            "Date": str(row["Date"]),
            "Product": str(row["Product"]),
            "Category": str(row["Category"]),
            "Store": str(row["Store"]),
            "Customer": str(row["Customer"]),
            "Quantity": int(row["Quantity"]),
            "Price": float(row["Price"]),
            "Payment": str(row["Payment"]),
            "Sales": float(row["Sales"]),
        }
        for row in df.to_dict(orient="records")
    ]

    schema = StructType([
        StructField("Date", StringType(), False),
        StructField("Product", StringType(), True),
        StructField("Category", StringType(), True),
        StructField("Store", StringType(), True),
        StructField("Customer", StringType(), True),
        StructField("Quantity", IntegerType(), True),
        StructField("Price", DoubleType(), True),
        StructField("Payment", StringType(), True),
        StructField("Sales", DoubleType(), True),
    ])

    spark_df = spark.createDataFrame(
        records,
        schema=schema
    )

    return (
        spark_df
        .withColumn(
            "Date",
            F.to_date("Date", "yyyy-MM-dd")
        )
        .filter(F.col("Date").isNotNull())
    )


# ============================================================
# 6. SIDEBAR: DATA SOURCE
# ============================================================

with st.sidebar:

    st.header("📂 Data")

    data_source = st.radio(
        "Choose data source",
        ["HDFS", "Local / Upload CSV"],
        index=0,
        help=(
            "HDFS reads the CSV directly with Spark. "
            "Local mode uses an upload, the bundled sample CSV, "
            "or generated sample data."
        ),
    )

    uploaded_file = None

    if data_source == "Local / Upload CSV":

        uploaded_file = st.file_uploader(
            "Upload retail sales CSV",
            type=["csv"],
        )

    st.caption(
        "Required columns: "
        + ", ".join(REQUIRED_COLUMNS)
    )


# ============================================================
# 7. LOAD AND PREPARE DATA
# ============================================================

try:

    with st.spinner("Loading and preparing sales data..."):

        spark = get_spark_session()

        if data_source == "HDFS":

            # Read CSV directly from HDFS using Spark.
            hdfs_input_df = (
                spark.read
                .option("header", True)
                .option("inferSchema", False)
                .option("mode", "PERMISSIVE")
                .csv(HDFS_PATH)
            )

            sales_df = prepare_spark_dataframe(
                spark,
                hdfs_input_df,
                source_is_hdfs=True
            )

        else:

            input_data = load_input_data(uploaded_file)

            sales_df = prepare_spark_dataframe(
                spark,
                input_data
            )

        sales_df = sales_df.persist(
            StorageLevel.MEMORY_AND_DISK
        )

        valid_record_count = sales_df.count()

    if valid_record_count == 0:

        sales_df.unpersist(blocking=False)

        st.error(
            "No valid sales records were found. "
            "Check the selected CSV and its required columns."
        )

        st.stop()

except Exception as error:

    st.error(
        f"Could not load or process data: {error}"
    )

    if data_source == "HDFS":

        st.info(
            f"Check that HDFS is running, the file exists at "
            f"{HDFS_PATH}, and the CSV contains all required columns."
        )

    else:

        st.info(
            "Verify the CSV column names and confirm Java "
            "and PySpark are installed correctly."
        )

    st.stop()


st.success(
    f"Loaded {valid_record_count:,} valid sales records "
    f"into Apache Spark from {data_source}."
)


# ============================================================
# 8. SIDEBAR: FILTERS
# ============================================================

date_limits = sales_df.agg(
    F.min("Date").alias("minimum"),
    F.max("Date").alias("maximum"),
).first()

minimum_date = date_limits["minimum"]
maximum_date = date_limits["maximum"]

with st.sidebar:

    st.header("🔎 Filters")

    selected_dates = st.date_input(
        "Select date range",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )

    stores = [
        row["Store"]
        for row in sales_df.select("Store")
        .distinct()
        .orderBy("Store")
        .collect()
    ]

    categories = [
        row["Category"]
        for row in sales_df.select("Category")
        .distinct()
        .orderBy("Category")
        .collect()
    ]

    payment_methods = [
        row["Payment"]
        for row in sales_df.select("Payment")
        .distinct()
        .orderBy("Payment")
        .collect()
    ]

    selected_stores = st.multiselect(
        "Stores",
        stores,
        default=stores
    )

    selected_categories = st.multiselect(
        "Categories",
        categories,
        default=categories
    )

    selected_payments = st.multiselect(
        "Payment methods",
        payment_methods,
        default=payment_methods
    )


# ============================================================
# 9. FILTER DATA
# ============================================================

if (
    isinstance(selected_dates, (tuple, list))
    and len(selected_dates) == 2
):

    start_date, end_date = selected_dates

else:

    start_date, end_date = minimum_date, maximum_date


if (
    not selected_stores
    or not selected_categories
    or not selected_payments
):

    sales_df.unpersist(blocking=False)

    st.warning(
        "Select at least one store, category, "
        "and payment method."
    )

    st.stop()


filtered_df = sales_df.filter(
    (F.col("Date") >= F.lit(start_date.isoformat()))
    & (F.col("Date") <= F.lit(end_date.isoformat()))
    & F.col("Store").isin(selected_stores)
    & F.col("Category").isin(selected_categories)
    & F.col("Payment").isin(selected_payments)
).persist(StorageLevel.MEMORY_AND_DISK)


# ============================================================
# 10. KPIs
# ============================================================

try:

    kpis = filtered_df.agg(
        F.sum("Sales").alias("revenue"),
        F.count("*").alias("transactions"),
        F.sum("Quantity").alias("units"),
        F.countDistinct("Customer").alias("customers"),
        F.avg("Sales").alias("average_sale"),
    ).first()

    if not kpis["transactions"]:

        filtered_df.unpersist(blocking=False)
        sales_df.unpersist(blocking=False)

        st.warning(
            "No transactions match the selected filters."
        )

        st.stop()

    st.subheader("📊 Key Performance Indicators")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Revenue",
        f"₹{(kpis['revenue'] or 0):,.2f}",
    )

    col2.metric(
        "Transactions",
        f"{kpis['transactions']:,}",
    )

    col3.metric(
        "Units Sold",
        f"{(kpis['units'] or 0):,}",
    )

    col4.metric(
        "Unique Customers",
        f"{kpis['customers']:,}",
    )

    col5.metric(
        "Average Sale",
        f"₹{(kpis['average_sale'] or 0):,.2f}",
    )


    # ========================================================
    # 11. AGGREGATIONS FOR CHARTS
    # ========================================================

    with st.spinner("Calculating sales analytics..."):

        daily_sales = (
            filtered_df.groupBy("Date")
            .agg(F.sum("Sales").alias("Sales"))
            .orderBy("Date")
            .toPandas()
        )

        category_sales = (
            filtered_df.groupBy("Category")
            .agg(F.sum("Sales").alias("Sales"))
            .orderBy(F.desc("Sales"))
            .toPandas()
        )

        store_sales = (
            filtered_df.groupBy("Store")
            .agg(
                F.sum("Sales").alias("Revenue"),
                F.count("*").alias("Transactions"),
                F.sum("Quantity").alias("Units"),
            )
            .orderBy(F.desc("Revenue"))
            .toPandas()
        )

        product_sales = (
            filtered_df.groupBy("Product")
            .agg(
                F.sum("Sales").alias("Revenue"),
                F.sum("Quantity").alias("Units_Sold"),
                F.count("*").alias("Transactions"),
            )
            .orderBy(F.desc("Revenue"))
            .toPandas()
        )

        payment_sales = (
            filtered_df.groupBy("Payment")
            .agg(F.sum("Sales").alias("Sales"))
            .orderBy(F.desc("Sales"))
            .toPandas()
        )

        top_customers = (
            filtered_df.groupBy("Customer")
            .agg(
                F.sum("Sales").alias("Total_Spent"),
                F.count("*").alias("Transactions"),
            )
            .orderBy(F.desc("Total_Spent"))
            .limit(10)
            .toPandas()
        )

        recent_records = (
            filtered_df.orderBy(F.desc("Date"))
            .limit(500)
            .toPandas()
        )

    filtered_df.unpersist(blocking=False)
    sales_df.unpersist(blocking=False)

except Exception as error:

    filtered_df.unpersist(blocking=False)
    sales_df.unpersist(blocking=False)

    st.error(
        f"Analytics processing failed: {error}"
    )

    st.stop()


# ============================================================
# 12. DISPLAY CHARTS
# ============================================================

daily_sales["Date"] = pd.to_datetime(
    daily_sales["Date"]
)

st.subheader("📈 Sales Analysis")

left, right = st.columns(2)

with left:

    st.plotly_chart(
        px.line(
            daily_sales,
            x="Date",
            y="Sales",
            markers=True,
            title="Daily Revenue Trend",
        ),
        width="stretch",
    )

with right:

    st.plotly_chart(
        px.bar(
            category_sales,
            x="Category",
            y="Sales",
            color="Category",
            title="Revenue by Category",
        ),
        width="stretch",
    )


left, right = st.columns(2)

with left:

    st.plotly_chart(
        px.pie(
            payment_sales,
            names="Payment",
            values="Sales",
            hole=0.35,
            title="Sales by Payment Method",
        ),
        width="stretch",
    )

with right:

    st.plotly_chart(
        px.bar(
            store_sales,
            x="Store",
            y="Revenue",
            color="Store",
            title="Store Performance",
        ),
        width="stretch",
    )


# ============================================================
# 13. PRODUCT AND CUSTOMER TABLES
# ============================================================

st.subheader("🏆 Top-Selling Products")

st.dataframe(
    product_sales.head(10),
    width="stretch",
    hide_index=True,
)


st.subheader("📉 Lowest-Selling Products")

st.dataframe(
    product_sales.tail(10)
    .sort_values("Revenue"),
    width="stretch",
    hide_index=True,
)


st.subheader("👥 Top 10 Customers")

st.dataframe(
    top_customers,
    width="stretch",
    hide_index=True,
)


# ============================================================
# 14. BUSINESS INSIGHTS
# ============================================================

st.subheader("💡 Business Insights")

if not category_sales.empty:

    best_category = category_sales.iloc[0]

    st.info(
        f"Top category: {best_category['Category']} — "
        f"₹{best_category['Sales']:,.2f} revenue."
    )


if not product_sales.empty:

    best_product = product_sales.iloc[0]

    st.info(
        f"Top product by revenue: {best_product['Product']} — "
        f"₹{best_product['Revenue']:,.2f}."
    )


if not store_sales.empty:

    best_store = store_sales.iloc[0]

    st.info(
        f"Best-performing store: {best_store['Store']} — "
        f"₹{best_store['Revenue']:,.2f} revenue."
    )


if not daily_sales.empty:

    peak_day = daily_sales.loc[
        daily_sales["Sales"].idxmax()
    ]

    st.info(
        f"Peak sales date: {peak_day['Date'].date()} — "
        f"₹{peak_day['Sales']:,.2f} revenue."
    )


# ============================================================
# 15. RECENT RECORDS AND CSV DOWNLOAD
# ============================================================

st.subheader("📄 Recent Sales Records")

if not recent_records.empty:

    recent_records["Date"] = pd.to_datetime(
        recent_records["Date"]
    )

st.caption(
    "Displaying up to 500 recent filtered records."
)

st.dataframe(
    recent_records[REQUIRED_COLUMNS],
    width="stretch",
    hide_index=True,
)


st.download_button(
    label="📥 Download Filtered Sales CSV",
    data=(
        recent_records[REQUIRED_COLUMNS]
        .to_csv(index=False)
        .encode("utf-8")
    ),
    file_name="retail_sales_records.csv",
    mime="text/csv",
)


st.caption(
    "Technology: Apache Spark (PySpark), Streamlit, Pandas, and Plotly. "
    "Spark runs in local[*] mode, providing local parallel processing "
    "on this computer rather than a multi-machine cluster."
)