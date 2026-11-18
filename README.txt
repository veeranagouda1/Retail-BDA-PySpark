RETAIL STORE BIG DATA ANALYTICS — PYSPARK VERSION

This version uses Apache Spark (PySpark) for data cleaning, filtering, and aggregations.
Streamlit and Plotly display the interactive dashboard.

Setup:
1. For compatibility, use Python 3.12 or another version supported by your PySpark release.
   Python 3.15 may not yet be supported by all PySpark releases.
2. Install a supported Java LTS runtime (Java 17 or 21 are good choices).
3. Open this folder in VS Code.
4. Install dependencies: python -m pip install -r requirements.txt
5. Run: python -m streamlit run retail_analytics_app.py

CSV columns required:
Date, Product, Category, Store, Customer, Quantity, Price, Payment, Sales

The app uses Spark local[*] mode, which performs local parallel execution on this computer.
Do not claim a multi-machine cluster unless one is separately configured.
Only aggregated results and up to 500 recent rows are collected for dashboard display.
