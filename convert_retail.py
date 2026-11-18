import pandas as pd

# Read the downloaded Excel dataset
df = pd.read_excel("Online Retail.xlsx")

# Convert the original dataset columns to dashboard columns
df = df.rename(columns={
    "InvoiceDate": "Date",
    "Description": "Product",
    "StockCode": "ProductCode",
    "CustomerID": "Customer",
    "Quantity": "Quantity",
    "UnitPrice": "Price",
    "Country": "Store",
    "PaymentMethod": "Payment"
})

# Remove rows without essential values
df = df.dropna(subset=["Date", "Product", "Customer", "Quantity", "Price"])

# Calculate sales
df["Sales"] = df["Quantity"] * df["Price"]

# Add Category and Payment columns if missing
if "Category" not in df.columns:
    df["Category"] = "General"

if "Payment" not in df.columns:
    df["Payment"] = "Unknown"

# Keep only the dashboard columns
df = df[[
    "Date", "Product", "Category", "Store",
    "Customer", "Quantity", "Price", "Payment", "Sales"
]]

# Save the converted dataset
df.to_csv("retail_sales_big.csv", index=False)

print("Conversion completed!")
print(f"Total records: {len(df):,}")
print("Output: retail_sales_big.csv")