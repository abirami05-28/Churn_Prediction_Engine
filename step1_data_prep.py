import pandas as pd
from config import RAW_EXCEL_PATH, CLEANED_TRANSACTIONS_PATH

SHEET_NAMES = ["Year 2009-2010", "Year 2010-2011"]


def load_raw_data() -> pd.DataFrame:
    if not RAW_EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_EXCEL_PATH}.\n"
            f"Download the UCI 'Online Retail II' Excel file from the UCI Machine "
            f"Learning Repository and save it at exactly that path (create the "
            f"'data' folder next to this script if it doesn't exist yet)."
        )

    frames = []
    for sheet in SHEET_NAMES:
        df = pd.read_excel(RAW_EXCEL_PATH, sheet_name=sheet)
        df["source_sheet"] = sheet
        frames.append(df)
        print(f"Loaded sheet '{sheet}': {len(df):,} rows")

    raw = pd.concat(frames, ignore_index=True)
    print(f"Total raw rows across both sheets: {len(raw):,}")
    return raw


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Standardise column names (handles stray whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Customer ID": "CustomerID"})

    n_start = len(df)

    # Parse InvoiceDate. The official UCI .xlsx stores this as a real Excel
    # datetime already (pandas reads it natively). If you were instead given
    # a CSV/text export like your sample ("01-12-2009 07:45"), this branch
    # parses that exact format, with a generic fallback for anything else.
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"], format="%d-%m-%Y %H:%M", errors="coerce"
        )
        still_bad = df["InvoiceDate"].isna()
        if still_bad.any():
            df.loc[still_bad, "InvoiceDate"] = pd.to_datetime(
                raw.loc[still_bad, "InvoiceDate"], errors="coerce"
            )

    n_bad_dates = df["InvoiceDate"].isna().sum()
    print(f"Rows with unparsable InvoiceDate (dropped): {n_bad_dates:,}")
    df = df[df["InvoiceDate"].notna()]

    # Drop rows with no Customer ID -- unusable for a customer-level model
    n_missing_customer = df["CustomerID"].isna().sum()
    print(f"Rows with missing CustomerID (dropped): {n_missing_customer:,}")
    df = df[df["CustomerID"].notna()]
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    # Drop rows with non-positive Price -- adjustments/errors, not real sales
    n_bad_price = (df["Price"] <= 0).sum()
    print(f"Rows with Price <= 0 (dropped): {n_bad_price:,}")
    df = df[df["Price"] > 0]

    # Drop rows with Quantity == 0 (no-ops)
    n_zero_quantity = (df["Quantity"] == 0).sum()
    print(f"Rows with Quantity == 0 (dropped): {n_zero_quantity:,}")
    df = df[df["Quantity"] != 0]
    # Flag cancellations: UCI convention is an Invoice number starting with 'C'
    df["Invoice"] = df["Invoice"].astype(str)
    df["is_cancellation"] = df["Invoice"].str.startswith("C")

    # Revenue: for cancellations, Quantity is already negative in the raw
    # data, so Revenue is naturally negative (a real refund) -- we never
    # flip signs or invent values.
    df["Revenue"] = df["Quantity"] * df["Price"]

    df = df.rename(columns={"Country": "country"})

    n_end = len(df)
    pct_kept = 100 * n_end / n_start if n_start else 0
    print(f"Rows kept: {n_end:,} / {n_start:,} raw rows ({pct_kept:.1f}%)")

    return df.reset_index(drop=True)


def main():
    raw = load_raw_data()
    cleaned = clean_transactions(raw)
    cleaned.to_csv(CLEANED_TRANSACTIONS_PATH, index=False)
    print(f"Saved cleaned transactions to: {CLEANED_TRANSACTIONS_PATH}")
    print(f"Date range: {cleaned['InvoiceDate'].min()} -> {cleaned['InvoiceDate'].max()}")
    print(f"Unique customers: {cleaned['CustomerID'].nunique():,}")
    print("STEP 1 COMPLETE")


if __name__ == "__main__":
    main()
