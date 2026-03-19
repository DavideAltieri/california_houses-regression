import pandas as pd

def info_data_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({
        "n_rows": len(df.index),
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
    })
    return report.sort_values("missing", ascending=False)

def print_head(df: pd.DataFrame, n: int = 10) -> None:
    print(f"\n=== FIRST {n} ROWS ===")
    print(df.head(n))

def print_info_report(df: pd.DataFrame) -> None:
    print("\n=== INFO DATA REPORT ===")
    print(info_data_report(df))

def print_describe(df: pd.DataFrame) -> None:
    print("\n=== DESCRIBE ===")
    print(df.describe())

def print_skewness(df: pd.DataFrame) -> None:
    print("\n=== SKEWNESS ===")
    print(df.skew(numeric_only=True).sort_values(ascending=False))

def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)

def print_correlation_matrix(df: pd.DataFrame) -> None:
    print("\n=== CORRELATION MATRIX ===")
    print(get_correlation_matrix(df))
    
def run_eda(df: pd.DataFrame) -> None:
    print_head(df)
    print_info_report(df)
    print_describe(df)
    print_skewness(df)
    print_correlation_matrix(df)