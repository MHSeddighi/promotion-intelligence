import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def merge_product_data(
        transactions: pd.DataFrame,
        products: pd.DataFrame
) -> pd.DataFrame:
    return transactions.merge(
        products,
        on="PRODUCT_ID",
        how="left"
    )


def load_data(
        trans_path: str = "../data/raw/transaction_data.csv",
        products_path: str = "../data/raw/product.csv",
) -> pd.DataFrame:
    transactions = load_csv(
        trans_path
    )

    products = load_csv(
        products_path
    )

    df = merge_product_data(
        transactions,
        products
    )

    return df
