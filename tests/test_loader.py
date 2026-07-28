import pandas as pd
from app.data.loader import DataLoader


def test_loader_initializes():
    loader = DataLoader()
    assert loader is not None


def test_load_transactions():
    loader = DataLoader()
    df = loader.load_transactions()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "PRODUCT_ID" in df.columns
    assert "QUANTITY" in df.columns
    assert "SALES_VALUE" in df.columns
    assert "DAY" in df.columns


def test_load_products():
    loader = DataLoader()
    df = loader.load_products()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "PRODUCT_ID" in df.columns
    assert "DEPARTMENT" in df.columns


def test_load_campaign_desc():
    loader = DataLoader()
    df = loader.load_campaign_desc()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "CAMPAIGN" in df.columns
    assert "START_DAY" in df.columns
    assert "END_DAY" in df.columns


def test_get_all_campaign_ids():
    loader = DataLoader()
    ids = loader.get_all_campaign_ids()
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_get_sales_with_products():
    loader = DataLoader()
    df = loader.get_sales_with_products()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "DEPARTMENT" in df.columns
    assert "COMMODITY_DESC" in df.columns
