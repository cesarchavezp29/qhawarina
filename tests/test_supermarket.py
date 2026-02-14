"""Tests for supermarket price scraper and index builder.

Tests use mock API responses — no real network calls.
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ingestion.supermarket import (
    CATEGORY_KEYWORDS,
    FOOD_CATEGORIES,
    STORES,
    SUPERMARKET_SERIES,
    PriceIndexBuilder,
    SupermarketAggregator,
    SupermarketScraper,
    VTEXClient,
    _classify_product,
)


# ── Mock VTEX API responses ──────────────────────────────────────────────────

MOCK_CATEGORY_TREE = [
    {
        "id": 1,
        "name": "Alimentos",
        "children": [
            {
                "id": 10,
                "name": "Lácteos",
                "children": [
                    {"id": 100, "name": "Leche", "children": []},
                    {"id": 101, "name": "Queso", "children": []},
                ],
            },
            {
                "id": 11,
                "name": "Carnes",
                "children": [
                    {"id": 110, "name": "Pollo", "children": []},
                ],
            },
        ],
    },
    {
        "id": 2,
        "name": "Bebidas",
        "children": [
            {"id": 20, "name": "Agua", "children": []},
        ],
    },
]

MOCK_PRODUCTS = [
    {
        "productName": "Leche Gloria Entera 1L",
        "brand": "Gloria",
        "items": [
            {
                "itemId": "12345",
                "ean": "7751271012345",
                "unitMultiplier": 1,
                "sellers": [
                    {
                        "commertialOffer": {
                            "Price": 5.50,
                            "ListPrice": 6.00,
                            "IsAvailable": True,
                        }
                    }
                ],
            }
        ],
    },
    {
        "productName": "Queso Edam Laive 200g",
        "brand": "Laive",
        "items": [
            {
                "itemId": "12346",
                "ean": "7751271012346",
                "unitMultiplier": 1,
                "sellers": [
                    {
                        "commertialOffer": {
                            "Price": 12.90,
                            "ListPrice": 14.50,
                            "IsAvailable": True,
                        }
                    }
                ],
            }
        ],
    },
    {
        "productName": "Arroz Extra Costeño 5kg",
        "brand": "Costeño",
        "items": [
            {
                "itemId": "12347",
                "ean": "7751271012347",
                "unitMultiplier": 1,
                "sellers": [
                    {
                        "commertialOffer": {
                            "Price": 21.90,
                            "ListPrice": 21.90,
                            "IsAvailable": True,
                        }
                    }
                ],
            }
        ],
    },
    {
        "productName": "Pollo Entero San Fernando",
        "brand": "San Fernando",
        "items": [
            {
                "itemId": "12348",
                "ean": "7751271012348",
                "unitMultiplier": 1,
                "sellers": [
                    {
                        "commertialOffer": {
                            "Price": 9.80,
                            "ListPrice": 10.50,
                            "IsAvailable": True,
                        }
                    }
                ],
            }
        ],
    },
]


# ── VTEXClient tests ─────────────────────────────────────────────────────────

class TestVTEXClient:
    def test_init(self):
        client = VTEXClient("www.plazavea.com.pe")
        assert client.store_domain == "www.plazavea.com.pe"
        assert client.base_url == "https://www.plazavea.com.pe"
        assert "NEXUS" in client.session.headers["User-Agent"]

    def test_extract_price_info_valid(self):
        info = VTEXClient.extract_price_info(MOCK_PRODUCTS[0])
        assert info is not None
        assert info["sku_id"] == "12345"
        assert info["product_name"] == "Leche Gloria Entera 1L"
        assert info["brand"] == "Gloria"
        assert info["price"] == 5.50
        assert info["list_price"] == 6.00
        assert info["available"] is True
        assert info["unit_multiplier"] == 1.0

    def test_extract_price_info_no_items(self):
        product = {"productName": "Test", "items": []}
        assert VTEXClient.extract_price_info(product) is None

    def test_extract_price_info_no_sellers(self):
        product = {
            "productName": "Test",
            "items": [{"itemId": "1", "sellers": []}],
        }
        assert VTEXClient.extract_price_info(product) is None

    def test_extract_price_info_zero_price(self):
        product = {
            "productName": "Test",
            "items": [
                {
                    "itemId": "1",
                    "sellers": [
                        {"commertialOffer": {"Price": 0, "ListPrice": 0, "IsAvailable": False}}
                    ],
                }
            ],
        }
        assert VTEXClient.extract_price_info(product) is None

    def test_extract_price_info_missing_fields(self):
        product = {"unexpected": "structure"}
        assert VTEXClient.extract_price_info(product) is None

    @patch("src.ingestion.supermarket.VTEXClient._get")
    def test_get_category_tree(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_CATEGORY_TREE
        mock_get.return_value = mock_resp

        client = VTEXClient("www.plazavea.com.pe")
        tree = client.get_category_tree()
        assert len(tree) == 2
        assert tree[0]["name"] == "Alimentos"

    @patch("src.ingestion.supermarket.VTEXClient._get")
    def test_get_leaf_categories(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_CATEGORY_TREE
        mock_get.return_value = mock_resp

        client = VTEXClient("www.plazavea.com.pe")
        leaves = client.get_leaf_categories()

        # Should find: Leche(100), Queso(101), Pollo(110), Agua(20)
        assert len(leaves) == 4
        ids = {l["id"] for l in leaves}
        assert ids == {100, 101, 110, 20}

        # Check hierarchy path
        leche = [l for l in leaves if l["id"] == 100][0]
        assert leche["l1"] == "Alimentos"
        assert leche["l2"] == "Lácteos"
        assert leche["l3"] == "Leche"

    @patch("src.ingestion.supermarket.VTEXClient._get")
    def test_search_products(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_PRODUCTS[:2]
        mock_get.return_value = mock_resp

        client = VTEXClient("www.plazavea.com.pe")
        products = client.search_products(100, offset=0)
        assert len(products) == 2

    @patch("src.ingestion.supermarket.VTEXClient._get")
    def test_get_all_products_pagination(self, mock_get):
        """Test pagination stops when fewer products than page size."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_PRODUCTS  # 4 products < 50 page size
        mock_get.return_value = mock_resp

        client = VTEXClient("www.plazavea.com.pe")
        products = client.get_all_products_in_category(100)
        assert len(products) == 4
        assert mock_get.call_count == 1  # Only one page needed


# ── Product classification tests ──────────────────────────────────────────────

class TestProductClassification:
    def test_classify_leche(self):
        assert _classify_product("Leche Gloria Entera 1L", "Alimentos") == "lacteos"

    def test_classify_arroz(self):
        assert _classify_product("Arroz Extra Costeño 5kg", "Alimentos") == "arroz_cereales"

    def test_classify_pollo(self):
        assert _classify_product("Pollo Entero San Fernando", "Carnes") == "carnes"

    def test_classify_huevos(self):
        assert _classify_product("Huevo Rosado x 15 unid", "Alimentos") == "huevos"

    def test_classify_unknown(self):
        assert _classify_product("Something Unknown Product", "Misc") == "other"

    def test_classify_case_insensitive(self):
        assert _classify_product("LECHE ENTERA", "Alimentos") == "lacteos"

    def test_all_food_categories_have_keywords(self):
        for cat in FOOD_CATEGORIES:
            assert cat in CATEGORY_KEYWORDS, f"Category {cat} has no keywords"


# ── PriceIndexBuilder tests ───────────────────────────────────────────────────

class TestPriceIndexBuilder:
    def _make_snapshot(
        self, d: date, prices: dict[str, float],
        store: str = "plazavea", category: str = "lacteos",
    ) -> pd.DataFrame:
        """Helper: create a mock daily snapshot with enough products per category."""
        rows = []
        for sku_id, price in prices.items():
            rows.append({
                "date": d.isoformat(),
                "store": store,
                "sku_id": sku_id,
                "product_name": f"Leche Product {sku_id}",
                "brand": "TestBrand",
                "category_l1": "Alimentos",
                "category_l2": "Lácteos",
                "category_l3": "Leche",
                "price": price,
                "list_price": price,
                "ean": "",
                "available": True,
                "unit_multiplier": 1.0,
            })
        return pd.DataFrame(rows)

    def _make_large_snapshot(self, d: date, base_price: float, multiplier: float = 1.0) -> pd.DataFrame:
        """Create snapshot with 6 products in lacteos (above 5-product minimum)."""
        prices = {
            f"SKU{i}": base_price * (1 + i * 0.1) * multiplier
            for i in range(6)
        }
        return self._make_snapshot(d, prices)

    def test_compute_daily_ratios_basic(self):
        """Test Jevons ratio with 6 matched products (above minimum)."""
        builder = PriceIndexBuilder()

        # 6 products: all increase by 10%
        prev_prices = {f"P{i}": 10.0 + i for i in range(6)}
        curr_prices = {f"P{i}": (10.0 + i) * 1.1 for i in range(6)}

        prev = self._make_snapshot(date(2025, 1, 1), prev_prices)
        prev["cpi_category"] = "lacteos"

        curr = self._make_snapshot(date(2025, 1, 2), curr_prices)
        curr["cpi_category"] = "lacteos"

        record = builder._compute_daily_ratios(prev, curr, date(2025, 1, 2))

        # All products have ratio 1.1, so Jevons = 1.1
        assert record["ratio_lacteos"] == pytest.approx(1.1, rel=1e-4)
        assert record["n_matched_all"] == 6
        # Weighted aggregate should also reflect the increase
        assert record["ratio_all"] > 1.0

    def test_compute_daily_ratios_no_match(self):
        builder = PriceIndexBuilder()

        prev = self._make_snapshot(date(2025, 1, 1), {"A": 10.0})
        prev["cpi_category"] = "lacteos"

        curr = self._make_snapshot(date(2025, 1, 2), {"B": 20.0})
        curr["cpi_category"] = "lacteos"

        record = builder._compute_daily_ratios(prev, curr, date(2025, 1, 2))
        assert record["ratio_all"] == 1.0
        assert record["n_matched_all"] == 0

    def test_compute_daily_ratios_extreme_trimmed(self):
        builder = PriceIndexBuilder()

        prev = self._make_snapshot(date(2025, 1, 1), {"A": 10.0, "B": 20.0})
        prev["cpi_category"] = "lacteos"

        # Product A has extreme price change (3x → trimmed)
        curr = self._make_snapshot(date(2025, 1, 2), {"A": 30.0, "B": 20.0})
        curr["cpi_category"] = "lacteos"

        record = builder._compute_daily_ratios(prev, curr, date(2025, 1, 2))
        # A has ratio 3.0 (trimmed), B has ratio 1.0
        # Only B survives filtering
        assert record["n_matched_all"] == 1

    def test_weighted_geometric_mean(self):
        """Test CPI-weighted geometric aggregation."""
        # Two categories: one goes up 10%, one flat
        cat_ratios = {"lacteos": 1.10, "carnes": 1.00}
        weights = {"lacteos": 0.4, "carnes": 0.6}

        result = PriceIndexBuilder._weighted_geometric_mean(cat_ratios, weights)
        # exp(0.4*ln(1.1) + 0.6*ln(1.0)) = exp(0.4*0.0953) = exp(0.0381) ≈ 1.0389
        expected = np.exp(0.4 * np.log(1.1) + 0.6 * np.log(1.0))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_weighted_geometric_mean_empty(self):
        assert PriceIndexBuilder._weighted_geometric_mean({}, {}) == 1.0

    def test_build_daily_index_chainlinks(self, tmp_path):
        """Test that daily index properly chain-links."""
        builder = PriceIndexBuilder()

        d1, d2, d3 = date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)

        # 6 products per snapshot (above 5-product minimum), prices increase each day
        snap1 = self._make_large_snapshot(d1, 10.0, multiplier=1.0)
        snap2 = self._make_large_snapshot(d2, 10.0, multiplier=1.05)
        snap3 = self._make_large_snapshot(d3, 10.0, multiplier=1.10)

        with patch.object(builder.scraper, "list_available_snapshots", return_value=[d1, d2, d3]):
            with patch.object(builder.scraper, "load_snapshot", side_effect=[snap1, snap2, snap3]):
                daily = builder.build_daily_index()

        assert len(daily) == 3  # base + 2 ratio days
        assert daily.iloc[0]["index_all"] == 100.0  # base

        # Index should increase since all prices go up
        assert daily.iloc[1]["index_all"] > 100.0
        assert daily.iloc[2]["index_all"] > daily.iloc[1]["index_all"]

    def test_build_monthly_index(self):
        """Test monthly aggregation from daily index."""
        builder = PriceIndexBuilder()

        # Create synthetic daily index
        dates = pd.date_range("2025-01-01", "2025-03-31", freq="D")
        daily_index = pd.DataFrame({
            "date": dates,
            "index_all": np.linspace(100, 103, len(dates)),
            "index_food": np.linspace(100, 104, len(dates)),
        })

        monthly = builder.build_monthly_index(daily_index)

        assert len(monthly) == 3  # Jan, Feb, Mar
        assert "var_all" in monthly.columns
        assert "var_food" in monthly.columns

        # First month has NaN variation (no previous month)
        assert pd.isna(monthly.iloc[0]["var_all"])

        # Subsequent months should have positive variation
        assert monthly.iloc[1]["var_all"] > 0
        assert monthly.iloc[2]["var_all"] > 0


# ── SupermarketAggregator tests ───────────────────────────────────────────────

class TestSupermarketAggregator:
    def test_build_panel_series(self):
        """Test conversion of monthly index to panel format."""
        agg = SupermarketAggregator()

        # Create synthetic monthly data
        dates = pd.date_range("2025-01-01", "2025-06-01", freq="MS")
        monthly = pd.DataFrame({
            "date": dates,
            "index_all": [100, 101, 102, 101.5, 103, 104],
            "index_food": [100, 101.5, 103, 102, 104, 105],
            "var_all": [np.nan, 1.0, 0.99, -0.49, 1.48, 0.97],
            "var_food": [np.nan, 1.5, 1.48, -0.97, 1.96, 0.96],
        })

        panel = agg.build_panel_series(monthly)

        assert not panel.empty
        assert "series_id" in panel.columns
        assert "value_raw" in panel.columns
        assert "source" in panel.columns
        assert panel["source"].unique().tolist() == ["SUPERMARKET"]

        # Should have at least the 4 main series
        series_ids = panel["series_id"].unique()
        assert "SUPERMARKET_ALL_INDEX" in series_ids
        assert "SUPERMARKET_ALL_VAR" in series_ids
        assert "SUPERMARKET_FOOD_INDEX" in series_ids
        assert "SUPERMARKET_FOOD_VAR" in series_ids

    def test_panel_series_schema(self):
        """Test that panel output has correct schema."""
        agg = SupermarketAggregator()

        dates = pd.date_range("2025-01-01", "2025-03-01", freq="MS")
        monthly = pd.DataFrame({
            "date": dates,
            "index_all": [100, 101, 102],
            "var_all": [np.nan, 1.0, 0.99],
        })

        panel = agg.build_panel_series(monthly)
        required_cols = [
            "date", "series_id", "series_name", "category",
            "value_raw", "value_sa", "value_log", "value_dlog", "value_yoy",
            "source", "frequency_original", "publication_lag_days",
        ]
        for col in required_cols:
            assert col in panel.columns, f"Missing column: {col}"

    def test_index_series_have_transforms(self):
        """Test that index series have log/dlog/yoy transforms."""
        agg = SupermarketAggregator()

        dates = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
        n = len(dates)
        monthly = pd.DataFrame({
            "date": dates,
            "index_all": np.linspace(100, 110, n),
            "var_all": [np.nan] + [0.4] * (n - 1),
        })

        panel = agg.build_panel_series(monthly)

        idx_panel = panel[panel["series_id"] == "SUPERMARKET_ALL_INDEX"]
        assert idx_panel["value_log"].notna().any()
        assert idx_panel["value_dlog"].notna().any()

    def test_var_series_no_log_transforms(self):
        """Test that variation series have NaN for log/dlog/yoy."""
        agg = SupermarketAggregator()

        dates = pd.date_range("2025-01-01", "2025-03-01", freq="MS")
        monthly = pd.DataFrame({
            "date": dates,
            "index_all": [100, 101, 102],
            "var_all": [np.nan, 1.0, 0.99],
        })

        panel = agg.build_panel_series(monthly)
        var_panel = panel[panel["series_id"] == "SUPERMARKET_ALL_VAR"]
        assert var_panel["value_log"].isna().all()
        assert var_panel["value_dlog"].isna().all()
        assert var_panel["value_yoy"].isna().all()


# ── SupermarketScraper tests ─────────────────────────────────────────────────

class TestSupermarketScraper:
    def test_stores_config(self):
        assert "plazavea" in STORES
        assert "metro" in STORES
        assert "wong" in STORES
        for domain in STORES.values():
            assert domain.endswith(".pe")

    def test_scrape_store_mock(self, tmp_path):
        """Test scraping a store with mocked API."""
        mock_client = MagicMock(spec=VTEXClient)
        # scrape_store now uses get_category_tree() + L1 search
        mock_client.get_category_tree.return_value = [
            {"id": 431, "name": "Abarrotes", "children": []},
        ]
        mock_client.get_all_products_in_category.return_value = MOCK_PRODUCTS

        # Build a mock class that returns our mock instance but keeps static methods
        mock_cls = MagicMock(return_value=mock_client)
        mock_cls.extract_price_info = VTEXClient.extract_price_info  # preserve static method

        with patch("src.ingestion.supermarket.RAW_SUPERMARKET_DIR", tmp_path):
            with patch("src.ingestion.supermarket.SUPERMARKET_SNAPSHOTS_DIR", tmp_path / "snapshots"):
                (tmp_path / "snapshots").mkdir()
                scraper = SupermarketScraper()
                with patch("src.ingestion.supermarket.VTEXClient", mock_cls):
                    df = scraper.scrape_store("plazavea", date(2025, 1, 15))

        assert not df.empty
        assert "price" in df.columns
        assert df["store"].unique().tolist() == ["plazavea"]
        assert len(df) == 4  # 4 mock products

    def test_scrape_store_invalid(self):
        scraper = SupermarketScraper()
        with pytest.raises(ValueError, match="Unknown store"):
            scraper.scrape_store("invalid_store")

    def test_save_and_load_snapshot(self, tmp_path):
        """Test snapshot persistence round-trip."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()

        with patch("src.ingestion.supermarket.RAW_SUPERMARKET_DIR", tmp_path):
            with patch("src.ingestion.supermarket.SUPERMARKET_SNAPSHOTS_DIR", snapshot_dir):
                scraper = SupermarketScraper()

                df = pd.DataFrame({
                    "date": ["2025-01-15"],
                    "store": ["plazavea"],
                    "sku_id": ["12345"],
                    "product_name": ["Test Product"],
                    "brand": ["Test"],
                    "category_l1": ["Alimentos"],
                    "category_l2": [""],
                    "category_l3": [""],
                    "price": [5.50],
                    "list_price": [6.00],
                    "ean": [""],
                    "available": [True],
                    "unit_multiplier": [1.0],
                })

                target_date = date(2025, 1, 15)
                path = scraper.save_daily_snapshot(df, target_date)
                assert path.exists()

                loaded = scraper.load_snapshot(target_date)
                assert len(loaded) == 1
                assert loaded.iloc[0]["price"] == 5.50

    def test_list_available_snapshots(self, tmp_path):
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()

        # Create some mock snapshot files
        for d in ["2025-01-15", "2025-01-16", "2025-01-17"]:
            pd.DataFrame({"x": [1]}).to_parquet(snapshot_dir / f"{d}.parquet")

        with patch("src.ingestion.supermarket.SUPERMARKET_SNAPSHOTS_DIR", snapshot_dir):
            scraper = SupermarketScraper()
            snapshots = scraper.list_available_snapshots()

        assert len(snapshots) == 3
        assert snapshots[0] == date(2025, 1, 15)
        assert snapshots[-1] == date(2025, 1, 17)


# ── Series definitions tests ─────────────────────────────────────────────────

class TestSeriesDefinitions:
    def test_supermarket_series_dict_valid(self):
        for sid, spec in SUPERMARKET_SERIES.items():
            assert "source_col" in spec
            assert "name" in spec
            assert "category" in spec
            assert "is_index" in spec

    def test_food_categories_have_series(self):
        """Each food category should have index + var series."""
        for cat in FOOD_CATEGORIES:
            idx_key = f"SUPERMARKET_{cat.upper()}_INDEX"
            var_key = f"SUPERMARKET_{cat.upper()}_VAR"
            assert idx_key in SUPERMARKET_SERIES, f"Missing {idx_key}"
            assert var_key in SUPERMARKET_SERIES, f"Missing {var_key}"

    def test_main_series_present(self):
        assert "SUPERMARKET_ALL_INDEX" in SUPERMARKET_SERIES
        assert "SUPERMARKET_ALL_VAR" in SUPERMARKET_SERIES
        assert "SUPERMARKET_FOOD_INDEX" in SUPERMARKET_SERIES
        assert "SUPERMARKET_FOOD_VAR" in SUPERMARKET_SERIES


# ── Integration tests ─────────────────────────────────────────────────────────

class TestIntegration:
    def test_supermarket_in_inflation_series(self):
        """Verify supermarket series are included in DFM inflation list."""
        from src.models.dfm import INFLATION_SERIES

        assert "SUPERMARKET_FOOD_VAR" in INFLATION_SERIES
        assert "SUPERMARKET_ALL_VAR" in INFLATION_SERIES

    def test_settings_paths_exist(self):
        """Verify settings exports the new path constants."""
        from config.settings import RAW_SUPERMARKET_DIR, SUPERMARKET_SNAPSHOTS_DIR

        assert RAW_SUPERMARKET_DIR is not None
        assert SUPERMARKET_SNAPSHOTS_DIR is not None
        assert "supermarket" in str(RAW_SUPERMARKET_DIR)

    def test_update_nexus_has_supermarket_step(self):
        """Verify supermarket step is registered in update_nexus."""
        from scripts.update_nexus import STEPS, VALID_STEPS

        assert "supermarket" in VALID_STEPS
        assert "supermarket" in STEPS
