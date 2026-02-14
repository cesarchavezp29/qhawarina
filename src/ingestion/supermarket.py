"""Supermarket price scraper — Billion Prices Project for Peru.

Scrapes unauthenticated VTEX public APIs from three major Peruvian
supermarket chains (Plaza Vea, Metro, Wong) to build high-frequency
price indices. Peru is NOT covered by PriceStats — this fills a
genuine gap. Food & beverages = 23% of CPI basket.

Architecture:
    1. VTEXClient        — Low-level API client with rate limiting
    2. SupermarketScraper — Daily collection orchestrator
    3. PriceIndexBuilder  — BPP-style Jevons index construction
    4. SupermarketAggregator — Panel-format output for nowcasting
"""

import csv
import logging
import random
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config.settings import (
    PROCESSED_NATIONAL_DIR,
    RAW_SUPERMARKET_DIR,
    SUPERMARKET_SNAPSHOTS_DIR,
)

logger = logging.getLogger("nexus.supermarket")

# ── Store configuration ──────────────────────────────────────────────────────

STORES = {
    "plazavea": "www.plazavea.com.pe",
    "metro": "www.metro.pe",
    "wong": "www.wong.pe",
}

# CPI-relevant food category keywords for classification
CATEGORY_KEYWORDS = {
    "arroz_cereales": [
        "arroz", "avena", "quinua", "trigo", "cereal", "maiz", "cebada",
        "fideos", "pasta", "spagueti", "tallarin", "lenteja", "frejol",
        "garbanzo", "pallar", "haba seca", "menestra", "granola",
        "tortilla", "taco", "maicena",
    ],
    "aceites_grasas": [
        "aceite", "mantequilla", "margarina", "manteca", "oliva",
        "aceituna",
    ],
    "azucar_dulces": [
        "azucar", "azúcar", "miel", "mermelada", "chocolate", "caramelo",
        "galleta", "cocoa", "paneton", "panetón", "wafer", "snack",
        "helado", "gomita", "marshmallow", "turrón", "turron",
        "manjar", "alfajor", "brownie", "cupcake", "torta", "kekon",
        "cookie", "chips ahoy",
    ],
    "lacteos": [
        "leche", "queso", "yogurt", "yogur", "crema de leche",
        "lacto", "lácteo", "lacteo", "shake", "batido",
    ],
    "carnes": [
        "pollo", "carne", "res ", "cerdo", "cordero", "pavo", "jamon",
        "salchicha", "chorizo", "tocino", "higado", "mondongo",
        "chicharron", "chicharrón", "embutido", "jamonada", "hot dog",
        "hamburguesa", "nugget", "apanado", "salame", "mortadela",
        "paté", "pate", "prosciutto", "lomo",
    ],
    "pescados_mariscos": [
        "pescado", "atun", "atún", "sardina", "trucha", "bonito", "jurel",
        "caballa", "langostino", "camaron", "camarón", "calamar",
        "conserva de pescado", "surimi", "filete de",
    ],
    "pan_harinas": [
        "pan ", "pan de", "harina", "molde", "baguette", "ciabatta",
        "integral",
    ],
    "frutas": [
        "manzana", "naranja", "platano", "plátano", "mandarina", "uva",
        "fresa", "mango", "papaya", "palta", "limon", "limón", "pera",
        "durazno", "piña", "sandia", "sandía", "melon", "melón",
        "chirimoya", "kiwi", "arándano", "arandano", "frambuesa",
        "cereza", "granada", "maracuya", "granadilla", "lucuma",
        "lúcuma", "coco",
    ],
    "verduras": [
        "lechuga", "tomate", "cebolla", "ajo", "zanahoria", "papa",
        "camote", "brócoli", "brocoli", "espinaca", "pepino", "zapallo",
        "pimiento", "choclo", "apio", "betarraga", "vainita",
        "holantao", "albahaca", "culantro", "perejil", "rocoto",
        "aceituna", "nabo", "col ", "repollo", "coliflor", "alcachofa",
    ],
    "huevos": [
        "huevo",
    ],
    "bebidas": [
        "agua mineral", "gaseosa", "jugo", "néctar", "nectar", "refresco",
        "cerveza", "vino", "energizante", "agua de mesa",
        "agua tónica", "agua tonica", "infusion", "infusión",
        " te ", "té ", "tisana", "manzanilla", "hierba",
        "limonada", "chicha", "emoliente", "soda", "sprite",
        "coca cola", "coca-cola", "inca kola", "pepsi", "fanta",
        "san mateo", "san luis", "cielo", "pisco", "ron ", "whisky",
        "vodka", "tequila", "espumante", "sangria", "champagne",
    ],
    "limpieza": [
        "detergente", "lejia", "lejía", "desinfectante", "jabon", "jabón",
        "lavavajilla", "suavizante", "limpiador", "ambientador",
        "trapeador", "escoba", "recogedor", "esponja", "guante",
        "bolsa de basura", "insecticida", "cera ", "quitamanchas",
        "cloro",
    ],
    "cuidado_personal": [
        "shampoo", "champu", "champú", "crema dental", "pasta dental",
        "desodorante", "papel higienico", "papel higiénico", "toalla",
        "acondicionador", "antitranspirante", "protector solar",
        "bloqueador", "crema corporal", "crema facial",
        "pañal", "panal", "cepillo dental", "enjuague bucal",
        "colonia", "perfume", "rastrillo", "afeitad", "gel de afeitar",
        "serum", "sérum", "tinte", "maquina de afeitar", "máquina de afeitar",
        "maquillaje", "base covergirl", "labial", "rimel", "rimmel",
        "pestañas", "preservativo", "condon", "condón",
        "hilo dental", "enjuague", "cepillo de dient",
        "vitamina", "suplemento", "colageno", "colágeno",
    ],
}

# L1 store category → CPI category fallback mapping
# Used when keyword matching fails: maps store category names to CPI categories.
L1_CATEGORY_MAP = {
    "Abarrotes": "arroz_cereales",
    "Frutas y Verduras": "verduras",  # mixed, but more veg by weight
    "Carnes, Aves y Pescados": "carnes",
    "Lácteos": "lacteos",
    "Lácteos y Huevos": "lacteos",
    "Quesos y Fiambres": "lacteos",
    "Embutidos y Fiambres": "carnes",
    "Congelados": "carnes",  # mostly frozen meat/fish
    "Panadería y Pastelería": "pan_harinas",
    "Desayuno": "arroz_cereales",
    "Desayunos": "arroz_cereales",
    "Comidas y Rostizados": "carnes",
    "Pollo Rostizado y Comidas Preparadas": "carnes",
    "Bebidas": "bebidas",
    "Aguas y Bebidas": "bebidas",
    "Cervezas, Vinos y Licores": "bebidas",
    "Vinos, licores y cervezas": "bebidas",
    "Limpieza": "limpieza",
    "Higiene, Salud y Belleza": "cuidado_personal",
    "Cuidado Personal y Salud": "cuidado_personal",
    "Mercado Saludable": "verduras",  # organic/healthy produce
}

# Food categories for the food-specific index
FOOD_CATEGORIES = {
    "arroz_cereales", "aceites_grasas", "azucar_dulces", "lacteos",
    "carnes", "pescados_mariscos", "pan_harinas", "frutas", "verduras",
    "huevos", "bebidas",
}

# ── CPI basket weights (INEI, base Dec 2021=100) ────────────────────────────
# Cavallo/Rigobon BPP methodology: weight category-level Jevons indices
# by official CPI expenditure shares to construct the aggregate.
# Source: INEI "Estructura de Ponderaciones del IPC Lima Metropolitana Base Dic 2021"
# Extracted from: estructura_ponderaciones_ipc_lima_dic2021.csv
# Only scrapable categories are included; weights are renormalized to sum to 1.
CPI_WEIGHTS_RAW = {
    "arroz_cereales": 4.231,    # 011100: Pan y cereales
    "aceites_grasas": 0.535,    # 011500: Aceites y grasas
    "azucar_dulces": 0.890,     # 011800: Azúcar, mermelada, miel, chocolate y dulces
    "lacteos": 2.848,           # 011400: Leche, queso y huevos
    "carnes": 5.227,            # 011200: Carne
    "pescados_mariscos": 1.089, # 011300: Pescados y mariscos
    "frutas": 2.500,            # 011600: Frutas
    "verduras": 2.869,          # 011700: Hortalizas, legumbres, papas y tubérculos
    "bebidas": 1.921,           # 012000: Bebidas no alcohólicas
    "limpieza": 1.319,          # 056100: Bienes para el hogar no duraderos
    "cuidado_personal": 4.791,  # 121000: Cuidado personal
}

# Normalize to sum to 1.0
_total_weight = sum(CPI_WEIGHTS_RAW.values())
CPI_WEIGHTS = {k: v / _total_weight for k, v in CPI_WEIGHTS_RAW.items()}

# Food-only weights (renormalized)
_food_total = sum(v for k, v in CPI_WEIGHTS_RAW.items() if k in FOOD_CATEGORIES)
CPI_WEIGHTS_FOOD = {
    k: v / _food_total
    for k, v in CPI_WEIGHTS_RAW.items()
    if k in FOOD_CATEGORIES
}

# CPI-relevant top-level category IDs per store (food + household essentials)
# These are manually identified from each store's VTEX category tree.
# Filtering to these avoids scraping thousands of irrelevant categories
# (electronics, clothing, furniture, etc.)
FOOD_CATEGORY_IDS = {
    "plazavea": {
        431,    # Abarrotes (staples: rice, oil, sugar, pasta)
        2,      # Bebidas
        77,     # Frutas y Verduras
        814,    # Carnes, Aves y Pescados
        845,    # Lácteos y Huevos
        621,    # Quesos y Fiambres
        210,    # Congelados
        493,    # Panadería y Pastelería
        478,    # Desayunos
        1073,   # Pollo Rostizado y Comidas Preparadas
        1383,   # Mercado Saludable
        399,    # Limpieza
        297,    # Cuidado Personal y Salud
        4180,   # Vinos, licores y cervezas
    },
    # Metro and Wong share Cencosud platform (same category IDs)
    "metro": {
        1700,       # Abarrotes
        1800,       # Aguas y Bebidas
        800,        # Frutas y Verduras
        1001327,    # Carnes, Aves y Pescados
        1001436,    # Lácteos
        1400,       # Embutidos y Fiambres
        1200,       # Congelados
        1001374,    # Panadería y Pastelería
        1001253,    # Desayuno
        1001402,    # Comidas y Rostizados
        1900,       # Limpieza
        2200,       # Higiene, Salud y Belleza
        2100,       # Cervezas, Vinos y Licores
    },
    "wong": {
        1700,       # Abarrotes
        1800,       # Aguas y Bebidas
        800,        # Frutas y Verduras
        1001327,    # Carnes, Aves y Pescados
        1001436,    # Lácteos
        1400,       # Embutidos y Fiambres
        1200,       # Congelados
        1001374,    # Panadería y Pastelería
        1001253,    # Desayuno
        1001402,    # Comidas y Rostizados
        1900,       # Limpieza
        2200,       # Higiene, Salud y Belleza
        2100,       # Cervezas, Vinos y Licores
    },
}

USER_AGENT = "NEXUS/1.0 (academic research)"
REQUEST_DELAY = 1.0  # base seconds between requests
REQUEST_JITTER = 0.5  # random jitter added
MAX_RETRIES = 3
PRODUCTS_PER_PAGE = 50
MAX_PRODUCTS_PER_QUERY = 2500


# ── VTEXClient ────────────────────────────────────────────────────────────────

class VTEXClient:
    """Low-level VTEX catalog API client with rate limiting.

    Parameters
    ----------
    store_domain : str
        Store domain (e.g., 'www.plazavea.com.pe').
    """

    def __init__(self, store_domain: str):
        self.store_domain = store_domain
        self.base_url = f"https://{store_domain}"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        delay = REQUEST_DELAY + random.uniform(0, REQUEST_JITTER)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()

    def _get(self, path: str, params: dict | None = None) -> requests.Response:
        """Make a GET request with rate limiting and exponential backoff."""
        url = f"{self.base_url}{path}"
        for attempt in range(MAX_RETRIES):
            self._rate_limit()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                    logger.warning(
                        "Rate limited (429) on %s, waiting %ds",
                        url, retry_after,
                    )
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "Request failed (%s), retry %d/%d in %ds",
                        e, attempt + 1, MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
        raise requests.exceptions.RetryError(f"Failed after {MAX_RETRIES} retries: {url}")

    def get_category_tree(self) -> list[dict]:
        """Fetch the full category hierarchy (3 levels deep).

        Returns
        -------
        list[dict]
            Nested category tree from VTEX catalog API.
        """
        resp = self._get("/api/catalog_system/pub/category/tree/3")
        return resp.json()

    def get_leaf_categories(self, filter_l1_ids: set[int] | None = None) -> list[dict]:
        """Extract leaf (deepest) categories from the tree.

        Parameters
        ----------
        filter_l1_ids : set[int] or None
            If provided, only include leaves under these top-level category IDs.
            This dramatically reduces scrape time by skipping non-food categories.

        Returns
        -------
        list[dict]
            Flat list of leaf categories with id, name, and parent info.
        """
        tree = self.get_category_tree()
        if filter_l1_ids:
            tree = [n for n in tree if n.get("id") in filter_l1_ids]
        leaves = []
        self._collect_leaves(tree, [], leaves)
        return leaves

    def _collect_leaves(self, nodes: list, parents: list, leaves: list):
        """Recursively collect leaf categories."""
        for node in nodes:
            path = parents + [node.get("name", "")]
            children = node.get("children", [])
            if not children:
                leaves.append({
                    "id": node["id"],
                    "name": node.get("name", ""),
                    "path": " > ".join(path),
                    "l1": parents[0] if len(parents) > 0 else node.get("name", ""),
                    "l2": parents[1] if len(parents) > 1 else "",
                    "l3": node.get("name", "") if len(parents) >= 2 else "",
                })
            else:
                self._collect_leaves(children, path, leaves)

    def search_products(self, category_id: int, offset: int = 0) -> list[dict]:
        """Search products in a category with pagination.

        Parameters
        ----------
        category_id : int
            VTEX category ID.
        offset : int
            Starting product index (0-based).

        Returns
        -------
        list[dict]
            List of product objects from VTEX search API.
        """
        _from = offset
        _to = offset + PRODUCTS_PER_PAGE - 1
        resp = self._get(
            "/api/catalog_system/pub/products/search/",
            params={
                "fq": f"C:/{category_id}/",
                "_from": _from,
                "_to": _to,
            },
        )
        return resp.json()

    def get_all_products_in_category(self, category_id: int) -> list[dict]:
        """Paginate through all products in a category.

        VTEX caps at 2500 products per query. For categories with more,
        results are truncated.

        Returns
        -------
        list[dict]
            All product objects found in this category.
        """
        all_products = []
        offset = 0
        while offset < MAX_PRODUCTS_PER_QUERY:
            products = self.search_products(category_id, offset)
            if not products:
                break
            all_products.extend(products)
            if len(products) < PRODUCTS_PER_PAGE:
                break
            offset += PRODUCTS_PER_PAGE
        return all_products

    @staticmethod
    def extract_price_info(product: dict) -> dict | None:
        """Extract price and metadata from a VTEX product object.

        Parameters
        ----------
        product : dict
            Raw product object from VTEX search API.

        Returns
        -------
        dict or None
            Extracted product info, or None if no valid price found.
        """
        try:
            items = product.get("items", [])
            if not items:
                return None

            item = items[0]
            sellers = item.get("sellers", [])
            if not sellers:
                return None

            offer = sellers[0].get("commertialOffer", {})  # VTEX typo
            price = offer.get("Price", 0)
            list_price = offer.get("ListPrice", 0)
            available = offer.get("IsAvailable", False)

            if price <= 0:
                return None

            return {
                "sku_id": str(item.get("itemId", "")),
                "product_name": product.get("productName", ""),
                "brand": product.get("brand", ""),
                "price": float(price),
                "list_price": float(list_price) if list_price else float(price),
                "ean": item.get("ean", ""),
                "available": bool(available),
                "unit_multiplier": float(item.get("unitMultiplier", 1)),
            }
        except (KeyError, IndexError, TypeError):
            return None


# ── SupermarketScraper ────────────────────────────────────────────────────────

class SupermarketScraper:
    """Daily collection orchestrator for supermarket prices.

    Scrapes all configured stores, saves daily snapshots as parquet,
    and maintains a scrape manifest for tracking.
    """

    def __init__(self):
        RAW_SUPERMARKET_DIR.mkdir(parents=True, exist_ok=True)
        SUPERMARKET_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    def scrape_store(self, store_key: str, target_date: date | None = None) -> pd.DataFrame:
        """Scrape all food-relevant products from a single store.

        Searches at the L1 (top-level) category level since VTEX stores
        typically only index products at L1. Paginates through each food
        category up to the 2500-product VTEX cap.

        Parameters
        ----------
        store_key : str
            Store key from STORES dict (e.g., 'plazavea').
        target_date : date or None
            Date to label the snapshot (default: today).

        Returns
        -------
        pd.DataFrame
            Product-level price data for the store.
        """
        if store_key not in STORES:
            raise ValueError(f"Unknown store: {store_key}. Choose from {list(STORES.keys())}")

        domain = STORES[store_key]
        target_date = target_date or date.today()
        client = VTEXClient(domain)

        logger.info("Scraping %s (%s)...", store_key, domain)

        # Get L1 category tree (filtered to CPI-relevant food categories)
        filter_ids = FOOD_CATEGORY_IDS.get(store_key) or None
        try:
            tree = client.get_category_tree()
        except Exception as e:
            logger.error("Failed to get categories for %s: %s", store_key, e)
            return pd.DataFrame()

        if filter_ids:
            food_l1s = [n for n in tree if n.get("id") in filter_ids]
        else:
            food_l1s = tree

        logger.info("  Scraping %d L1 food categories", len(food_l1s))

        rows = []
        categories_scraped = 0
        seen_skus = set()

        for l1_node in food_l1s:
            l1_name = l1_node.get("name", "")
            l1_id = l1_node["id"]
            try:
                products = client.get_all_products_in_category(l1_id)
                categories_scraped += 1

                n_new = 0
                for product in products:
                    info = VTEXClient.extract_price_info(product)
                    if info is None:
                        continue

                    # Deduplicate across categories
                    sku_key = info["sku_id"]
                    if sku_key in seen_skus:
                        continue
                    seen_skus.add(sku_key)

                    # Extract category from product's own categories if available
                    cat_tree = product.get("categories", [""])
                    cat_path = cat_tree[0] if cat_tree else ""
                    parts = [p.strip() for p in cat_path.strip("/").split("/") if p.strip()]

                    info["date"] = target_date.isoformat()
                    info["store"] = store_key
                    info["category_l1"] = l1_name
                    info["category_l2"] = parts[1] if len(parts) > 1 else ""
                    info["category_l3"] = parts[2] if len(parts) > 2 else ""
                    rows.append(info)
                    n_new += 1

                logger.info(
                    "    %s (ID=%d): %d products fetched, %d new",
                    l1_name, l1_id, len(products), n_new,
                )

            except Exception as e:
                logger.warning(
                    "  Failed category %s (%s): %s",
                    l1_id, l1_name, e,
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            # Ensure column order
            col_order = [
                "date", "store", "sku_id", "product_name", "brand",
                "category_l1", "category_l2", "category_l3",
                "price", "list_price", "ean", "available", "unit_multiplier",
            ]
            df = df[[c for c in col_order if c in df.columns]]

        logger.info(
            "  %s: %d products from %d categories",
            store_key, len(df), categories_scraped,
        )
        return df

    def scrape_all_stores(self, target_date: date | None = None) -> pd.DataFrame:
        """Scrape all configured stores and combine results.

        Parameters
        ----------
        target_date : date or None
            Date to label the snapshot.

        Returns
        -------
        pd.DataFrame
            Combined product-level data across all stores.
        """
        target_date = target_date or date.today()
        all_dfs = []

        for store_key in STORES:
            try:
                df = self.scrape_store(store_key, target_date)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logger.error("Store %s failed: %s", store_key, e)

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)

    def save_daily_snapshot(self, df: pd.DataFrame, target_date: date | None = None) -> Path:
        """Save a daily snapshot as parquet, merging with existing data.

        If a snapshot already exists for this date (e.g., from a previous
        store scrape), merges the data — keeping the latest price per
        (store, sku_id) pair.

        Parameters
        ----------
        df : pd.DataFrame
            Product-level price data.
        target_date : date or None
            Date for the filename.

        Returns
        -------
        Path
            Path to the saved parquet file.
        """
        target_date = target_date or date.today()
        snapshot_path = SUPERMARKET_SNAPSHOTS_DIR / f"{target_date.isoformat()}.parquet"

        # Merge with existing snapshot if present (e.g., different stores scraped separately)
        if snapshot_path.exists():
            existing = pd.read_parquet(snapshot_path)
            df = pd.concat([existing, df], ignore_index=True)
            # Deduplicate: keep last occurrence per (store, sku_id)
            df = df.drop_duplicates(subset=["store", "sku_id"], keep="last")

        df.to_parquet(snapshot_path, index=False)
        logger.info("Snapshot saved: %s (%d rows)", snapshot_path, len(df))

        # Update manifest
        self._update_manifest(df, target_date)
        return snapshot_path

    def _update_manifest(self, df: pd.DataFrame, target_date: date):
        """Append scrape summary to manifest CSV."""
        manifest_path = RAW_SUPERMARKET_DIR / "scrape_manifest.csv"
        write_header = not manifest_path.exists()

        with open(manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "date", "store", "categories_scraped", "products_found", "status",
            ])
            if write_header:
                writer.writeheader()

            for store_key in df["store"].unique() if not df.empty else []:
                store_df = df[df["store"] == store_key]
                writer.writerow({
                    "date": target_date.isoformat(),
                    "store": store_key,
                    "categories_scraped": store_df["category_l1"].nunique(),
                    "products_found": len(store_df),
                    "status": "ok",
                })

    def load_snapshot(self, target_date: date) -> pd.DataFrame:
        """Load a previously saved daily snapshot.

        Parameters
        ----------
        target_date : date
            Date of the snapshot.

        Returns
        -------
        pd.DataFrame
            Product-level data, or empty DataFrame if not found.
        """
        snapshot_path = SUPERMARKET_SNAPSHOTS_DIR / f"{target_date.isoformat()}.parquet"
        if snapshot_path.exists():
            return pd.read_parquet(snapshot_path)
        return pd.DataFrame()

    def list_available_snapshots(self) -> list[date]:
        """List dates with available snapshots.

        Returns
        -------
        list[date]
            Sorted list of available snapshot dates.
        """
        dates = []
        for f in SUPERMARKET_SNAPSHOTS_DIR.glob("*.parquet"):
            try:
                d = date.fromisoformat(f.stem)
                dates.append(d)
            except ValueError:
                continue
        return sorted(dates)


# ── PriceIndexBuilder ─────────────────────────────────────────────────────────

def _classify_product(product_name: str, category_l1: str) -> str:
    """Classify a product into a CPI-relevant category.

    Two-pass classification:
        1. Keyword matching on the product name (case-insensitive)
        2. L1 store category fallback via L1_CATEGORY_MAP

    Parameters
    ----------
    product_name : str
        Product name from VTEX.
    category_l1 : str
        Top-level store category.

    Returns
    -------
    str
        Category key from CATEGORY_KEYWORDS, or 'other'.
    """
    name_lower = product_name.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return cat
    # Fallback: map L1 store category to CPI category
    # Normalize L1 name (strip encoding artifacts)
    l1_clean = category_l1.strip()
    if l1_clean in L1_CATEGORY_MAP:
        return L1_CATEGORY_MAP[l1_clean]
    return "other"


class PriceIndexBuilder:
    """BPP-style price index following Cavallo/Rigobon methodology.

    Implements the weighted Jevons bilateral index:
        1. Match products across days by (store, sku_id) — matched-model
        2. Compute category-level Jevons ratios (geometric mean of p_t/p_{t-1})
        3. Aggregate categories using INEI CPI basket weights (weighted geometric)
        4. Chain-link into cumulative index (base = first day = 100)
        5. Monthly: average daily index values, compute MoM variation

    References:
        Cavallo, A. (2013). "Online and Official Price Indexes: Measuring
        Argentina's Inflation." Journal of Monetary Economics, 60(2), 152-165.

        Cavallo, A. & Rigobon, R. (2016). "The Billion Prices Project: Using
        Online Prices for Measurement and Research." JEP, 30(2), 151-178.
    """

    def __init__(self):
        self.scraper = SupermarketScraper()

    def build_daily_index(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Build daily price indices from available snapshots.

        Parameters
        ----------
        start_date : date or None
            First date to include (default: earliest available).
        end_date : date or None
            Last date to include (default: latest available).

        Returns
        -------
        pd.DataFrame
            Columns: date, index_all, index_food, plus category-level indices.
        """
        available = self.scraper.list_available_snapshots()
        if not available:
            logger.warning("No snapshots available for index construction")
            return pd.DataFrame()

        if start_date:
            available = [d for d in available if d >= start_date]
        if end_date:
            available = [d for d in available if d <= end_date]

        if len(available) < 2:
            logger.warning("Need at least 2 snapshots for price ratios, got %d", len(available))
            return pd.DataFrame()

        # Load all snapshots and classify products
        snapshots = {}
        for d in available:
            df = self.scraper.load_snapshot(d)
            if not df.empty:
                df["cpi_category"] = df.apply(
                    lambda r: _classify_product(
                        r.get("product_name", ""),
                        r.get("category_l1", ""),
                    ),
                    axis=1,
                )
                snapshots[d] = df

        if len(snapshots) < 2:
            return pd.DataFrame()

        sorted_dates = sorted(snapshots.keys())
        base_date = sorted_dates[0]

        # Build daily bilateral ratios
        daily_records = []
        prev_date = sorted_dates[0]

        for curr_date in sorted_dates[1:]:
            prev_df = snapshots[prev_date]
            curr_df = snapshots[curr_date]

            record = self._compute_daily_ratios(prev_df, curr_df, curr_date)
            daily_records.append(record)
            prev_date = curr_date

        if not daily_records:
            return pd.DataFrame()

        daily_df = pd.DataFrame(daily_records)
        daily_df["date"] = pd.to_datetime(daily_df["date"])

        # Chain-link into cumulative index (base = 100)
        ratio_cols = [c for c in daily_df.columns if c.startswith("ratio_")]
        for col in ratio_cols:
            idx_col = col.replace("ratio_", "index_")
            daily_df[idx_col] = 100.0 * daily_df[col].cumprod()

        # Drop ratio columns, keep index columns
        keep_cols = ["date"] + [c for c in daily_df.columns if c.startswith("index_")]
        daily_df = daily_df[keep_cols]

        # Add base date row
        base_row = {"date": pd.Timestamp(base_date)}
        for col in daily_df.columns:
            if col.startswith("index_"):
                base_row[col] = 100.0
        base_df = pd.DataFrame([base_row])
        daily_df = pd.concat([base_df, daily_df], ignore_index=True)
        daily_df = daily_df.sort_values("date").reset_index(drop=True)

        return daily_df

    def _compute_daily_ratios(
        self, prev_df: pd.DataFrame, curr_df: pd.DataFrame, curr_date: date
    ) -> dict:
        """Compute CPI-weighted Jevons price ratios between two days.

        Cavallo methodology:
            1. Match products by (store, sku_id) across days
            2. Trim extreme ratios (0.5 < r < 2.0) to exclude promotions/errors
            3. Compute category-level Jevons index (geometric mean within category)
            4. Aggregate categories using INEI CPI expenditure weights
               (weighted geometric mean: exp(Σ w_k * ln(R_k)))

        Parameters
        ----------
        prev_df : pd.DataFrame
            Previous day's snapshot.
        curr_df : pd.DataFrame
            Current day's snapshot.
        curr_date : date
            Current date.

        Returns
        -------
        dict
            Daily ratio record with weighted aggregate and category-level ratios.
        """
        # Step 1: Match products by (store, sku_id) — matched-model approach
        prev = prev_df[["store", "sku_id", "price", "cpi_category"]].copy()
        prev = prev.rename(columns={"price": "price_prev"})

        curr = curr_df[["store", "sku_id", "price", "cpi_category"]].copy()
        curr = curr.rename(columns={"price": "price_curr"})

        merged = pd.merge(
            prev, curr,
            on=["store", "sku_id", "cpi_category"],
            how="inner",
        )

        # Filter: both prices positive
        merged = merged[
            (merged["price_prev"] > 0) & (merged["price_curr"] > 0)
        ].copy()
        merged["ratio"] = merged["price_curr"] / merged["price_prev"]

        # Step 2: Trim extreme ratios (Cavallo uses similar bounds)
        merged = merged[(merged["ratio"] > 0.5) & (merged["ratio"] < 2.0)]

        record = {"date": curr_date.isoformat()}

        # Step 3: Category-level Jevons indices (geometric mean within category)
        cat_ratios = {}  # {category: jevons_ratio}
        all_categories = set(FOOD_CATEGORIES) | {"limpieza", "cuidado_personal"}

        for cat in all_categories:
            cat_data = merged[merged["cpi_category"] == cat]
            if len(cat_data) >= 5:  # minimum products for reliable estimate
                jevons = float(np.exp(np.log(cat_data["ratio"]).mean()))
                cat_ratios[cat] = jevons
                record[f"ratio_{cat}"] = jevons
            else:
                record[f"ratio_{cat}"] = 1.0

        # Step 4: CPI-weighted aggregate (weighted geometric mean)
        # All-products index: weight by CPI_WEIGHTS
        record["ratio_all"] = self._weighted_geometric_mean(cat_ratios, CPI_WEIGHTS)
        record["n_matched_all"] = len(merged)

        # Food-only index: weight by CPI_WEIGHTS_FOOD
        food_ratios = {k: v for k, v in cat_ratios.items() if k in FOOD_CATEGORIES}
        record["ratio_food"] = self._weighted_geometric_mean(food_ratios, CPI_WEIGHTS_FOOD)
        food_mask = merged["cpi_category"].isin(FOOD_CATEGORIES)
        record["n_matched_food"] = int(food_mask.sum())

        # Unweighted Jevons (for comparison / backward compatibility)
        if len(merged) > 0:
            record["ratio_all_unweighted"] = float(np.exp(np.log(merged["ratio"]).mean()))
        else:
            record["ratio_all_unweighted"] = 1.0

        return record

    @staticmethod
    def _weighted_geometric_mean(
        cat_ratios: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """Compute CPI-weighted geometric mean of category-level ratios.

        Formula: exp(Σ w_k * ln(R_k)) where w_k are renormalized CPI weights
        for categories with enough matched products.

        Parameters
        ----------
        cat_ratios : dict
            {category: jevons_ratio} for categories with sufficient data.
        weights : dict
            {category: weight} from CPI basket (should sum to ~1.0).

        Returns
        -------
        float
            Weighted geometric mean ratio.
        """
        if not cat_ratios:
            return 1.0

        # Renormalize weights to categories with available data
        available_weight = sum(weights.get(k, 0) for k in cat_ratios)
        if available_weight <= 0:
            # Fallback: equal weights
            n = len(cat_ratios)
            log_sum = sum(np.log(r) for r in cat_ratios.values()) / n
            return float(np.exp(log_sum))

        # Weighted geometric mean
        log_sum = sum(
            (weights.get(k, 0) / available_weight) * np.log(r)
            for k, r in cat_ratios.items()
        )
        return float(np.exp(log_sum))

    def build_monthly_index(
        self,
        daily_index: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Aggregate daily index into monthly levels and MoM variations.

        Parameters
        ----------
        daily_index : pd.DataFrame or None
            Daily index from build_daily_index(). If None, builds it.

        Returns
        -------
        pd.DataFrame
            Monthly index with columns: date, *_level, *_var.
        """
        if daily_index is None:
            daily_index = self.build_daily_index()

        if daily_index.empty:
            return pd.DataFrame()

        daily_index["date"] = pd.to_datetime(daily_index["date"])
        daily_index["month"] = daily_index["date"].dt.to_period("M")

        index_cols = [c for c in daily_index.columns if c.startswith("index_")]

        # Monthly average of daily index values
        monthly = daily_index.groupby("month")[index_cols].mean().reset_index()
        monthly["date"] = monthly["month"].dt.to_timestamp()
        monthly = monthly.drop(columns=["month"]).sort_values("date").reset_index(drop=True)

        # Compute MoM variation (%)
        for col in index_cols:
            var_col = col.replace("index_", "var_")
            monthly[var_col] = (monthly[col] / monthly[col].shift(1) - 1) * 100

        return monthly

    def save_monthly_index(self, monthly_df: pd.DataFrame) -> Path:
        """Save monthly index to processed directory.

        Parameters
        ----------
        monthly_df : pd.DataFrame
            Monthly index data.

        Returns
        -------
        Path
            Path to saved parquet file.
        """
        output_path = PROCESSED_NATIONAL_DIR / "supermarket_monthly_prices.parquet"
        PROCESSED_NATIONAL_DIR.mkdir(parents=True, exist_ok=True)
        monthly_df.to_parquet(output_path, index=False)
        logger.info("Monthly index saved: %s (%d rows)", output_path, len(monthly_df))
        return output_path


# ── SupermarketAggregator ─────────────────────────────────────────────────────

# Series definitions for panel integration
SUPERMARKET_SERIES = {
    "SUPERMARKET_ALL_INDEX": {
        "source_col": "index_all",
        "name": "Supermarket price index (all products)",
        "category": "supermarket_prices",
        "is_index": True,
    },
    "SUPERMARKET_ALL_VAR": {
        "source_col": "var_all",
        "name": "Supermarket price variation MoM% (all products)",
        "category": "supermarket_prices",
        "is_index": False,
    },
    "SUPERMARKET_FOOD_INDEX": {
        "source_col": "index_food",
        "name": "Supermarket food price index",
        "category": "supermarket_prices",
        "is_index": True,
    },
    "SUPERMARKET_FOOD_VAR": {
        "source_col": "var_food",
        "name": "Supermarket food price variation MoM%",
        "category": "supermarket_prices",
        "is_index": False,
    },
}

# Add category-level series
for _cat in FOOD_CATEGORIES:
    SUPERMARKET_SERIES[f"SUPERMARKET_{_cat.upper()}_INDEX"] = {
        "source_col": f"index_{_cat}",
        "name": f"Supermarket {_cat.replace('_', ' ')} price index",
        "category": "supermarket_prices",
        "is_index": True,
    }
    SUPERMARKET_SERIES[f"SUPERMARKET_{_cat.upper()}_VAR"] = {
        "source_col": f"var_{_cat}",
        "name": f"Supermarket {_cat.replace('_', ' ')} price variation MoM%",
        "category": "supermarket_prices",
        "is_index": False,
    }


class SupermarketAggregator:
    """Convert monthly supermarket indices to panel long format.

    Matches the existing panel schema for integration with
    panel_builder.py and DFM nowcasting.
    """

    def __init__(self):
        self.builder = PriceIndexBuilder()

    def build_panel_series(
        self,
        monthly_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build panel-format series from monthly index data.

        Parameters
        ----------
        monthly_df : pd.DataFrame or None
            Monthly index data. If None, reads from disk.

        Returns
        -------
        pd.DataFrame
            Long-format panel with standard schema.
        """
        if monthly_df is None:
            path = PROCESSED_NATIONAL_DIR / "supermarket_monthly_prices.parquet"
            if not path.exists():
                logger.warning("No monthly supermarket data found at %s", path)
                return pd.DataFrame()
            monthly_df = pd.read_parquet(path)

        if monthly_df.empty:
            return pd.DataFrame()

        monthly_df["date"] = pd.to_datetime(monthly_df["date"])
        frames = []

        for series_id, spec in SUPERMARKET_SERIES.items():
            source_col = spec["source_col"]
            if source_col not in monthly_df.columns:
                continue

            ts = monthly_df.set_index("date")[source_col].dropna()
            if ts.empty:
                continue

            if spec["is_index"]:
                # Index series: compute SA + transforms
                value_sa = ts.copy()  # daily-aggregated, no seasonality yet
                value_log = np.log(ts)
                value_dlog = np.log(ts).diff()
                value_yoy = (ts / ts.shift(12) - 1) * 100
            else:
                # Variation series: already stationary
                value_sa = ts.copy()
                value_log = pd.Series(np.nan, index=ts.index)
                value_dlog = pd.Series(np.nan, index=ts.index)
                value_yoy = pd.Series(np.nan, index=ts.index)

            frame = pd.DataFrame({
                "date": ts.index,
                "series_id": series_id,
                "series_name": spec["name"],
                "category": spec["category"],
                "value_raw": ts.values,
                "value_sa": value_sa.values,
                "value_log": value_log.values,
                "value_dlog": value_dlog.values,
                "value_yoy": value_yoy.values,
                "source": "SUPERMARKET",
                "frequency_original": "D",
                "publication_lag_days": 1,
            })
            frames.append(frame)

        if not frames:
            return pd.DataFrame()

        panel = pd.concat(frames, ignore_index=True)
        logger.info(
            "Supermarket panel: %d series, %d rows",
            panel["series_id"].nunique(), len(panel),
        )
        return panel
