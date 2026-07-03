from fastapi import FastAPI, HTTPException
from pathlib import Path
import json

app = FastAPI(title="RetailX Product Catalog API")

# Resolve project root safely
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "source" / "product_catalog"

def get_latest_catalog_file():
    """
    We need to find the most recent catalog file so the API always serves fresh data.
    The files are prefixed with YYYYMMDD, so we can just grab the max filename.
    """
    if not SOURCE_DIR.exists():
        return None
    
    # Check for all JSON files that match our catalog pattern
    catalog_files = list(SOURCE_DIR.glob("*_product_catalog.json"))
    if not catalog_files:
        return None
    
    # Sorting by the 'date' part of the filename (YYYYMMDD) to get the newest one
    latest_file = max(catalog_files, key=lambda p: p.name.split('_')[0])
    return latest_file

@app.get("/products")
def get_products():
    try:
        data_file = get_latest_catalog_file()
        if not data_file or not data_file.exists():
            raise HTTPException(status_code=404, detail="No product catalog files found in source")

        print(f"API serving data from: {data_file}")
        with open(data_file, "r") as f:
            products = json.load(f)

        return {
            "count": len(products),
            "products": products
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
