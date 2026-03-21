import os

from databricks.sdk import WorkspaceClient

CATALOG = os.getenv("UC_CATALOG", "main")
SCHEMA = os.getenv("UC_SCHEMA", "mlops_e2e")

_ws: WorkspaceClient | None = None


def _get_client() -> WorkspaceClient:
    global _ws
    if _ws is None:
        _ws = WorkspaceClient()
    return _ws


def get_table_info(table_name: str):
    ws = _get_client()
    full_name = f"{CATALOG}.{SCHEMA}.{table_name}"
    try:
        table = ws.tables.get(full_name)
        return {
            "name": table.full_name,
            "table_type": str(table.table_type),
            "created_at": table.created_at,
            "updated_at": table.updated_at,
            "columns": [{"name": c.name, "type": str(c.type_name)} for c in (table.columns or [])],
        }
    except Exception:
        return None


def list_tables():
    ws = _get_client()
    try:
        tables = list(ws.tables.list(catalog_name=CATALOG, schema_name=SCHEMA))
        return [
            {
                "name": t.full_name,
                "table_type": str(t.table_type),
            }
            for t in tables
        ]
    except Exception:
        return []
