import sqlite3
from pathlib import Path
import pytest
from scripts.web_preview_data import prepare_preview_data


def test_prepare_preview_data_copies_and_redirects(tmp_path):
    src_dir, dest_dir = tmp_path / "prod", tmp_path / "preview"
    src_dir.mkdir()
    (src_dir / ".env").write_text("ALPACA_SECRET_KEY=secret\n")

    cartera_db = src_dir / "cartera.db"
    with sqlite3.connect(cartera_db) as conn:
        conn.execute("CREATE TABLE orders (id INT PRIMARY KEY, symbol TEXT)")
        conn.execute("INSERT INTO orders VALUES (1, 'SPY')")

    maestro_db = src_dir / "maestro.db"
    orig_url = f"sqlite:///{cartera_db.resolve()}"
    with sqlite3.connect(maestro_db) as conn:
        conn.execute("CREATE TABLE accounts (id TEXT PRIMARY KEY, database_url TEXT, status TEXT)")
        conn.execute("INSERT INTO accounts VALUES ('m4', ?, 'active')", (orig_url,))

    dest_maestro = prepare_preview_data(maestro_db, dest_dir)
    dest_cartera = dest_dir / "cartera.db"

    # Redirección efectiva y conservación de filas
    with sqlite3.connect(dest_maestro) as conn:
        new_url = conn.execute("SELECT database_url FROM accounts WHERE id = 'm4'").fetchone()[0]
    assert new_url != orig_url, "Falla si se omite la redirección de database_url"
    assert new_url == f"sqlite:///{dest_cartera.resolve()}"

    with sqlite3.connect(dest_cartera) as conn:
        assert conn.execute("SELECT symbol FROM orders WHERE id = 1").fetchone()[0] == "SPY"
        conn.execute("INSERT INTO orders VALUES (2, 'QQQ')")

    # Modificar copia no cambia el origen
    with sqlite3.connect(cartera_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
    with sqlite3.connect(dest_maestro) as conn:
        conn.execute("UPDATE accounts SET status = 'archived' WHERE id = 'm4'")
    with sqlite3.connect(maestro_db) as conn:
        assert conn.execute("SELECT status FROM accounts WHERE id = 'm4'").fetchone()[0] == "active"

    # No se copian secretos
    assert not (dest_dir / ".env").exists()
    assert sorted(p.name for p in dest_dir.iterdir()) == ["cartera.db", "maestro.db"]


def test_prepare_preview_data_missing_db_raises(tmp_path):
    dest_dir = tmp_path / "preview"
    with pytest.raises(FileNotFoundError):
        prepare_preview_data(tmp_path / "missing.db", dest_dir)
