"""Copy maestro and account SQLite databases for isolated web preview."""
import argparse
from pathlib import Path
import sqlite3


def _backup_sqlite(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{src.resolve()}?mode=ro", uri=True) as s, sqlite3.connect(dst) as d:
        s.backup(d)


def _extract_path(url: str) -> Path:
    raw = url.removeprefix("sqlite:///") if url.startswith("sqlite:///") else url.removeprefix("sqlite://")
    p = Path(raw)
    return Path("/" + raw) if not p.is_absolute() and Path("/" + raw).is_file() else p


def _format_url(orig: str, target: Path) -> str:
    path = str(target.resolve())
    return f"sqlite:///{path}" if orig.startswith("sqlite:///") else path


def prepare_preview_data(source_maestro: Path | str, dest_dir: Path | str) -> Path:
    source_maestro, dest_dir = Path(source_maestro).resolve(), Path(dest_dir).resolve()
    if not source_maestro.is_file():
        raise FileNotFoundError(f"Source maestro not found: {source_maestro}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_maestro = dest_dir / source_maestro.name
    if dest_maestro == source_maestro:
        raise ValueError("Destination maestro cannot be source maestro")

    _backup_sqlite(source_maestro, dest_maestro)
    copied: dict[Path, Path] = {}
    with sqlite3.connect(dest_maestro) as conn:
        has_accounts = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='accounts'").fetchone()
        for aid, url in conn.execute("SELECT id, database_url FROM accounts").fetchall() if has_accounts else []:
            if not url:
                continue
            src = _extract_path(url)
            src = src.resolve() if src.is_absolute() else (source_maestro.parent / src).resolve()
            if not src.is_file():
                raise FileNotFoundError(f"Account '{aid}' database not found: {src}")
            target = copied.setdefault(src, dest_dir / src.name)
            if not target.is_file():
                _backup_sqlite(src, target)
            conn.execute("UPDATE accounts SET database_url = ? WHERE id = ?", (_format_url(url, target), aid))
        conn.commit()
    return dest_maestro


def main():
    p = argparse.ArgumentParser(description="Prepare isolated SQLite copies for preview.")
    p.add_argument("dest_dir", type=Path, help="Target directory for preview data")
    p.add_argument("--source-maestro", type=Path, default=Path("/home/azureuser/web-ops/maestro.db"))
    args = p.parse_args()
    print(f"Isolated preview data prepared at {prepare_preview_data(args.source_maestro, args.dest_dir)}")


if __name__ == "__main__":
    main()
