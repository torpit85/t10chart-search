#!/usr/bin/env python3
from __future__ import annotations
import sqlite3
from pathlib import Path

DB = Path("t10.sqlite")

def clean_part(s: str | None) -> list[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    # ignore everything after "|"
    s = s.split("|", 1)[0].strip()
    if not s:
        return []
    # split on "/" into tokens
    parts = [p.strip() for p in s.split("/") if p.strip()]
    return parts

def normalize(im1: str | None, im2: str | None) -> tuple[str | None, str | None]:
    tokens = clean_part(im1) + clean_part(im2)

    # de-dupe preserving order
    seen = set()
    uniq: list[str] = []
    for t in tokens:
        key = t.casefold()
        if key not in seen:
            seen.add(key)
            uniq.append(t)

    if not uniq:
        return None, None
    if len(uniq) == 1:
        return uniq[0], None
    return uniq[0], uniq[1]  # only two slots

def main() -> None:
    if not DB.exists():
        raise SystemExit(f"DB not found: {DB}")

    con = sqlite3.connect(DB)
    cur = con.cursor()

    cur.execute("SELECT id, imprint_1, imprint_2 FROM t10_entry")
    rows = cur.fetchall()

    changed = 0
    for row_id, im1, im2 in rows:
        n1, n2 = normalize(im1, im2)
        if (im1 or None) != n1 or (im2 or None) != n2:
            cur.execute(
                "UPDATE t10_entry SET imprint_1 = ?, imprint_2 = ? WHERE id = ?",
                (n1, n2, row_id),
            )
            changed += 1

    con.commit()
    con.close()
    print(f"Normalized imprints on {changed} row(s).")

if __name__ == "__main__":
    main()
