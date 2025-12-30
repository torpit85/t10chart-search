PRAGMA foreign_keys=OFF;
BEGIN;

-- Rename old table
ALTER TABLE t10_entry RENAME TO t10_entry_old;

-- Recreate with a 'pos' column that is unique per week.
-- rank can now repeat (ties), but pos cannot.
CREATE TABLE t10_entry (
  id INTEGER PRIMARY KEY,
  week_ending TEXT NOT NULL,
  week_number INTEGER NOT NULL,
  rank INTEGER NOT NULL,
  pos INTEGER NOT NULL,
  last_week TEXT,
  show_id INTEGER NOT NULL,
  raw_title TEXT NOT NULL,
  imprint_1 TEXT,
  imprint_2 TEXT,
  gross_millions REAL,
  FOREIGN KEY(show_id) REFERENCES show(show_id) ON DELETE CASCADE,
  UNIQUE(week_ending, pos)
);

-- Copy existing data; generate pos per week in rank order.
-- SQLite supports window functions in modern versions.
INSERT INTO t10_entry(
  id, week_ending, week_number, rank, pos, last_week,
  show_id, raw_title, imprint_1, imprint_2, gross_millions
)
SELECT
  id, week_ending, week_number, rank,
  ROW_NUMBER() OVER (PARTITION BY week_ending ORDER BY rank, id) AS pos,
  last_week, show_id, raw_title, imprint_1, imprint_2, gross_millions
FROM t10_entry_old;

-- Drop the old table
DROP TABLE t10_entry_old;

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_t10_entry_week ON t10_entry(week_ending);
CREATE INDEX IF NOT EXISTS idx_t10_entry_weekrank ON t10_entry(week_ending, rank);
CREATE INDEX IF NOT EXISTS idx_t10_entry_weeknum ON t10_entry(week_number);

-- Rebuild FTS so MATCH continues to work with new row ids.
DROP TABLE IF EXISTS t10_fts;

CREATE VIRTUAL TABLE t10_fts USING fts5(
  raw_title,
  imprint_1,
  imprint_2,
  content='t10_entry',
  content_rowid='id'
);

CREATE TRIGGER t10_entry_ai AFTER INSERT ON t10_entry BEGIN
  INSERT INTO t10_fts(rowid, raw_title, imprint_1, imprint_2)
  VALUES (new.id, new.raw_title, new.imprint_1, new.imprint_2);
END;

CREATE TRIGGER t10_entry_ad AFTER DELETE ON t10_entry BEGIN
  INSERT INTO t10_fts(t10_fts, rowid, raw_title, imprint_1, imprint_2)
  VALUES('delete', old.id, old.raw_title, old.imprint_1, old.imprint_2);
END;

CREATE TRIGGER t10_entry_au AFTER UPDATE ON t10_entry BEGIN
  INSERT INTO t10_fts(t10_fts, rowid, raw_title, imprint_1, imprint_2)
  VALUES('delete', old.id, old.raw_title, old.imprint_1, old.imprint_2);
  INSERT INTO t10_fts(rowid, raw_title, imprint_1, imprint_2)
  VALUES (new.id, new.raw_title, new.imprint_1, new.imprint_2);
END;

-- Populate FTS from content table
INSERT INTO t10_fts(t10_fts) VALUES('rebuild');

COMMIT;
PRAGMA foreign_keys=ON;
