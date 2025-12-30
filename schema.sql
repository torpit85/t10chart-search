
PRAGMA foreign_keys = ON;

-- Canonical shows table (normalized)
CREATE TABLE IF NOT EXISTS show (
  show_id INTEGER PRIMARY KEY,
  canonical_title TEXT NOT NULL UNIQUE
);

-- Title aliases map raw titles to canonical shows
CREATE TABLE IF NOT EXISTS show_alias (
  alias_id INTEGER PRIMARY KEY,
  alias_title TEXT NOT NULL UNIQUE,
  show_id INTEGER NOT NULL,
  FOREIGN KEY(show_id) REFERENCES show(show_id) ON DELETE CASCADE
);

-- Chart entries (raw facts)
CREATE TABLE IF NOT EXISTS t10_entry (
  id INTEGER PRIMARY KEY,
  week_number INTEGER NOT NULL,
  week_ending TEXT NOT NULL,              -- YYYY-MM-DD
  rank INTEGER NOT NULL CHECK(rank >= 1),
  last_week TEXT,                         -- allows NEW / RE / numeric text
  show_id INTEGER NOT NULL,
  raw_title TEXT NOT NULL,                -- as printed on chart
  imprint_1 TEXT,
  imprint_2 TEXT,
  gross_millions REAL,
  source_file TEXT,                       -- provenance
  FOREIGN KEY(show_id) REFERENCES show(show_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_t10_date_rank ON t10_entry(week_ending, rank);
CREATE INDEX IF NOT EXISTS idx_t10_week ON t10_entry(week_number);
CREATE INDEX IF NOT EXISTS idx_t10_date ON t10_entry(week_ending);
CREATE INDEX IF NOT EXISTS idx_t10_rank ON t10_entry(rank);
CREATE INDEX IF NOT EXISTS idx_t10_show_id ON t10_entry(show_id);
CREATE INDEX IF NOT EXISTS idx_t10_imprint ON t10_entry(imprint_1, imprint_2);

-- Full-text search over show title + imprints (search engine)
CREATE VIRTUAL TABLE IF NOT EXISTS t10_fts
USING fts5(
  title,
  imprint_1,
  imprint_2,
  content='t10_entry',
  content_rowid='id'
);

-- Keep FTS in sync
CREATE TRIGGER IF NOT EXISTS t10_ai AFTER INSERT ON t10_entry BEGIN
  INSERT INTO t10_fts(rowid, title, imprint_1, imprint_2)
  VALUES (new.id, new.raw_title, new.imprint_1, new.imprint_2);
END;

CREATE TRIGGER IF NOT EXISTS t10_ad AFTER DELETE ON t10_entry BEGIN
  INSERT INTO t10_fts(t10_fts, rowid, title, imprint_1, imprint_2)
  VALUES ('delete', old.id, old.raw_title, old.imprint_1, old.imprint_2);
END;

CREATE TRIGGER IF NOT EXISTS t10_au AFTER UPDATE ON t10_entry BEGIN
  INSERT INTO t10_fts(t10_fts, rowid, title, imprint_1, imprint_2)
  VALUES ('delete', old.id, old.raw_title, old.imprint_1, old.imprint_2);
  INSERT INTO t10_fts(rowid, title, imprint_1, imprint_2)
  VALUES (new.id, new.raw_title, new.imprint_1, new.imprint_2);
END;

-- Helpful derived views (no stored "total weeks")
CREATE VIEW IF NOT EXISTS v_entry AS
SELECT
  e.id,
  e.week_number,
  e.week_ending,
  e.rank,
  e.last_week,
  s.canonical_title,
  e.raw_title,
  e.imprint_1,
  e.imprint_2,
  e.gross_millions,
  e.source_file
FROM t10_entry e
JOIN show s ON s.show_id = e.show_id;

CREATE VIEW IF NOT EXISTS v_show_stats AS
SELECT
  s.show_id,
  s.canonical_title,
  COUNT(*) AS weeks_on_chart,
  MIN(e.week_ending) AS first_appearance,
  MAX(e.week_ending) AS last_appearance,
  MIN(e.rank) AS peak_rank,
  AVG(e.rank) AS avg_rank,
  SUM(COALESCE(e.gross_millions,0)) AS total_gross_millions,
  AVG(e.gross_millions) AS avg_gross_millions
FROM show s
JOIN t10_entry e ON e.show_id = s.show_id
GROUP BY s.show_id, s.canonical_title;

CREATE VIEW IF NOT EXISTS v_company_stats AS
SELECT
  COALESCE(imprint_1,'(Unknown)') AS company,
  COUNT(*) AS entries,
  COUNT(DISTINCT show_id) AS unique_shows,
  SUM(COALESCE(gross_millions,0)) AS total_gross_millions,
  AVG(gross_millions) AS avg_gross_millions
FROM t10_entry
GROUP BY COALESCE(imprint_1,'(Unknown)');
