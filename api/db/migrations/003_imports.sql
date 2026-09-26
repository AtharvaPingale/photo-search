-- Ledger of files auto-imported from camera-roll backup folders. Keyed by content
-- hash so the same photo re-uploaded by a backup app is skipped, and a photo you
-- deleted from the library is not imported again on the next sync.
CREATE TABLE imports (
    file_hash TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    dest_path TEXT,
    status TEXT NOT NULL,          -- imported | duplicate | error
    error TEXT,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
