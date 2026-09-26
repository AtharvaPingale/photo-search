-- Core schema. Timestamps in `taken_at` are the camera's wall-clock time stored
-- as if it were UTC (EXIF rarely carries an offset). All sessions run with
-- TimeZone=UTC, so "photos from June 2025" compares wall-clock to wall-clock.

CREATE EXTENSION IF NOT EXISTS vector;

-- array_to_string is only STABLE, which a generated column can't use.
CREATE FUNCTION immutable_array_join(TEXT[]) RETURNS TEXT
    LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT coalesce(array_to_string($1, ' '), '') $$;

CREATE TABLE photos (
    id UUID PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    size_bytes BIGINT,
    mtime DOUBLE PRECISION,
    media_type TEXT NOT NULL DEFAULT 'image',   -- image | video
    format TEXT,                                 -- jpeg, heic, raw, png, mp4, ...
    taken_at TIMESTAMPTZ,
    tz_offset_min INT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    place_name TEXT,
    admin1 TEXT,
    country TEXT,
    camera TEXT,
    lens TEXT,
    focal_length REAL,
    focal_length_35mm REAL,
    aperture REAL,
    iso INT,
    shutter TEXT,
    exposure_s REAL,
    width INT,
    height INT,
    orientation INT,
    keywords TEXT[] NOT NULL DEFAULT '{}',      -- Lightroom / XMP dc:subject
    keywords_tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', immutable_array_join(keywords))) STORED,
    is_video_frame BOOLEAN NOT NULL DEFAULT FALSE,
    source_video_id UUID NULL REFERENCES photos(id) ON DELETE CASCADE,
    frame_ts REAL NULL,
    duration_s REAL NULL,
    thumb_path TEXT,
    phash TEXT,
    sharpness REAL,
    faces_indexed_at TIMESTAMPTZ,
    frames_indexed_at TIMESTAMPTZ,          -- videos: scene frames extracted
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    error TEXT
);

CREATE INDEX photos_file_hash_idx ON photos (file_hash);
CREATE INDEX photos_taken_at_idx ON photos (taken_at);
CREATE INDEX photos_place_idx ON photos (place_name);
CREATE INDEX photos_camera_idx ON photos (camera);
CREATE INDEX photos_lens_idx ON photos (lens);
CREATE INDEX photos_focal_idx ON photos (focal_length);
CREATE INDEX photos_source_video_idx ON photos (source_video_id) WHERE source_video_id IS NOT NULL;
CREATE INDEX photos_keywords_idx ON photos USING gin (keywords);
CREATE INDEX photos_keywords_tsv_idx ON photos USING gin (keywords_tsv);

-- One row per (photo, model) so base and fine-tuned embeddings live side by side.
-- The column is dimensionless; each model gets a partial HNSW index on a typed
-- cast (created by api.db.vector_index.ensure_image_index), which is how pgvector
-- supports several dimensions in one table.
CREATE TABLE image_embeddings (
    photo_id UUID NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    embedding VECTOR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (photo_id, model)
);

CREATE TABLE captions (
    photo_id UUID NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    caption TEXT NOT NULL,
    embed_model TEXT,
    caption_embedding VECTOR(384),
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', caption)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (photo_id, model)
);
CREATE INDEX captions_tsv_idx ON captions USING gin (tsv);
CREATE INDEX captions_embedding_idx ON captions USING hnsw (caption_embedding vector_cosine_ops);

-- A row exists for every photo the OCR stage has looked at, including ones the
-- text gate skipped (text = ''), so re-runs don't redo them.
CREATE TABLE ocr_text (
    photo_id UUID PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
    engine TEXT NOT NULL,
    gate_score REAL,
    ran_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    text TEXT NOT NULL DEFAULT '',
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ocr_tsv_idx ON ocr_text USING gin (tsv);

CREATE TABLE faces (
    id UUID PRIMARY KEY,
    photo_id UUID NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    bbox INT[4] NOT NULL,
    det_score REAL,
    embedding VECTOR(512) NOT NULL,
    cluster_id INT NULL,
    -- set when a person was assigned by hand (merge/split/move); reclustering keeps it
    manual BOOLEAN NOT NULL DEFAULT FALSE,
    crop_path TEXT
);
CREATE INDEX faces_photo_idx ON faces (photo_id);
CREATE INDEX faces_cluster_idx ON faces (cluster_id);

CREATE TABLE people (
    cluster_id INT PRIMARY KEY,
    name TEXT NULL,
    aliases TEXT[] NOT NULL DEFAULT '{}',   -- e.g. {'me'} for the library owner
    hidden BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE albums (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT,
    start_at TIMESTAMPTZ,
    end_at TIMESTAMPTZ,
    place_name TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    cover_photo_id UUID REFERENCES photos(id) ON DELETE SET NULL,
    auto BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE album_photos (
    album_id UUID NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
    photo_id UUID NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    PRIMARY KEY (album_id, photo_id)
);

-- Near-duplicate groups and bursts. One photo belongs to at most one group per kind.
CREATE TABLE photo_groups (
    kind TEXT NOT NULL,          -- duplicate | burst
    group_id INT NOT NULL,
    photo_id UUID NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    score REAL,                  -- sharpness for bursts, similarity for duplicates
    is_best BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (kind, photo_id)
);
CREATE INDEX photo_groups_group_idx ON photo_groups (kind, group_id);
