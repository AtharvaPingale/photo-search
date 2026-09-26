-- "More like this" / "less like this" clicks from search. Used as relevance
-- feedback at query time and, later, as (query, photo) training pairs for the
-- fine-tuned CLIP. Never mixed with the eval set (build_pairs.py filters out
-- anything resembling an eval query).
CREATE TABLE feedback (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    photo_id UUID NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    label SMALLINT NOT NULL CHECK (label IN (-1, 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX feedback_photo_idx ON feedback (photo_id);
