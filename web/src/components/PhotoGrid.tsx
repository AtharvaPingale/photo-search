import type { RefObject } from "react";

import type { GridPhoto } from "../api";
import { formatTs } from "../format";
import { Icon } from "./Icon";

type Props = {
  photos: GridPhoto[];
  onOpen: (index: number) => void;
  sentinel?: RefObject<HTMLDivElement | null>;
  loading?: boolean;
  selected?: Set<string>;
  highlight?: (p: GridPhoto) => "relevant" | "highly" | null;
};

export function PhotoGrid({ photos, onOpen, sentinel, loading, selected, highlight }: Props) {
  return (
    <>
      <div className="grid">
        {photos.map((p, i) => {
          const mark = highlight?.(p);
          return (
            <button
              key={`${p.photo_id}-${i}`}
              className={`tile${selected?.has(p.photo_id) ? " selected" : ""}${mark ? ` mark-${mark}` : ""}`}
              onClick={() => onOpen(i)}
              aria-label="open photo"
            >
              <img src={p.thumb_url} loading="lazy" decoding="async" alt="" draggable={false} />
              {p.is_video_frame && (
                <span className="badge video"><Icon name="play" size={10} /> {p.frame_ts != null ? formatTs(p.frame_ts) : ""}</span>
              )}
              {p.badge && <span className="badge">{p.badge}</span>}
            </button>
          );
        })}
      </div>
      {sentinel && <div ref={sentinel} className="sentinel" />}
      {loading && <div className="spinner" aria-label="loading" />}
    </>
  );
}
