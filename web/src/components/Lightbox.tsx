import { useCallback, useEffect, useRef, useState } from "react";

import { api, photoUrls, type PhotoDetails } from "../api";
import { formatDate, formatTs } from "../format";
import { downloadPhoto, sharePhoto } from "../share";
import { Icon } from "./Icon";

export type LightboxItem = {
  photo_id: string;
  display_url?: string;
  is_video_frame?: boolean;
  frame_ts?: number | null;
  path?: string;
  caption?: string | null;
};

type Props = {
  items: LightboxItem[];
  index: number;
  onIndex: (i: number) => void;
  onClose: () => void;
  onMoreLike?: (id: string) => void;
  onLessLike?: (id: string) => void;
  onSimilar?: (id: string) => void;
  onNearEnd?: () => void;
};

const SWIPE_PX = 60;

export function Lightbox({ items, index, onIndex, onClose, onMoreLike, onLessLike, onSimilar, onNearEnd }: Props) {
  const item = items[index];
  const [dx, setDx] = useState(0);
  const [info, setInfo] = useState(false);
  const [details, setDetails] = useState<PhotoDetails | null>(null);
  const [fullRes, setFullRes] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const start = useRef<{ x: number; y: number; t: number } | null>(null);

  const go = useCallback(
    (d: number) => {
      const next = index + d;
      if (next >= 0 && next < items.length) {
        onIndex(next);
        setFullRes(false);
      }
      if (next >= items.length - 5) onNearEnd?.();
    },
    [index, items.length, onIndex, onNearEnd],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") go(1);
      else if (e.key === "ArrowLeft") go(-1);
      else if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, onClose]);

  // Lock page scroll while open; the phone's Back gesture closes the lightbox
  // instead of leaving the page. Runs once: re-running would push/pop history.
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    document.body.style.overflow = "hidden";
    history.pushState({ lightbox: true }, "");
    const onPop = () => closeRef.current();
    window.addEventListener("popstate", onPop);
    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("popstate", onPop);
      if (history.state?.lightbox) history.back();
    };
  }, []);

  useEffect(() => {
    setDetails(null);
    if (info && item) api.photo(item.photo_id).then(setDetails).catch(() => setDetails(null));
  }, [info, item]);

  // warm the neighbours so swiping feels instant
  useEffect(() => {
    for (const d of [1, -1, 2]) {
      const n = items[index + d];
      if (n) new Image().src = n.display_url ?? photoUrls(n.photo_id).display;
    }
  }, [index, items]);

  if (!item) return null;
  const src = fullRes ? photoUrls(item.photo_id).original : (item.display_url ?? photoUrls(item.photo_id).display);

  const onTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length !== 1) return; // leave pinch-zoom to the browser
    start.current = { x: e.touches[0].clientX, y: e.touches[0].clientY, t: Date.now() };
  };
  const onTouchMove = (e: React.TouchEvent) => {
    if (!start.current || e.touches.length !== 1) return;
    const mx = e.touches[0].clientX - start.current.x;
    const my = e.touches[0].clientY - start.current.y;
    if (Math.abs(mx) > Math.abs(my)) setDx(mx);
  };
  const onTouchEnd = (e: React.TouchEvent) => {
    if (!start.current) return;
    const end = e.changedTouches[0];
    const mx = end.clientX - start.current.x;
    const my = end.clientY - start.current.y;
    const fast = Date.now() - start.current.t < 250;
    start.current = null;
    setDx(0);
    if (Math.abs(mx) > SWIPE_PX || (fast && Math.abs(mx) > 25)) go(mx < 0 ? 1 : -1);
    else if (my > 120 && Math.abs(my) > Math.abs(mx)) onClose(); // swipe down to dismiss
  };

  const flash = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 1600);
  };

  const name = item.path?.split(/[\\/]/).pop() ?? "photo.jpg";
  return (
    <div className="lightbox" role="dialog" aria-modal="true">
      <div className="lb-top">
        <button className="icon" onClick={onClose} aria-label="close"><Icon name="x" size={24} /></button>
        <span className="lb-count">
          {index + 1} / {items.length}
        </span>
        <button className={`icon${info ? " active" : ""}`} onClick={() => setInfo((v) => !v)} aria-label="details">
          <Icon name="info" size={22} />
        </button>
      </div>

      <div
        className="lb-stage"
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
        style={{ transform: `translateX(${dx}px)`, transition: dx ? "none" : "transform 0.2s" }}
      >
        {item.is_video_frame ? (
          <video
            key={item.photo_id}
            src={`${photoUrls(item.photo_id).original}#t=${item.frame_ts ?? 0}`}
            controls
            playsInline
            autoPlay
            muted
            poster={item.display_url}
          />
        ) : (
          <img key={src} src={src} alt={item.caption ?? ""} onDoubleClick={() => setFullRes(true)} />
        )}
        <button className="lb-nav prev" onClick={() => go(-1)} disabled={index === 0} aria-label="previous">
          <Icon name="left" size={28} />
        </button>
        <button className="lb-nav next" onClick={() => go(1)} disabled={index >= items.length - 1} aria-label="next">
          <Icon name="right" size={28} />
        </button>
      </div>

      {info && (
        <div className="lb-info">
          {!details ? (
            <div className="spinner small" />
          ) : (
            <Details d={details} />
          )}
        </div>
      )}

      <div className="lb-actions">
        {onMoreLike && (
          <button onClick={() => { onMoreLike(item.photo_id); flash("Showing more like this"); }} title="More like this">
            <Icon name="up" /><span>More</span>
          </button>
        )}
        {onLessLike && (
          <button onClick={() => { onLessLike(item.photo_id); flash("Showing fewer like this"); }} title="Less like this">
            <Icon name="down" /><span>Less</span>
          </button>
        )}
        {onSimilar && (
          <button onClick={() => onSimilar(item.photo_id)} title="Find similar photos">
            <Icon name="similar" /><span>Similar</span>
          </button>
        )}
        {!item.is_video_frame && !fullRes && (
          <button onClick={() => setFullRes(true)} title="Load full resolution">
            <Icon name="full" /><span>Full res</span>
          </button>
        )}
        <button
          onClick={async () => {
            const r = await sharePhoto(item.photo_id, name);
            if (r === "downloaded") flash("Downloading");
          }}
          title="Share"
        >
          <Icon name="share" /><span>Share</span>
        </button>
        <button onClick={() => { downloadPhoto(item.photo_id); flash("Downloading original"); }} title="Download original">
          <Icon name="download" /><span>Save</span>
        </button>
      </div>
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

function Details({ d }: { d: PhotoDetails }) {
  const exposure = [
    d.focal_length ? `${Math.round(d.focal_length)}mm` : null,
    d.aperture ? `ƒ/${d.aperture}` : null,
    d.shutter ? `${d.shutter}s` : null,
    d.iso ? `ISO ${d.iso}` : null,
  ].filter(Boolean);
  const people = [...new Set(d.faces.map((f) => f.name).filter(Boolean))];
  return (
    <dl>
      <dt>Taken</dt>
      <dd>{formatDate(d.taken_at, true) || "unknown"}</dd>
      {(d.place_name || d.country) && (
        <>
          <dt>Place</dt>
          <dd>{[d.place_name, d.admin1, d.country].filter(Boolean).join(", ")}</dd>
        </>
      )}
      {(d.camera || d.lens) && (
        <>
          <dt>Camera</dt>
          <dd>
            {[d.camera, d.lens].filter(Boolean).join(" · ")}
            {exposure.length > 0 && <div className="muted">{exposure.join("  ")}</div>}
          </dd>
        </>
      )}
      {d.width && (
        <>
          <dt>Size</dt>
          <dd>
            {d.width} × {d.height} {d.format ? `· ${d.format.toUpperCase()}` : ""}
            {d.is_video_frame && d.frame_ts != null ? ` · frame at ${formatTs(d.frame_ts)}` : ""}
          </dd>
        </>
      )}
      {people.length > 0 && (
        <>
          <dt>People</dt>
          <dd>{people.join(", ")}</dd>
        </>
      )}
      {d.captions[0] && (
        <>
          <dt>Caption</dt>
          <dd>{d.captions[0].caption}</dd>
        </>
      )}
      {d.ocr_text && (
        <>
          <dt>Text in photo</dt>
          <dd className="pre">{d.ocr_text}</dd>
        </>
      )}
      {d.keywords.length > 0 && (
        <>
          <dt>Keywords</dt>
          <dd>{d.keywords.join(", ")}</dd>
        </>
      )}
      {d.albums.length > 0 && (
        <>
          <dt>Albums</dt>
          <dd>
            {d.albums.map((a) => (
              <a key={a.id} href={`#/albums/${a.id}`}>
                {a.title}
              </a>
            ))}
          </dd>
        </>
      )}
      <dt>File</dt>
      <dd className="muted small">{d.path}</dd>
    </dl>
  );
}
