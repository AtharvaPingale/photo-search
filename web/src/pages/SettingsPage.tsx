import { useEffect, useState } from "react";

import { api } from "../api";
import { formatDate } from "../format";

export function SettingsPage({ authRequired }: { authRequired: boolean }) {
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  useEffect(() => void api.stats().then(setStats).catch(() => setStats({})), []);
  const standalone = window.matchMedia("(display-mode: standalone)").matches;
  const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
  const n = (k: string) => (stats?.[k] as number | undefined)?.toLocaleString() ?? "–";
  return (
    <div className="page">
      <header className="page-header">
        <h1>Library</h1>
      </header>
      {stats === null ? (
        <div className="spinner" />
      ) : (
        <dl className="stats">
          <dt>Photos</dt>
          <dd>{n("photos")}</dd>
          <dt>Videos</dt>
          <dd>{n("videos")}</dd>
          <dt>Captioned</dt>
          <dd>{n("captions")}</dd>
          <dt>OCR'd</dt>
          <dd>{n("ocr")}</dd>
          <dt>Faces</dt>
          <dd>{n("faces")} ({n("named_people")} named people)</dd>
          <dt>Albums</dt>
          <dd>{n("albums")}</dd>
          <dt>Span</dt>
          <dd>
            {formatDate(stats.first_photo as string)} – {formatDate(stats.last_photo as string)}
          </dd>
          <dt>Embeddings</dt>
          <dd>
            {Object.entries((stats.embeddings as Record<string, number>) ?? {})
              .map(([m, c]) => `${m}: ${c.toLocaleString()}`)
              .join(" · ") || "–"}
          </dd>
        </dl>
      )}
      {!standalone && (
        <section className="card">
          <h3>Install on your phone</h3>
          <p className="small">
            {ios
              ? "In Safari, tap Share, then “Add to Home Screen”. It opens full screen like an app."
              : "Use your browser menu's “Install app” / “Add to Home screen”."}
          </p>
        </section>
      )}
      <section className="card">
        <h3>Links</h3>
        <p className="small">
          <a href="#/organize">Duplicates &amp; bursts</a> · <a href="#/eval">Eval labelling &amp; reports</a> ·{" "}
          <a href="/docs" target="_blank" rel="noreferrer">
            API docs
          </a>
        </p>
      </section>
      {authRequired && (
        <button
          className="ghost danger"
          onClick={async () => {
            await api.logout();
            window.location.reload();
          }}
        >
          Log out this device
        </button>
      )}
    </div>
  );
}
