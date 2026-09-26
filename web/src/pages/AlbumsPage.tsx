import { useEffect, useState } from "react";

import { api, type Album, type Group } from "../api";
import { Lightbox } from "../components/Lightbox";
import { PhotoGrid } from "../components/PhotoGrid";
import { formatDate } from "../format";
import { navigate, type Route } from "../router";

export function AlbumsPage({ route }: { route: Route }) {
  return route.path[1] ? <AlbumDetail id={route.path[1]} /> : <AlbumList />;
}

function AlbumList() {
  const [albums, setAlbums] = useState<Album[] | null>(null);
  useEffect(() => void api.albums().then(setAlbums).catch(() => setAlbums([])), []);
  return (
    <div className="page">
      <header className="page-header">
        <h1>Albums</h1>
      </header>
      {albums === null && <div className="spinner" />}
      {albums?.length === 0 && <div className="empty">No albums yet. Run <code>photo-search organize</code>.</div>}
      <div className="album-grid">
        {albums?.map((a) => (
          <button key={a.id} className="album" onClick={() => navigate(`albums/${a.id}`)}>
            {a.cover_url && <img src={a.cover_url} loading="lazy" alt="" />}
            <div className="album-text">
              <b>{a.title}</b>
              <span className="muted small">
                {formatDate(a.start_at)}
                {a.end_at && formatDate(a.end_at) !== formatDate(a.start_at) ? ` – ${formatDate(a.end_at)}` : ""} · {a.n_photos}
              </span>
              {a.summary && <span className="small">{a.summary}</span>}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function AlbumDetail({ id }: { id: string }) {
  const [album, setAlbum] = useState<Awaited<ReturnType<typeof api.album>> | null>(null);
  const [open, setOpen] = useState<number | null>(null);
  useEffect(() => void api.album(id).then(setAlbum), [id]);
  const photos = (album?.photos ?? []).map((p) => ({ photo_id: p.id, thumb_url: p.thumb_url, is_video_frame: p.is_video_frame }));
  return (
    <div className="page">
      <header className="page-header">
        <button className="ghost" onClick={() => navigate("albums")}>
          ‹ Albums
        </button>
        <h1>{album?.title ?? "…"}</h1>
      </header>
      {album?.summary && <p className="muted">{album.summary}</p>}
      <PhotoGrid photos={photos} onOpen={setOpen} loading={album === null} />
      {open !== null && <Lightbox items={photos} index={open} onIndex={setOpen} onClose={() => setOpen(null)} />}
    </div>
  );
}

export function OrganizePage({ route }: { route: Route }) {
  const kind = route.params.get("kind") === "burst" ? "burst" : "duplicate";
  const [groups, setGroups] = useState<Group[] | null>(null);
  const [open, setOpen] = useState<{ g: number; i: number } | null>(null);
  useEffect(() => {
    setGroups(null);
    void api.groups(kind).then(setGroups).catch(() => setGroups([]));
  }, [kind]);
  return (
    <div className="page">
      <header className="page-header">
        <h1>Organize</h1>
        <div className="tabs">
          <button className={kind === "duplicate" ? "active" : ""} onClick={() => navigate("organize", { kind: "duplicate" })}>
            Duplicates
          </button>
          <button className={kind === "burst" ? "active" : ""} onClick={() => navigate("organize", { kind: "burst" })}>
            Bursts
          </button>
        </div>
      </header>
      <p className="muted small">
        {kind === "duplicate"
          ? "Near-identical copies (same content by CLIP and by perceptual hash). ★ marks the one to keep: most pixels, then sharpest."
          : "Frames shot within a couple of seconds that look alike. ★ marks the sharpest."}
      </p>
      {groups === null && <div className="spinner" />}
      {groups?.length === 0 && <div className="empty">Nothing found. Run <code>photo-search organize</code>.</div>}
      {groups?.map((g, gi) => (
        <div key={g.group_id} className="group">
          <div className="muted small">{g.size} photos · {formatDate(g.photos[0]?.taken_at, true)}</div>
          <PhotoGrid
            photos={g.photos.map((p) => ({ photo_id: p.photo_id, thumb_url: p.thumb_url, badge: p.is_best ? "★ keep" : undefined }))}
            onOpen={(i) => setOpen({ g: gi, i })}
          />
        </div>
      ))}
      {open && groups && (
        <Lightbox
          items={groups[open.g].photos}
          index={open.i}
          onIndex={(i) => setOpen({ g: open.g, i })}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  );
}
