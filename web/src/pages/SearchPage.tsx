import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, type Filters, type Hit, type Parsed, type SearchResponse } from "../api";
import { FilterChips } from "../components/FilterChips";
import { Lightbox } from "../components/Lightbox";
import { PhotoGrid } from "../components/PhotoGrid";
import { SearchBar } from "../components/SearchBar";
import { isEmptyFilters } from "../format";
import { useInfinite } from "../hooks/useInfinite";
import { navigate, type Route } from "../router";

const PAGE = 60;

type Meta = { resp: SearchResponse | null };

export function SearchPage({ route }: { route: Route }) {
  const q = route.params.get("q") ?? "";
  const like = route.params.get("like");
  const [override, setOverride] = useState<{ semantic: string; filters: Filters } | null>(null);
  const [pos, setPos] = useState<string[]>([]);
  const [neg, setNeg] = useState<string[]>([]);
  const [imageResult, setImageResult] = useState<SearchResponse | null>(null);
  const [imageName, setImageName] = useState<string | null>(null);
  const [open, setOpen] = useState<number | null>(null);
  const [meta, setMeta] = useState<Meta>({ resp: null });
  // what later pages repeat: the first page's parse (or its fallback), never re-parsed
  const base = useRef<{ semantic: string; filters: Filters | null }>({ semantic: q, filters: null });

  const mode = imageResult ? "image" : like ? "similar" : q || override ? "text" : "browse";
  const key = JSON.stringify({ q, like, override, pos, neg, image: imageResult?.query ?? null, n: imageName });

  const fetchPage = useCallback(
    async (offset: number): Promise<Hit[]> => {
      if (mode === "image") return offset === 0 ? (imageResult?.hits ?? []) : [];
      if (mode === "browse") {
        const rows = await api.browse({}, PAGE, offset);
        return rows.map((r) => ({ ...(r as unknown as Hit), photo_id: r.id }));
      }
      if (mode === "similar") {
        const r = await api.search({ like_photo_id: like, k: PAGE, offset });
        if (offset === 0) setMeta({ resp: r });
        return r.hits;
      }
      if (offset === 0 && !override) {
        const r = await api.search({ q, parse: "auto", k: PAGE, offset: 0, positive_ids: pos, negative_ids: neg });
        base.current = r.fallback_used ? { semantic: q, filters: null } : { semantic: r.parsed.semantic, filters: r.parsed.filters };
        setMeta({ resp: r });
        return r.hits;
      }
      const src = override ?? base.current;
      const r = await api.search({
        q: src.semantic, parse: "off", filters: src.filters, k: PAGE, offset, positive_ids: pos, negative_ids: neg,
      });
      if (offset === 0) setMeta({ resp: { ...r, parsed: { ...r.parsed, source: "user" } } });
      return r.hits;
    },
    [key],
  );

  // a new query must not show the previous one's parse, chips or result count while it loads
  useEffect(() => setMeta({ resp: null }), [key]);
  const { items, loading, done, error, sentinel } = useInfinite<Hit>(key, fetchPage, PAGE);

  const search = (text: string) => {
    setOverride(null);
    setPos([]);
    setNeg([]);
    setImageResult(null);
    setImageName(null);
    navigate("search", text ? { q: text } : {});
  };

  const parsed: Parsed | null = meta.resp?.parsed ?? null;
  const shownFilters = override?.filters ?? (mode === "text" ? parsed?.filters : undefined);
  const feedback = (id: string, label: 1 | -1) => {
    const query = override?.semantic ?? q;
    if (query) void api.feedback(query, id, label).catch(() => undefined);
    if (label === 1) setPos((p) => [...new Set([...p, id])]);
    else setNeg((n) => [...new Set([...n, id])]);
    setOpen(null);
  };

  const summary = useMemo(() => {
    const r = meta.resp;
    if (!r || mode === "browse") return null;
    const parts = [`${items.length}${done ? "" : "+"} results`, `${Math.round(r.timings_ms.total ?? 0)} ms`];
    if (r.signals.length) parts.push(r.signals.join(" + "));
    return parts.join(" · ");
  }, [meta.resp, items.length, done, mode]);

  return (
    <div className="page">
      <header className="search-header">
        <SearchBar
          value={q}
          busy={loading && items.length === 0}
          onSearch={search}
          onImage={async (file) => {
            setImageName(file.name);
            try {
              const r = await api.searchByImage(file);
              setImageResult(r);
              setMeta({ resp: r });
            } catch {
              setImageResult(null);
            }
          }}
        />
        {mode === "text" && shownFilters && (
          <FilterChips
            filters={shownFilters}
            source={override ? "user" : parsed?.source}
            onChange={(f) => setOverride({ semantic: override?.semantic ?? parsed?.semantic ?? q, filters: f })}
          />
        )}
        {mode === "text" && parsed && override === null && parsed.semantic !== q && (
          <div className="muted small parsed-line">
            searching for “{parsed.semantic || "anything"}”{!isEmptyFilters(parsed.filters) ? " with these filters" : ""}
          </div>
        )}
      </header>

      {mode === "similar" && (
        <div className="banner">
          Photos similar to one you picked · <button className="link" onClick={() => search("")}>clear</button>
        </div>
      )}
      {mode === "image" && (
        <div className="banner">
          Photos like <b>{imageName}</b> · <button className="link" onClick={() => search(q)}>clear</button>
        </div>
      )}
      {(pos.length > 0 || neg.length > 0) && (
        <div className="banner">
          Refined by {pos.length} more-like and {neg.length} less-like ·{" "}
          <button className="link" onClick={() => { setPos([]); setNeg([]); }}>reset</button>
        </div>
      )}
      {meta.resp?.fallback_used && mode === "text" && !override && (
        <div className="banner warn">No photos matched those filters, so these are plain visual matches.</div>
      )}
      {summary && <div className="muted small summary">{summary}</div>}
      {mode === "browse" && <h2 className="section-title">Recent</h2>}
      {error && <div className="banner warn">{error}</div>}
      {!loading && done && items.length === 0 && !error && <div className="empty">No photos found.</div>}

      <PhotoGrid photos={items} onOpen={setOpen} sentinel={sentinel} loading={loading && items.length > 0} />

      {open !== null && (
        <Lightbox
          items={items}
          index={open}
          onIndex={setOpen}
          onClose={() => setOpen(null)}
          onMoreLike={mode === "text" ? (id) => feedback(id, 1) : undefined}
          onLessLike={mode === "text" ? (id) => feedback(id, -1) : undefined}
          onSimilar={(id) => {
            setOpen(null);
            setImageResult(null);
            navigate("search", { like: id });
          }}
        />
      )}
    </div>
  );
}
