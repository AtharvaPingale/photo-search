import { useCallback, useEffect, useMemo, useState } from "react";

import { api, type Candidate, type EvalQuery, type ReportSummary } from "../api";
import { Lightbox } from "../components/Lightbox";
import { PhotoGrid } from "../components/PhotoGrid";
import { navigate, type Route } from "../router";

export function EvalPage({ route }: { route: Route }) {
  const tab = route.params.get("tab") ?? "label";
  return (
    <div className="page">
      <header className="page-header">
        <h1>Eval</h1>
        <div className="tabs">
          <button className={tab === "label" ? "active" : ""} onClick={() => navigate("eval", { tab: "label" })}>
            Label
          </button>
          <button className={tab === "reports" ? "active" : ""} onClick={() => navigate("eval", { tab: "reports" })}>
            Reports
          </button>
        </div>
      </header>
      {tab === "reports" ? <Reports /> : <Labeler selectedId={route.params.get("id")} />}
    </div>
  );
}

function Labeler({ selectedId }: { selectedId: string | null }) {
  const [data, setData] = useState<Awaited<ReturnType<typeof api.evalQueries>> | null>(null);
  const [filter, setFilter] = useState<"todo" | "labeled" | "all">("todo");
  const [category, setCategory] = useState("");
  const load = useCallback(() => api.evalQueries().then(setData), []);
  useEffect(() => void load(), [load]);

  const selected = data?.queries.find((q) => q.id === selectedId) ?? null;
  const shown = (data?.queries ?? []).filter(
    (q) =>
      (filter === "all" || (filter === "todo" ? q.relevant.length === 0 : q.relevant.length > 0)) &&
      (!category || q.category === category),
  );

  if (selectedId !== null) {
    return (
      <QueryEditor
        key={selectedId}
        initial={selected ?? { id: "", query: "", category: category || "objects", split: "dev", relevant: [], grades: {}, notes: "" }}
        categories={data?.categories ?? []}
        onSaved={(q) => {
          void load();
          navigate("eval", { tab: "label", id: q.id }, true);
        }}
      />
    );
  }

  return (
    <>
      <p className="muted small">
        Label which photos are relevant for each query. Every retrieval change is judged on these labels, so pooled
        candidates from several systems are shown, not just the current best. {data && (
          <>
            {data.counts.labeled} / {data.counts.total} labelled ({data.counts.dev} dev, {data.counts.test} held-out test).
          </>
        )}
      </p>
      <div className="row">
        <select value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
          <option value="todo">To label</option>
          <option value="labeled">Labelled</option>
          <option value="all">All</option>
        </select>
        <select value={category} onChange={(e) => setCategory(e.target.value)}>
          <option value="">All categories</option>
          {data?.categories.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
        <button className="primary" onClick={() => navigate("eval", { tab: "label", id: "" })}>
          + New query
        </button>
      </div>
      {data === null && <div className="spinner" />}
      <ul className="query-list">
        {shown.map((q) => (
          <li key={q.id}>
            <button onClick={() => navigate("eval", { tab: "label", id: q.id })}>
              <span className={`split ${q.split}`}>{q.split}</span>
              <span className="q">{q.query}</span>
              <span className="muted small">
                {q.category} · {q.relevant.length} relevant
              </span>
            </button>
          </li>
        ))}
      </ul>
    </>
  );
}

function QueryEditor({ initial, categories, onSaved }: { initial: EvalQuery; categories: string[]; onSaved: (q: EvalQuery) => void }) {
  const [q, setQ] = useState(initial);
  const [extra, setExtra] = useState("");
  const [extras, setExtras] = useState<string[]>([]);
  const [cands, setCands] = useState<Candidate[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState<number | null>(null);
  const [saved, setSaved] = useState(false);

  const grades = useMemo(() => {
    const g: Record<string, number> = {};
    for (const h of q.relevant) g[h] = q.grades[h] ?? 1;
    return g;
  }, [q]);

  const fetchCands = useCallback(async () => {
    if (!q.query.trim()) return;
    setBusy(true);
    try {
      const r = await api.candidates(q.query, extras, q.relevant);
      setCands(r.candidates);
    } finally {
      setBusy(false);
    }
  }, [q.query, extras]);
  useEffect(() => {
    if (initial.query) void fetchCands();
  }, [fetchCands, initial.query]);

  // tap cycles: not relevant -> relevant -> highly relevant -> not relevant
  const cycle = (hash: string) => {
    setSaved(false);
    setQ((prev) => {
      const cur = prev.relevant.includes(hash) ? (prev.grades[hash] ?? 1) : 0;
      const next = (cur + 1) % 3;
      const relevant = prev.relevant.filter((h) => h !== hash);
      const g = { ...prev.grades };
      delete g[hash];
      if (next >= 1) relevant.push(hash);
      if (next === 2) g[hash] = 2;
      return { ...prev, relevant, grades: g };
    });
  };

  const save = async () => {
    const out = await api.saveEvalQuery({ id: q.id || undefined, query: q.query, category: q.category, relevant: q.relevant, grades: q.grades, notes: q.notes });
    setQ(out);
    setSaved(true);
    onSaved(out);
  };

  const photos = (cands ?? []).map((c) => ({ photo_id: c.photo_id, thumb_url: c.thumb_url, file_hash: c.file_hash }));
  return (
    <>
      <div className="row">
        <button className="ghost" onClick={() => navigate("eval", { tab: "label" })}>
          ‹ Queries
        </button>
        {q.id && <span className={`split ${q.split}`}>{q.id} · {q.split}</span>}
      </div>
      <div className="form-grid">
        <label className="wide">
          Query
          <input value={q.query} onChange={(e) => setQ({ ...q, query: e.target.value })} onBlur={() => void fetchCands()} />
        </label>
        <label>
          Category
          <select value={q.category} onChange={(e) => setQ({ ...q, category: e.target.value })}>
            {categories.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </label>
        <label>
          Notes
          <input value={q.notes} onChange={(e) => setQ({ ...q, notes: e.target.value })} />
        </label>
        <label className="wide">
          Find more with another phrasing (adds to the pool only)
          <span className="row">
            <input value={extra} onChange={(e) => setExtra(e.target.value)} placeholder="e.g. cafe chalkboard prices" />
            <button className="ghost" onClick={() => { if (extra.trim()) { setExtras([...extras, extra.trim()]); setExtra(""); } }}>
              Add
            </button>
          </span>
        </label>
      </div>
      <div className="row sticky-actions">
        <span className="muted small">
          {q.relevant.length} relevant ({Object.values(grades).filter((g) => g === 2).length} highly) · tap: relevant → highly → off
        </span>
        <span style={{ flex: 1 }} />
        <button className="primary" disabled={!q.query.trim()} onClick={save}>
          {saved ? "Saved ✓" : "Save"}
        </button>
      </div>
      {busy && <div className="spinner" />}
      <PhotoGrid
        photos={photos}
        onOpen={(i) => cycle(photos[i].file_hash)}
        highlight={(p) => {
          const g = grades[(p as unknown as { file_hash: string }).file_hash];
          return g === 2 ? "highly" : g === 1 ? "relevant" : null;
        }}
      />
      {cands && cands.length > 0 && (
        <button className="ghost" onClick={() => setOpen(0)}>
          View full size
        </button>
      )}
      {open !== null && cands && <Lightbox items={cands} index={open} onIndex={setOpen} onClose={() => setOpen(null)} />}
    </>
  );
}

function Reports() {
  const [reports, setReports] = useState<ReportSummary[] | null>(null);
  useEffect(() => void api.reports().then(setReports).catch(() => setReports([])), []);
  const f = (x?: number) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(3));
  return (
    <>
      <p className="muted small">
        Runs from <code>make eval</code>, <code>make ablation</code> and friends. Each is also an MLflow run.
      </p>
      {reports === null && <div className="spinner" />}
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>run</th>
              <th>split</th>
              <th>nDCG@10</th>
              <th>R@20</th>
              <th>MRR</th>
              <th>p50 ms</th>
              <th>when</th>
            </tr>
          </thead>
          <tbody>
            {reports
              ?.filter((r) => r.kind === "retrieval")
              .map((r) => (
                <tr key={r.file}>
                  <td>{r.name}</td>
                  <td>{r.split}</td>
                  <td>{f(r.overall?.["ndcg@10"])}</td>
                  <td>{f(r.overall?.["recall@20"])}</td>
                  <td>{f(r.overall?.mrr)}</td>
                  <td>{r.latency_ms ? Math.round(r.latency_ms.total_p50) : "–"}</td>
                  <td className="small">{r.created_at?.slice(0, 16).replace("T", " ")}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
