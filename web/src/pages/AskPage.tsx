import { useState } from "react";

import { api, type AgentAnswer } from "../api";
import { Lightbox } from "../components/Lightbox";

const EXAMPLES = [
  "What was my most photographed city in 2025?",
  "When did I last shoot the Columbus skyline?",
  "Show my best sunset from each trip this year",
];

export function AskPage() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<AgentAnswer[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<{ a: number; i: number } | null>(null);

  const ask = async (q: string) => {
    if (!q.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const a = await api.ask(q.trim());
      setHistory((h) => [a, ...h]);
      setQuestion("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <h1>Ask</h1>
      </header>
      <form className="ask-form" onSubmit={(e) => { e.preventDefault(); void ask(question); }}>
        <input
          placeholder="Ask about your library…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          enterKeyHint="send"
        />
        <button className="primary" disabled={busy || !question.trim()}>
          {busy ? "…" : "Ask"}
        </button>
      </form>
      {history.length === 0 && !busy && (
        <div className="examples">
          {EXAMPLES.map((q) => (
            <button key={q} className="chip" onClick={() => void ask(q)}>
              {q}
            </button>
          ))}
        </div>
      )}
      {busy && <div className="spinner" />}
      {error && <div className="banner warn">{error}</div>}
      {history.map((a, ai) => (
        <article key={ai} className="answer">
          <h3>{a.question}</h3>
          {a.error ? <p className="error">{a.error}</p> : <p>{a.answer}</p>}
          {a.evidence.length > 0 && (
            <div className="evidence">
              {a.evidence.map((e, i) => (
                <button key={e.photo_id} onClick={() => setOpen({ a: ai, i })} title={e.note}>
                  <img src={e.thumb_url} alt="" loading="lazy" />
                  <span className="small">{e.note}</span>
                </button>
              ))}
            </div>
          )}
          <details>
            <summary className="muted small">
              {a.tool_calls} tool calls · {(a.latency_ms / 1000).toFixed(1)} s ·{" "}
              {a.grounded ? "grounded in cited photos" : a.invalid_citations.length ? "⚠ cited photos it never saw" : "no photo evidence"}
            </summary>
            <ol className="steps small">
              {a.steps.map((s, i) => (
                <li key={i}>
                  <code>{s.tool}</code> {JSON.stringify(s.args)} → {s.error ? <span className="error">{s.error}</span> : `${s.n_photos} photos`}
                </li>
              ))}
            </ol>
          </details>
        </article>
      ))}
      {open && (
        <Lightbox
          items={history[open.a].evidence}
          index={open.i}
          onIndex={(i) => setOpen({ a: open.a, i })}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  );
}
