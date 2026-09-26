import { useEffect, useState } from "react";

import { api, type Filters } from "../api";
import { filterChips, withoutChip } from "../format";
import { Icon } from "./Icon";

type Props = {
  filters: Filters;
  source?: string;
  onChange: (f: Filters) => void;
};

/** What the query parser understood, as chips the user can remove or edit. */
export function FilterChips({ filters, source, onChange }: Props) {
  const [editing, setEditing] = useState(false);
  const chips = filterChips(filters);
  return (
    <div className="chips">
      {chips.map((c) => (
        <span key={c.key} className="chip" onClick={() => setEditing(true)}>
          <Icon name={c.icon} size={14} className="chip-icon" />
          {c.label}
          <button
            className="chip-x"
            aria-label={`remove ${c.label}`}
            onClick={(e) => {
              e.stopPropagation();
              onChange(withoutChip(filters, c.key));
            }}
          >
            ×
          </button>
        </span>
      ))}
      <button className="chip add" onClick={() => setEditing(true)}>
        <Icon name="sliders" size={14} /> {chips.length ? "Edit" : "Filters"}
      </button>
      {source === "llm" && chips.length > 0 && <span className="chip-note">parsed by LLM</span>}
      {editing && <FilterEditor filters={filters} onClose={() => setEditing(false)} onApply={(f) => { onChange(f); setEditing(false); }} />}
    </div>
  );
}

type Vocab = { cameras: string[]; lenses: string[]; places: string[]; people: string[] };
let vocabCache: Promise<Vocab> | null = null;

function FilterEditor({ filters, onApply, onClose }: { filters: Filters; onApply: (f: Filters) => void; onClose: () => void }) {
  const [f, setF] = useState<Filters>({ ...filters, people: filters.people ?? [] });
  const [vocab, setVocab] = useState<Vocab | null>(null);
  useEffect(() => {
    vocabCache ??= api.vocab();
    vocabCache.then(setVocab).catch(() => (vocabCache = null));
  }, []);

  const set = <K extends keyof Filters>(k: K, v: Filters[K]) => setF((prev) => ({ ...prev, [k]: v }));
  const num = (v: string) => (v === "" ? null : Number(v));
  const text = (v: string) => (v.trim() === "" ? null : v);

  return (
    <div className="sheet-backdrop" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>
        <h3>Filters</h3>
        <div className="form-grid">
          <label>
            From
            <input type="date" value={f.date_from ?? ""} onChange={(e) => set("date_from", text(e.target.value))} />
          </label>
          <label>
            To
            <input type="date" value={f.date_to ?? ""} onChange={(e) => set("date_to", text(e.target.value))} />
          </label>
          <label className="wide">
            Place
            <input list="dl-places" value={f.place ?? ""} onChange={(e) => set("place", text(e.target.value))} />
          </label>
          <label className="wide">
            People <span className="muted">(comma separated, all must appear)</span>
            <input
              list="dl-people"
              value={(f.people ?? []).join(", ")}
              onChange={(e) => set("people", e.target.value.split(",").map((s) => s.trim()).filter(Boolean))}
            />
          </label>
          <label>
            Camera
            <input list="dl-cameras" value={f.camera ?? ""} onChange={(e) => set("camera", text(e.target.value))} />
          </label>
          <label>
            Lens
            <input list="dl-lenses" value={f.lens ?? ""} onChange={(e) => set("lens", text(e.target.value))} />
          </label>
          <label>
            Focal ≥ (mm)
            <input type="number" inputMode="decimal" value={f.focal_min ?? ""} onChange={(e) => set("focal_min", num(e.target.value))} />
          </label>
          <label>
            Focal ≤ (mm)
            <input type="number" inputMode="decimal" value={f.focal_max ?? ""} onChange={(e) => set("focal_max", num(e.target.value))} />
          </label>
          <label>
            Aperture ≥ ƒ
            <input type="number" step="0.1" inputMode="decimal" value={f.aperture_min ?? ""} onChange={(e) => set("aperture_min", num(e.target.value))} />
          </label>
          <label>
            Aperture ≤ ƒ
            <input type="number" step="0.1" inputMode="decimal" value={f.aperture_max ?? ""} onChange={(e) => set("aperture_max", num(e.target.value))} />
          </label>
          <label>
            ISO ≥
            <input type="number" inputMode="numeric" value={f.iso_min ?? ""} onChange={(e) => set("iso_min", num(e.target.value))} />
          </label>
          <label>
            ISO ≤
            <input type="number" inputMode="numeric" value={f.iso_max ?? ""} onChange={(e) => set("iso_max", num(e.target.value))} />
          </label>
          <label>
            Type
            <select value={f.media ?? ""} onChange={(e) => set("media", (text(e.target.value) as Filters["media"]) ?? null)}>
              <option value="">Photos + videos</option>
              <option value="photo">Photos</option>
              <option value="video">Videos</option>
            </select>
          </label>
          <label>
            Orientation
            <select value={f.orientation ?? ""} onChange={(e) => set("orientation", (text(e.target.value) as Filters["orientation"]) ?? null)}>
              <option value="">Any</option>
              <option value="landscape">Landscape</option>
              <option value="portrait">Portrait</option>
              <option value="square">Square</option>
            </select>
          </label>
        </div>
        <datalist id="dl-places">{vocab?.places.slice(0, 300).map((p) => <option key={p} value={p} />)}</datalist>
        <datalist id="dl-people">{vocab?.people.map((p) => <option key={p} value={p} />)}</datalist>
        <datalist id="dl-cameras">{vocab?.cameras.map((p) => <option key={p} value={p} />)}</datalist>
        <datalist id="dl-lenses">{vocab?.lenses.map((p) => <option key={p} value={p} />)}</datalist>
        <div className="sheet-actions">
          <button className="ghost" onClick={() => onApply({ people: [] })}>
            Clear all
          </button>
          <span style={{ flex: 1 }} />
          <button className="ghost" onClick={onClose}>
            Cancel
          </button>
          <button className="primary" onClick={() => onApply(f)}>
            Apply
          </button>
        </div>
      </div>
    </div>
  );
}
