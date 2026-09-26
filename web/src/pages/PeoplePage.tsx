import { useCallback, useEffect, useState } from "react";

import { api, type Face, type Person } from "../api";
import { Lightbox } from "../components/Lightbox";
import { navigate, type Route } from "../router";

export function PeoplePage({ route }: { route: Route }) {
  const id = route.path[1];
  return id === "unassigned" ? <Unassigned /> : id !== undefined ? <PersonDetail clusterId={Number(id)} /> : <PeopleList />;
}

function PeopleList() {
  const [people, setPeople] = useState<Person[] | null>(null);
  const [selected, setSelected] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => api.people().then(setPeople).catch((e) => setError(String(e))), []);
  useEffect(() => void load(), [load]);

  const toggle = (id: number) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  const merge = async () => {
    if (!people || selected.length < 2) return;
    // keep a named cluster as the target if there is one
    const named = people.filter((p) => selected.includes(p.cluster_id) && p.name);
    const target = named[0]?.cluster_id ?? selected[0];
    await api.merge(selected.filter((x) => x !== target), target);
    setSelected([]);
    await load();
  };

  return (
    <div className="page">
      <header className="page-header">
        <h1>People</h1>
        <div className="actions">
          {selected.length >= 2 && (
            <button className="primary" onClick={merge}>
              Merge {selected.length}
            </button>
          )}
          {selected.length > 0 && (
            <button className="ghost" onClick={() => setSelected([])}>
              Cancel
            </button>
          )}
          <a className="ghost button" href="#/people/unassigned">
            Unassigned
          </a>
        </div>
      </header>
      <p className="muted small">
        Faces are grouped automatically. Name a group to search for that person (“Rohan hiking”); select several to merge.
        Face data stays on this machine; <code>photo-search faces wipe</code> removes all of it.
      </p>
      {error && <div className="banner warn">{error}</div>}
      {people === null && <div className="spinner" />}
      {people?.length === 0 && (
        <div className="empty">No face groups yet. Faces are opt-in: set PS_FACE_ROOTS, then run faces detect + cluster.</div>
      )}
      <div className="people-grid">
        {people?.map((p) => (
          <div key={p.cluster_id} className={`person${selected.includes(p.cluster_id) ? " selected" : ""}`}>
            <button className="person-faces" onClick={() => navigate(`people/${p.cluster_id}`)}>
              {p.sample_crops.slice(0, 4).map((u) => (
                <img key={u} src={u} loading="lazy" alt="" />
              ))}
            </button>
            <NameInput person={p} onSaved={load} />
            <div className="muted small">
              {p.n_photos} photos
              <label className="select-box">
                <input type="checkbox" checked={selected.includes(p.cluster_id)} onChange={() => toggle(p.cluster_id)} /> select
              </label>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function NameInput({ person, onSaved }: { person: Person; onSaved: () => void }) {
  const [name, setName] = useState(person.name ?? "");
  const save = async () => {
    if ((person.name ?? "") === name.trim()) return;
    await api.updatePerson(person.cluster_id, { name: name.trim() || null });
    onSaved();
  };
  return (
    <input
      className="name-input"
      placeholder="Add a name"
      value={name}
      onChange={(e) => setName(e.target.value)}
      onBlur={save}
      onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
    />
  );
}

function PersonDetail({ clusterId }: { clusterId: number }) {
  const [faces, setFaces] = useState<Face[] | null>(null);
  const [person, setPerson] = useState<Person | null>(null);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [open, setOpen] = useState<number | null>(null);
  const [aliases, setAliases] = useState("");

  const load = useCallback(async () => {
    const [f, ps] = await Promise.all([api.personFaces(clusterId), api.people()]);
    setFaces(f);
    const p = ps.find((x) => x.cluster_id === clusterId) ?? null;
    setPerson(p);
    setAliases((p?.aliases ?? []).join(", "));
    setSel(new Set());
  }, [clusterId]);
  useEffect(() => void load(), [load]);

  const toggle = (id: string) =>
    setSel((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  return (
    <div className="page">
      <header className="page-header">
        <button className="ghost" onClick={() => navigate("people")}>
          ‹ People
        </button>
        <h1>{person?.name ?? `Person ${clusterId}`}</h1>
        <div className="actions">
          {person?.name && (
            <button className="ghost" onClick={() => navigate("search", { q: person.name ?? "" })}>
              Search photos
            </button>
          )}
        </div>
      </header>
      {person && (
        <div className="row">
          <NameInput person={person} onSaved={load} />
          <input
            className="name-input"
            placeholder="Aliases, e.g. me"
            value={aliases}
            onChange={(e) => setAliases(e.target.value)}
            onBlur={() =>
              api.updatePerson(clusterId, {
                name: person.name,
                aliases: aliases.split(",").map((s) => s.trim()).filter(Boolean),
              })
            }
          />
        </div>
      )}
      {sel.size > 0 && (
        <div className="banner">
          {sel.size} selected ·{" "}
          <button className="link" onClick={async () => { await api.split([...sel]); await load(); }}>
            split into a new person
          </button>{" "}
          ·{" "}
          <button
            className="link"
            onClick={async () => {
              await Promise.all([...sel].map((f) => api.assignFace(f, null)));
              await load();
            }}
          >
            not this person
          </button>{" "}
          · <button className="link" onClick={() => setSel(new Set())}>clear</button>
        </div>
      )}
      <p className="muted small">Tap faces to select them; open a photo with its ⤢ button.</p>
      {faces === null && <div className="spinner" />}
      <div className="face-grid">
        {faces?.map((f, i) => (
          <div key={f.id} className={`face${sel.has(f.id) ? " selected" : ""}`}>
            <button onClick={() => toggle(f.id)} aria-label="select face">
              <img src={f.crop_url} loading="lazy" alt="" />
            </button>
            <button className="face-open" onClick={() => setOpen(i)} aria-label="open photo">
              ⤢
            </button>
          </div>
        ))}
      </div>
      {open !== null && faces && (
        <Lightbox items={faces.map((f) => ({ photo_id: f.photo_id }))} index={open} onIndex={setOpen} onClose={() => setOpen(null)} />
      )}
    </div>
  );
}

function Unassigned() {
  const [faces, setFaces] = useState<Face[] | null>(null);
  const [people, setPeople] = useState<Person[]>([]);
  const [target, setTarget] = useState<string>("");
  const [sel, setSel] = useState<Set<string>>(new Set());
  const load = useCallback(async () => {
    const [f, p] = await Promise.all([api.unassignedFaces(), api.people()]);
    setFaces(f);
    setPeople(p);
    setSel(new Set());
  }, []);
  useEffect(() => void load(), [load]);

  return (
    <div className="page">
      <header className="page-header">
        <button className="ghost" onClick={() => navigate("people")}>
          ‹ People
        </button>
        <h1>Unassigned faces</h1>
      </header>
      {sel.size > 0 && (
        <div className="banner row">
          Assign {sel.size} to
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="">choose…</option>
            {people.map((p) => (
              <option key={p.cluster_id} value={p.cluster_id}>
                {p.name ?? `Person ${p.cluster_id}`}
              </option>
            ))}
          </select>
          <button
            className="primary"
            disabled={!target}
            onClick={async () => {
              await Promise.all([...sel].map((f) => api.assignFace(f, Number(target))));
              await load();
            }}
          >
            Assign
          </button>
          <button className="ghost" onClick={async () => { await api.split([...sel]); await load(); }}>
            New person
          </button>
        </div>
      )}
      {faces === null && <div className="spinner" />}
      <div className="face-grid">
        {faces?.map((f) => (
          <div key={f.id} className={`face${sel.has(f.id) ? " selected" : ""}`}>
            <button
              onClick={() =>
                setSel((s) => {
                  const n = new Set(s);
                  if (n.has(f.id)) n.delete(f.id);
                  else n.add(f.id);
                  return n;
                })
              }
            >
              <img src={f.crop_url} loading="lazy" alt="" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
