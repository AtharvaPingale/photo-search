import { useState } from "react";

import { api } from "../api";

export function Login({ onDone }: { onDone: () => void }) {
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <div className="login">
      <img src="/icons/icon-192.png" width={72} height={72} alt="" />
      <h1>Photo Search</h1>
      <p className="muted">
        Enter the access token (<code>PS_AUTH_TOKEN</code>), or scan the QR code from <code>photo-search pair</code>.
      </p>
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          setError(null);
          try {
            await api.login(token.trim());
            onDone();
          } catch (err) {
            setError(err instanceof Error ? err.message : "login failed");
          } finally {
            setBusy(false);
          }
        }}
      >
        <input
          type="password"
          autoComplete="current-password"
          placeholder="access token"
          value={token}
          onChange={(e) => setToken(e.target.value)}
        />
        <button className="primary" disabled={!token || busy}>
          {busy ? "…" : "Unlock"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
