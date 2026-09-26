import { useEffect, useState } from "react";

// Hash routing (#/search?q=..., #/people/3): works from the home-screen PWA and
// needs no server rewrite rules.
export type Route = { path: string[]; params: URLSearchParams };

export function parseHash(hash: string): Route {
  const h = hash.replace(/^#\/?/, "");
  const [p, q = ""] = h.split("?");
  return { path: p.split("/").filter(Boolean), params: new URLSearchParams(q) };
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export function navigate(path: string, params?: Record<string, string>, replace = false) {
  const q = params ? new URLSearchParams(params).toString() : "";
  const hash = `#/${path}${q ? `?${q}` : ""}`;
  if (replace) window.history.replaceState(null, "", hash);
  else window.location.hash = hash;
  if (replace) window.dispatchEvent(new HashChangeEvent("hashchange"));
}
