import { useEffect, useState } from "react";

import { api } from "./api";
import { Icon, type IconName } from "./components/Icon";
import { Login } from "./components/Login";
import { AlbumsPage, OrganizePage } from "./pages/AlbumsPage";
import { AskPage } from "./pages/AskPage";
import { EvalPage } from "./pages/EvalPage";
import { PeoplePage } from "./pages/PeoplePage";
import { SearchPage } from "./pages/SearchPage";
import { SettingsPage } from "./pages/SettingsPage";
import { navigate, useRoute } from "./router";

type Auth = "checking" | "ok" | "login";

const TABS: { path: string; label: string; icon: IconName }[] = [
  { path: "search", label: "Search", icon: "search" },
  { path: "people", label: "People", icon: "users" },
  { path: "albums", label: "Albums", icon: "albums" },
  { path: "ask", label: "Ask", icon: "sparkle" },
  { path: "library", label: "Library", icon: "menu" },
];

export default function App() {
  const route = useRoute();
  const [auth, setAuth] = useState<Auth>("checking");
  const [authRequired, setAuthRequired] = useState(false);

  useEffect(() => {
    (async () => {
      // pairing link from `photo-search pair`: #token=... (fragments never reach the server)
      const m = window.location.hash.match(/^#token=(.+)$/);
      if (m) {
        try {
          await api.login(decodeURIComponent(m[1]));
        } catch {
          /* fall through to the login screen */
        }
        navigate("search", {}, true);
      }
      try {
        const s = await api.authStatus();
        setAuthRequired(s.required);
        setAuth(s.authenticated ? "ok" : "login");
      } catch {
        setAuth("ok"); // API unreachable: show the app and let requests report errors
      }
    })();
    const onAuth = () => setAuth("login");
    window.addEventListener("auth-required", onAuth);
    return () => window.removeEventListener("auth-required", onAuth);
  }, []);

  if (auth === "checking") return <div className="spinner center" />;
  if (auth === "login") return <Login onDone={() => setAuth("ok")} />;

  const section = route.path[0] ?? "search";
  let page;
  switch (section) {
    case "people":
      page = <PeoplePage route={route} />;
      break;
    case "albums":
      page = <AlbumsPage route={route} />;
      break;
    case "organize":
      page = <OrganizePage route={route} />;
      break;
    case "ask":
      page = <AskPage />;
      break;
    case "eval":
      page = <EvalPage route={route} />;
      break;
    case "library":
      page = <SettingsPage authRequired={authRequired} />;
      break;
    default:
      page = <SearchPage route={route} />;
  }
  const active = ["organize", "eval"].includes(section) ? "library" : section;

  return (
    <div className="app">
      <nav className="nav">
        <a className="brand" href="#/search">
          <img src="/icons/favicon.svg" width={24} height={24} alt="" /> Photo Search
        </a>
        {TABS.map((t) => (
          <a key={t.path} href={`#/${t.path}`} className={active === t.path ? "active" : ""}>
            <Icon name={t.icon} size={22} className="nav-icon" />
            <span className="nav-label">{t.label}</span>
          </a>
        ))}
      </nav>
      <main>{page}</main>
    </div>
  );
}
