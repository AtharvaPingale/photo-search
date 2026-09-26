import { createRoot } from "react-dom/client";
import { registerSW } from "virtual:pwa-register";

import App from "./App";
import "./styles.css";

registerSW({ immediate: true });

// No StrictMode: its dev-only double effects would push/pop the lightbox's
// history entry and close it immediately.
createRoot(document.getElementById("root")!).render(<App />);
