import ReactDOM from "react-dom/client";
import "@fontsource-variable/jetbrains-mono";
import "@fontsource-variable/inter";
import "../tokens.css";
import "./styles/app.css";
import App from "./App";

// Deliberately not wrapped in StrictMode: its double-invoked effects would
// register the Tauri event listeners twice in dev, double-counting every
// streamed delta and corrupting the throughput numbers this app exists to
// measure.
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <App />,
);
