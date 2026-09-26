import type { Filters } from "./api";
import type { IconName } from "./components/Icon";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export type Chip = { key: keyof Filters | "focal" | "aperture" | "iso" | "dates"; label: string; icon: IconName };

/** Human-readable chips for what the query parser extracted. */
export function filterChips(f: Filters): Chip[] {
  const chips: Chip[] = [];
  if (f.date_from || f.date_to) {
    const a = f.date_from ?? "…";
    const b = f.date_to ?? "…";
    chips.push({ key: "dates", icon: "calendar", label: a === b ? a : `${a} → ${b}` });
  }
  if (f.months?.length) chips.push({ key: "months", icon: "calendar", label: f.months.map((m) => MONTHS[m - 1]).join(", ") });
  if (f.years?.length) chips.push({ key: "years", icon: "calendar", label: f.years.join(", ") });
  if (f.place) chips.push({ key: "place", icon: "pin", label: f.place });
  if (f.people?.length) chips.push({ key: "people", icon: "user", label: f.people.join(" + ") });
  if (f.camera) chips.push({ key: "camera", icon: "camera", label: f.camera });
  if (f.lens) chips.push({ key: "lens", icon: "lens", label: f.lens });
  if (f.focal_min != null || f.focal_max != null)
    chips.push({ key: "focal", icon: "focal", label: range(f.focal_min, f.focal_max, "mm") });
  if (f.aperture_min != null || f.aperture_max != null)
    chips.push({ key: "aperture", icon: "lens", label: range(f.aperture_min, f.aperture_max, "", "f/") });
  if (f.iso_min != null || f.iso_max != null) chips.push({ key: "iso", icon: "iso", label: range(f.iso_min, f.iso_max, "", "ISO ") });
  if (f.media) chips.push({ key: "media", icon: "film", label: f.media === "video" ? "videos" : "photos" });
  if (f.orientation) chips.push({ key: "orientation", icon: "orientation", label: f.orientation });
  if (f.keywords?.length) chips.push({ key: "keywords", icon: "hash", label: f.keywords.join(", ") });
  if (f.album_id) chips.push({ key: "album_id", icon: "folder", label: "album" });
  return chips;
}

function range(a: number | null | undefined, b: number | null | undefined, unit: string, prefix = ""): string {
  const fmt = (x: number) => `${prefix}${Math.round(x * 10) / 10}${unit}`;
  if (a != null && b != null) return Math.abs(a - b) < 0.2 ? fmt((a + b) / 2) : `${fmt(a)}–${fmt(b)}`;
  if (a != null) return `≥ ${fmt(a)}`;
  return `≤ ${fmt(b as number)}`;
}

/** Remove the fields behind one chip. */
export function withoutChip(f: Filters, key: Chip["key"]): Filters {
  const g: Filters = { ...f };
  const clear: Record<string, (keyof Filters)[]> = {
    dates: ["date_from", "date_to"],
    focal: ["focal_min", "focal_max"],
    aperture: ["aperture_min", "aperture_max"],
    iso: ["iso_min", "iso_max"],
  };
  for (const k of clear[key] ?? [key as keyof Filters]) {
    if (k === "people" || k === "keywords") g[k] = [];
    else (g as Record<string, unknown>)[k] = null;
  }
  return g;
}

export function isEmptyFilters(f: Filters): boolean {
  return filterChips(f).length === 0;
}

export function formatDate(iso: string | null | undefined, withTime = false): string {
  if (!iso) return "";
  // taken_at is camera wall-clock stored as UTC: format in UTC so 18:30 stays 18:30
  const d = new Date(iso);
  const opts: Intl.DateTimeFormatOptions = { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" };
  if (withTime) Object.assign(opts, { hour: "2-digit", minute: "2-digit" });
  return d.toLocaleString(undefined, opts);
}

export function formatTs(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
