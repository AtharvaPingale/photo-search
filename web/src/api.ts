// Typed client for the FastAPI backend. Same-origin, cookie auth: any 401 fires
// an "auth-required" event and the app swaps to the login screen.

export type Filters = {
  date_from?: string | null;
  date_to?: string | null;
  months?: number[] | null;
  years?: number[] | null;
  place?: string | null;
  camera?: string | null;
  lens?: string | null;
  focal_min?: number | null;
  focal_max?: number | null;
  aperture_min?: number | null;
  aperture_max?: number | null;
  iso_min?: number | null;
  iso_max?: number | null;
  people?: string[];
  media?: "photo" | "video" | null;
  orientation?: "landscape" | "portrait" | "square" | null;
  keywords?: string[];
  album_id?: string | null;
};

export type Hit = {
  photo_id: string;
  score: number;
  ranks: Record<string, number>;
  path: string;
  file_hash: string;
  thumb_url: string;
  display_url: string;
  taken_at: string | null;
  place_name: string | null;
  country: string | null;
  camera: string | null;
  lens: string | null;
  focal_length: number | null;
  aperture: number | null;
  width: number | null;
  height: number | null;
  caption: string | null;
  is_video_frame: boolean;
  video_id: string | null;
  frame_ts: number | null;
};

export type Parsed = {
  text: string;
  semantic: string;
  filters: Filters;
  source: "llm" | "rules" | "none" | "user";
  error: string | null;
  latency_ms: number;
};

export type SearchResponse = {
  query: string;
  parsed: Parsed;
  hits: Hit[];
  signals: string[];
  fallback_used: boolean;
  timings_ms: Record<string, number>;
};

export type SearchRequest = {
  q?: string;
  filters?: Filters | null;
  parse?: "auto" | "rules" | "llm" | "hybrid" | "off";
  k?: number;
  offset?: number;
  positive_ids?: string[];
  negative_ids?: string[];
  like_photo_id?: string | null;
};

export type PhotoDetails = Record<string, unknown> & {
  id: string;
  path: string;
  taken_at: string | null;
  place_name: string | null;
  admin1: string | null;
  country: string | null;
  camera: string | null;
  lens: string | null;
  focal_length: number | null;
  aperture: number | null;
  iso: number | null;
  shutter: string | null;
  width: number | null;
  height: number | null;
  keywords: string[];
  captions: { model: string; caption: string }[];
  ocr_text: string | null;
  faces: { id: string; cluster_id: number | null; name: string | null }[];
  albums: { id: string; title: string }[];
  thumb_url: string;
  display_url: string;
  original_url: string;
  download_url: string;
  media_type: string;
  format: string | null;
  is_video_frame: boolean;
  frame_ts: number | null;
};

export type Person = {
  cluster_id: number;
  name: string | null;
  aliases: string[];
  hidden: boolean;
  n_faces: number;
  n_photos: number;
  sample_crops: string[];
};

export type Face = { id: string; photo_id: string; det_score: number; manual?: boolean; crop_url: string; thumb_url?: string };

export type Album = {
  id: string;
  title: string;
  summary: string | null;
  start_at: string | null;
  end_at: string | null;
  place_name: string | null;
  n_photos: number;
  cover_url: string | null;
};

export type GridPhoto = {
  photo_id: string;
  thumb_url: string;
  is_video_frame?: boolean;
  frame_ts?: number | null;
  badge?: string;
};

export type Group = {
  group_id: number;
  size: number;
  photos: { photo_id: string; is_best: boolean; path: string; taken_at: string | null; width: number; height: number; sharpness: number; thumb_url: string }[];
};

export type AgentAnswer = {
  question: string;
  answer: string;
  evidence: { photo_id: string; note: string; thumb_url: string }[];
  invalid_citations: string[];
  grounded: boolean;
  tool_calls: number;
  steps: { tool: string; args: Record<string, unknown>; n_photos: number; error: string | null }[];
  latency_ms: number;
  error: string | null;
};

export type EvalQuery = {
  id: string;
  query: string;
  category: string;
  split: "dev" | "test";
  relevant: string[];
  grades: Record<string, number>;
  notes: string;
  labeled_at?: string | null;
};

export type Candidate = { photo_id: string; file_hash: string; thumb_url: string; path: string; found_by: Record<string, number> };

export type ReportSummary = {
  file: string;
  kind: string;
  name: string;
  split: string;
  created_at: string;
  git_commit: string;
  overall?: Record<string, number>;
  latency_ms?: Record<string, number>;
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown, isForm = false): Promise<T> {
  const init: RequestInit = { method, credentials: "same-origin", headers: {} };
  if (body !== undefined) {
    if (isForm) init.body = body as FormData;
    else {
      init.body = JSON.stringify(body);
      (init.headers as Record<string, string>)["Content-Type"] = "application/json";
    }
  }
  const res = await fetch(path, init);
  if (res.status === 401 && !path.startsWith("/api/auth/")) {
    window.dispatchEvent(new Event("auth-required"));
  }
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg);
  }
  return (await res.json()) as T;
}

const qs = (params: Record<string, string | number | boolean | undefined | null>) =>
  new URLSearchParams(
    Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null && v !== "")
      .map(([k, v]) => [k, String(v)]),
  ).toString();

export const api = {
  authStatus: () => request<{ required: boolean; authenticated: boolean }>("GET", "/api/auth/status"),
  login: (token: string) => request<{ ok: boolean }>("POST", "/api/auth/login", { token }),
  logout: () => request<{ ok: boolean }>("POST", "/api/auth/logout"),

  search: (req: SearchRequest) => request<SearchResponse>("POST", "/api/search", req),
  searchByImage: (file: File, k = 120) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<SearchResponse>("POST", `/api/search/image?${qs({ k })}`, fd, true);
  },
  parse: (q: string) => request<Parsed>("GET", `/api/parse?${qs({ q })}`),
  feedback: (query: string, photo_id: string, label: 1 | -1) =>
    request<{ ok: boolean }>("POST", "/api/feedback", { query, photo_id, label }),
  browse: (filters: Filters, limit: number, offset: number) =>
    request<(GridPhoto & { id: string })[]>("POST", `/api/photos/browse?${qs({ limit, offset })}`, filters),
  photo: (id: string) => request<PhotoDetails>("GET", `/api/photos/${id}`),
  stats: () => request<Record<string, unknown>>("GET", "/api/stats"),
  vocab: () => request<{ cameras: string[]; lenses: string[]; places: string[]; people: string[] }>("GET", "/api/vocab"),

  people: () => request<Person[]>("GET", "/api/people"),
  personFaces: (id: number) => request<Face[]>("GET", `/api/people/${id}/faces`),
  unassignedFaces: () => request<Face[]>("GET", "/api/faces/unassigned"),
  updatePerson: (id: number, body: { name?: string | null; aliases?: string[]; hidden?: boolean }) =>
    request<{ ok: boolean }>("PUT", `/api/people/${id}`, body),
  merge: (sources: number[], target: number) => request<{ moved_faces: number }>("POST", "/api/people/merge", { sources, target }),
  split: (face_ids: string[]) => request<{ new_cluster_id: number }>("POST", "/api/people/split", { face_ids }),
  assignFace: (faceId: string, cluster_id: number | null) =>
    request<{ ok: boolean }>("POST", `/api/faces/${faceId}/assign`, { cluster_id }),

  albums: () => request<Album[]>("GET", "/api/albums"),
  album: (id: string) =>
    request<Album & { photos: { id: string; thumb_url: string; is_video_frame: boolean }[] }>("GET", `/api/albums/${id}`),
  groups: (kind: "duplicate" | "burst") => request<Group[]>("GET", `/api/groups/${kind}`),

  ask: (question: string) => request<AgentAnswer>("POST", "/api/agent/ask", { question }),

  evalQueries: () =>
    request<{ categories: string[]; queries: EvalQuery[]; counts: Record<string, number> }>("GET", "/api/eval/queries"),
  saveEvalQuery: (q: Partial<EvalQuery> & { query: string }) => request<EvalQuery>("POST", "/api/eval/queries", q),
  deleteEvalQuery: (id: string) => request<{ deleted: boolean }>("DELETE", `/api/eval/queries/${id}`),
  candidates: (query: string, extra_queries: string[], relevant: string[]) =>
    request<{ candidates: Candidate[]; n: number }>("POST", "/api/eval/candidates", { query, extra_queries, relevant }),
  reports: () => request<ReportSummary[]>("GET", "/api/eval/reports"),
};

export const photoUrls = (id: string) => ({
  original: `/api/photos/${id}/original`,
  download: `/api/photos/${id}/original?download=1`,
  display: `/api/photos/${id}/display`,
  thumb: `/api/photos/${id}/thumb?size=256`,
});
