import type {
  DiscoverRequest,
  DiscoverResponse,
  HealthResponse,
  MovieDetailResponse,
} from "./types";

// All requests hit same-origin /api/*, proxied to FastAPI by Vite in dev
// (see vite.config.ts). No base URL needed.

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    // Network / server-down: surface a friendly, actionable message.
    throw new ApiError(0, "Cannot reach the API. Is `uvicorn api.main:app` running?");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body — keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export function discover(body: DiscoverRequest): Promise<DiscoverResponse> {
  return request<DiscoverResponse>("/api/discover", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function fetchMovie(movieId: number): Promise<MovieDetailResponse> {
  return request<MovieDetailResponse>(`/api/movies/${movieId}`);
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}
