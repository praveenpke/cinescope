// Mirror of the FastAPI Pydantic schemas in api/schemas.py. Kept intentionally
// hand-written (rather than codegen) so the contract is visible and reviewable.

export interface YearRange {
  start: number | null;
  end: number | null;
}

export interface QuerySpec {
  reference_titles: string[];
  mood_adjustments: string[];
  genres_include: string[];
  genres_exclude: string[];
  year_range: YearRange | null;
  min_rating: number | null;
  similarity_text: string;
}

export interface Why {
  semantic_similarity: number | null;
  behavioral_boost: number | null;
  quality_score: number | null;
  matched_filters: string[];
  liked_by_fans_of: string[];
}

export interface MovieSummary {
  movie_id: number;
  tmdb_id: number | null;
  title: string;
  release_year: number | null;
  overview: string | null;
  genres: string[];
  poster_url: string | null;
  vote_average: number | null;
  rating_count: number | null;
  source: string;
}

export interface DiscoverResult extends MovieSummary {
  score: number;
  why: Why;
}

/** parser is 'claude', 'heuristic_fallback', or 'provided_spec' (chip re-query). */
export interface DiscoverResponse {
  query: string;
  parser: string;
  spec: QuerySpec;
  table: string;
  results: DiscoverResult[];
}

export interface MovieDetail extends MovieSummary {
  keywords: string[];
  runtime: number | null;
  popularity: number | null;
  vote_count: number | null;
  rating_mean: number | null;
  bayes_score: number | null;
}

export interface SimilarList {
  label: string;
  basis: string; // 'embedding' | 'als_factors'
  results: MovieSummary[];
}

export interface MovieDetailResponse {
  movie: MovieDetail;
  more_like_this: SimilarList[];
}

export interface HealthResponse {
  status: string;
  table: string;
  titles: number;
  parser: string;
}

/** POST /api/discover body: free text OR a pre-parsed spec (editable chips). */
export interface DiscoverRequest {
  query?: string;
  spec?: QuerySpec;
  limit?: number;
}

export const EMPTY_SPEC: QuerySpec = {
  reference_titles: [],
  mood_adjustments: [],
  genres_include: [],
  genres_exclude: [],
  year_range: null,
  min_rating: null,
  similarity_text: "",
};
