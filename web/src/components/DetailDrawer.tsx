import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchMovie } from "../lib/api";
import type { MovieSummary, SimilarList } from "../lib/types";
import { Poster } from "./Poster";

interface Props {
  movieId: number;
  onClose: () => void;
  onNavigate: (id: number) => void;
}

const BASIS_SUB: Record<string, string> = {
  embedding: "plot & genre embedding neighbors",
  als_factors: "collaborative-filtering neighbors",
};

/** Slide-in movie detail with two labeled more-like-this rows. */
export function DetailDrawer({ movieId, onClose, onNavigate }: Props) {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["movie", movieId],
    queryFn: () => fetchMovie(movieId),
  });

  // Close on Escape; lock body scroll while open.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  return (
    <>
      <div className="drawer-overlay" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label="Movie detail">
        <button className="drawer-close" onClick={onClose} aria-label="Close">
          ×
        </button>

        {isLoading && (
          <div className="state">
            <div className="spinner" />
            Loading detail…
          </div>
        )}
        {isError && (
          <div className="state error" style={{ margin: 24 }}>
            <h4>Couldn’t load this movie</h4>
            <p>{(error as Error).message}</p>
          </div>
        )}

        {data && (
          <>
            <div className="drawer-hero">
              <div style={{ position: "relative", flex: "none" }}>
                <div style={{ width: 130 }}>
                  <Poster
                    title={data.movie.title}
                    year={data.movie.release_year}
                    posterUrl={data.movie.poster_url}
                  />
                </div>
              </div>
              <div>
                <h2>{data.movie.title}</h2>
                <div className="card-sub" style={{ fontSize: 13 }}>
                  {data.movie.release_year !== null && <span>{data.movie.release_year}</span>}
                  {data.movie.runtime !== null && <span>{data.movie.runtime} min</span>}
                  <span>{data.movie.source}</span>
                </div>
                <div className="meta-row">
                  {data.movie.genres.map((g) => (
                    <span key={g} className="meta-tag">
                      {g}
                    </span>
                  ))}
                </div>
                <div className="stat-row">
                  {data.movie.bayes_score !== null && (
                    <div className="stat">
                      <div className="v">{data.movie.bayes_score.toFixed(2)}</div>
                      <div className="k">Bayes score</div>
                    </div>
                  )}
                  {data.movie.rating_count !== null && (
                    <div className="stat">
                      <div className="v">{data.movie.rating_count.toLocaleString()}</div>
                      <div className="k">ratings</div>
                    </div>
                  )}
                  {data.movie.rating_mean !== null && (
                    <div className="stat">
                      <div className="v">{data.movie.rating_mean.toFixed(1)}</div>
                      <div className="k">mean / 5</div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <p className={`overview${data.movie.overview ? "" : " muted"}`}>
              {data.movie.overview ??
                "No plot synopsis available in offline mode (movielens_fallback). Add a TMDB_API_KEY and re-hydrate to populate overviews and posters."}
            </p>

            {data.movie.keywords.length > 0 && (
              <div className="meta-row" style={{ padding: "4px 24px 0" }}>
                {data.movie.keywords.slice(0, 12).map((k) => (
                  <span key={k} className="meta-tag">
                    #{k}
                  </span>
                ))}
              </div>
            )}

            <div className="mlt-section">
              {data.more_like_this.map((list) => (
                <MoreLikeThisRow key={list.basis} list={list} onNavigate={onNavigate} />
              ))}
            </div>
          </>
        )}
      </aside>
    </>
  );
}

function MoreLikeThisRow({
  list,
  onNavigate,
}: {
  list: SimilarList;
  onNavigate: (id: number) => void;
}) {
  return (
    <div>
      <div className={`mlt-head ${list.basis}`}>
        <span className="dot" />
        <h4>{list.label}</h4>
      </div>
      <div className="mlt-head" style={{ marginTop: -6, marginBottom: 10 }}>
        <span className="sub">{BASIS_SUB[list.basis] ?? list.basis}</span>
      </div>
      {list.results.length === 0 ? (
        <div className="mlt-empty">No neighbors available for this signal.</div>
      ) : (
        <div className="mlt-row">
          {list.results.map((m: MovieSummary) => (
            <div key={m.movie_id} className="mini-card" onClick={() => onNavigate(m.movie_id)}>
              <Poster title={m.title} year={m.release_year} posterUrl={m.poster_url} showText={false} />
              <div className="mini-title">{m.title}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
