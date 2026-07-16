import type { DiscoverResult } from "../lib/types";
import { Poster } from "./Poster";
import { WhySignals } from "./WhySignals";

interface Props {
  movie: DiscoverResult;
  onOpen: (id: number) => void;
}

export function ResultCard({ movie, onOpen }: Props) {
  return (
    <div
      className="card"
      role="button"
      tabIndex={0}
      onClick={() => onOpen(movie.movie_id)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen(movie.movie_id);
        }
      }}
    >
      <div style={{ position: "relative" }}>
        <Poster title={movie.title} year={movie.release_year} posterUrl={movie.poster_url} />
        <span className="score-badge" title="Hybrid rank score">
          {Math.round(movie.score * 100)}
        </span>
      </div>
      <div className="card-body">
        <div className="card-title">{movie.title}</div>
        <div className="card-sub">
          {movie.release_year !== null && <span>{movie.release_year}</span>}
          {movie.genres.length > 0 && <span>{movie.genres.slice(0, 2).join(" · ")}</span>}
        </div>
        <WhySignals why={movie.why} />
      </div>
    </div>
  );
}
