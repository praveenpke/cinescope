import type { Why } from "../lib/types";

/** Compact "why this matched" signal chips shown on each result card. */
export function WhySignals({ why }: { why: Why }) {
  return (
    <div className="why">
      {why.semantic_similarity !== null && (
        <span className="sig semantic" title="Cosine similarity of query text to plot/genre embedding">
          ◈ {Math.round(why.semantic_similarity * 100)}% match
        </span>
      )}
      {why.behavioral_boost !== null && (
        <span
          className="sig behavioral"
          title={
            why.liked_by_fans_of.length
              ? `Liked by fans of ${why.liked_by_fans_of.join(", ")}`
              : "Collaborative-filtering signal"
          }
        >
          ♥ fans of {why.liked_by_fans_of[0] ?? "similar"}
        </span>
      )}
      {why.quality_score !== null && (
        <span className="sig quality" title="Bayesian-weighted MovieLens rating">
          ★ {why.quality_score.toFixed(1)}
        </span>
      )}
      {why.matched_filters.map((f) => (
        <span key={f} className="sig filter">
          {f}
        </span>
      ))}
    </div>
  );
}
