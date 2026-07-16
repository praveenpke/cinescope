import { useState } from "react";

// Fallback records (movielens_fallback source) have no poster_path, so most
// tiles render a deterministic gradient placeholder derived from the title.
// This must look intentional, not broken.

const PALETTES: ReadonlyArray<readonly [string, string]> = [
  ["#3a1c71", "#d76d77"],
  ["#0f2027", "#2c5364"],
  ["#42275a", "#734b6d"],
  ["#1a2a6c", "#b21f1f"],
  ["#134e5e", "#71b280"],
  ["#232526", "#414345"],
  ["#41295a", "#2f0743"],
  ["#16222a", "#3a6073"],
  ["#5f2c82", "#49a09d"],
  ["#2b5876", "#4e4376"],
];

function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

interface Props {
  title: string;
  year: number | null;
  posterUrl: string | null;
  showText?: boolean;
}

/** Movie poster with a styled gradient fallback when no image is available. */
export function Poster({ title, year, posterUrl, showText = true }: Props) {
  const [broken, setBroken] = useState(false);
  const [a, b] = PALETTES[hash(title) % PALETTES.length];

  if (posterUrl && !broken) {
    return (
      <div className="poster">
        <img src={posterUrl} alt={title} loading="lazy" onError={() => setBroken(true)} />
      </div>
    );
  }
  return (
    <div className="poster">
      <div
        className="poster-fallback"
        style={{ ["--fb-a" as string]: a, ["--fb-b" as string]: b }}
      >
        <span className="film-icon" aria-hidden>
          🎞
        </span>
        {showText && (
          <>
            <div className="ttl">{title}</div>
            {year !== null && <div className="yr">{year}</div>}
          </>
        )}
      </div>
    </div>
  );
}
