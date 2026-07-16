import { useState } from "react";

const EXAMPLES = [
  "like Inception but funnier",
  "dark sci-fi thrillers from the 90s",
  "feel-good comedies, nothing scary",
  "movies fans of The Matrix love",
];

interface Props {
  onSearch: (query: string) => void;
  loading: boolean;
}

export function SearchBar({ onSearch, loading }: Props) {
  const [text, setText] = useState("");

  function submit(q: string) {
    const trimmed = q.trim();
    if (trimmed) onSearch(trimmed);
  }

  return (
    <div className="hero">
      <h2>
        Describe a movie. <span className="grad">Find it.</span>
      </h2>
      <p>
        Natural-language discovery over 25M+ MovieLens ratings and TMDB titles — hybrid semantic
        embeddings + collaborative-filtering signals.
      </p>
      <form
        className="searchbar"
        onSubmit={(e) => {
          e.preventDefault();
          submit(text);
        }}
      >
        <input
          type="text"
          value={text}
          placeholder="like Inception but funnier"
          onChange={(e) => setText(e.target.value)}
          aria-label="Search movies by description"
          autoFocus
        />
        <button type="submit" disabled={loading || !text.trim()}>
          {loading ? "Searching…" : "Discover"}
        </button>
      </form>
      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            className="example"
            onClick={() => {
              setText(ex);
              submit(ex);
            }}
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
