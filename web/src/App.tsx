import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { discover, fetchHealth } from "./lib/api";
import type { DiscoverRequest, DiscoverResponse, QuerySpec } from "./lib/types";
import { SearchBar } from "./components/SearchBar";
import { Interpretation } from "./components/Interpretation";
import { ResultCard } from "./components/ResultCard";
import { DetailDrawer } from "./components/DetailDrawer";

export default function App() {
  const [response, setResponse] = useState<DiscoverResponse | null>(null);
  const [openId, setOpenId] = useState<number | null>(null);

  const health = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    retry: false,
    staleTime: 30_000,
  });

  const search = useMutation({
    mutationFn: (req: DiscoverRequest) => discover(req),
    onSuccess: (data) => setResponse(data),
  });

  function runQuery(query: string) {
    search.mutate({ query });
  }

  // Chip edits re-query by POSTing the modified spec back (no re-parse).
  function runSpec(spec: QuerySpec) {
    search.mutate({ spec });
  }

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <h1>
            Cine<span className="lens">Scope</span>
          </h1>
          <span className="tag">semantic movie discovery</span>
        </div>
        <StatusPill
          ok={health.isSuccess}
          down={health.isError}
          text={
            health.isSuccess
              ? `${health.data.titles.toLocaleString()} titles · ${health.data.table}`
              : health.isError
                ? "API offline"
                : "connecting…"
          }
        />
      </header>

      <SearchBar onSearch={runQuery} loading={search.isPending} />

      {response && (
        <Interpretation
          spec={response.spec}
          parser={response.parser}
          onSpecChange={runSpec}
        />
      )}

      <ResultsArea
        response={response}
        loading={search.isPending}
        error={search.isError ? (search.error as Error) : null}
        onOpen={setOpenId}
      />

      {openId !== null && (
        <DetailDrawer movieId={openId} onClose={() => setOpenId(null)} onNavigate={setOpenId} />
      )}

      <footer className="foot">
        CineScope — PySpark · FastAPI · sentence-transformers · pgvector · React ·{" "}
        <a href="https://github.com/praveenpke/cinescope" target="_blank" rel="noreferrer">
          source
        </a>
      </footer>
    </div>
  );
}

function StatusPill({ ok, down, text }: { ok: boolean; down: boolean; text: string }) {
  const cls = ok ? "ok" : down ? "down" : "";
  return (
    <span className="status-pill">
      <span className={`status-dot ${cls}`} />
      {text}
    </span>
  );
}

interface ResultsProps {
  response: DiscoverResponse | null;
  loading: boolean;
  error: Error | null;
  onOpen: (id: number) => void;
}

function ResultsArea({ response, loading, error, onOpen }: ResultsProps) {
  if (loading) {
    return (
      <div className="skeleton-grid">
        {Array.from({ length: 10 }).map((_, i) => (
          <div key={i} className="skeleton">
            <div className="sk-poster" />
            <div className="sk-line" style={{ width: "80%" }} />
            <div className="sk-line" style={{ width: "50%" }} />
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="state error">
        <div className="big">⚠</div>
        <h4>Search failed</h4>
        <p>{error.message}</p>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="state">
        <div className="big">🔍</div>
        <h4>Start with a description</h4>
        <p>Try one of the examples above, or describe the movie you’re in the mood for.</p>
      </div>
    );
  }

  if (response.results.length === 0) {
    return (
      <div className="state">
        <div className="big">🎬</div>
        <h4>No matches</h4>
        <p>
          Nothing in the {response.table} catalog fit those filters. Try removing a chip — high
          rating floors often return nothing in the 675-title sample.
        </p>
      </div>
    );
  }

  return (
    <>
      <div className="results-head">
        <h3>Results</h3>
        <span className="count">
          {response.results.length} title{response.results.length === 1 ? "" : "s"} · ranked by
          hybrid score
        </span>
      </div>
      <div className="grid">
        {response.results.map((m) => (
          <ResultCard key={m.movie_id} movie={m} onOpen={onOpen} />
        ))}
      </div>
    </>
  );
}
