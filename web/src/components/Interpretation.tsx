import type { QuerySpec } from "../lib/types";
import { removeChip, specToChips, type Chip } from "../lib/spec";

interface Props {
  spec: QuerySpec;
  parser: string;
  onSpecChange: (spec: QuerySpec) => void;
}

const PARSER_BADGE: Record<string, { cls: string; text: string }> = {
  claude: { cls: "claude", text: "Claude parse" },
  heuristic_fallback: { cls: "heuristic", text: "heuristic parse" },
  provided_spec: { cls: "provided", text: "edited filters" },
};

/**
 * Shows the parsed interpretation as editable chips. Removing a chip mutates
 * the spec and re-queries via `onSpecChange` — sent back in the `spec` field so
 * the backend skips re-parsing.
 */
export function Interpretation({ spec, parser, onSpecChange }: Props) {
  const chips = specToChips(spec);
  const badge = PARSER_BADGE[parser] ?? PARSER_BADGE.provided_spec;

  function drop(chip: Chip) {
    onSpecChange(removeChip(spec, chip));
  }

  return (
    <div className="interpretation">
      <div className="interpretation-head">
        <span className="h">We understood</span>
        <span className={`badge ${badge.cls}`}>{badge.text}</span>
        {parser === "heuristic_fallback" && (
          <span className="chips-empty" title="Set ANTHROPIC_API_KEY to enable Claude parsing">
            (offline — no ANTHROPIC_API_KEY)
          </span>
        )}
      </div>
      {chips.length === 0 ? (
        <div className="chips-empty">
          No structured filters — matching on the free-text description only.
        </div>
      ) : (
        <div className="chips">
          {chips.map((chip) => (
            <span key={`${chip.kind}:${chip.value}`} className={`chip ${chip.kind}`}>
              {chip.label}
              <button
                className="x"
                aria-label={`Remove ${chip.label}`}
                title="Remove and re-query"
                onClick={() => drop(chip)}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
