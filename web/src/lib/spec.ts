import type { QuerySpec } from "./types";

// A "chip" is one editable atom of the parsed interpretation. Removing or
// editing a chip produces a new QuerySpec that is POSTed back in the `spec`
// field, so the backend re-queries WITHOUT re-parsing (parser='provided_spec').

export type ChipKind =
  | "reference"
  | "mood"
  | "genre_include"
  | "genre_exclude"
  | "year"
  | "min_rating";

export interface Chip {
  kind: ChipKind;
  label: string; // what the user sees
  value: string; // the underlying token (for list kinds)
}

const YEAR_KINDS: ReadonlySet<ChipKind> = new Set(["year", "min_rating"]);

/** Flatten a spec into the ordered list of chips shown under the search box. */
export function specToChips(spec: QuerySpec): Chip[] {
  const chips: Chip[] = [];
  for (const t of spec.reference_titles) {
    chips.push({ kind: "reference", label: `like ${t}`, value: t });
  }
  for (const m of spec.mood_adjustments) {
    chips.push({ kind: "mood", label: m, value: m });
  }
  for (const g of spec.genres_include) {
    chips.push({ kind: "genre_include", label: g, value: g });
  }
  for (const g of spec.genres_exclude) {
    chips.push({ kind: "genre_exclude", label: `not ${g}`, value: g });
  }
  if (spec.year_range && (spec.year_range.start !== null || spec.year_range.end !== null)) {
    const { start, end } = spec.year_range;
    let label: string;
    if (start !== null && end !== null) label = start === end ? `${start}` : `${start}–${end}`;
    else if (start !== null) label = `${start}+`;
    else label = `up to ${end}`;
    chips.push({ kind: "year", label: `year: ${label}`, value: "year" });
  }
  if (spec.min_rating !== null) {
    chips.push({ kind: "min_rating", label: `rating ≥ ${spec.min_rating}/10`, value: "min_rating" });
  }
  return chips;
}

/** Immutably remove one chip from the spec, returning a fresh spec. */
export function removeChip(spec: QuerySpec, chip: Chip): QuerySpec {
  const next: QuerySpec = structuredClone(spec);
  switch (chip.kind) {
    case "reference":
      next.reference_titles = next.reference_titles.filter((v) => v !== chip.value);
      break;
    case "mood":
      next.mood_adjustments = next.mood_adjustments.filter((v) => v !== chip.value);
      break;
    case "genre_include":
      next.genres_include = next.genres_include.filter((v) => v !== chip.value);
      break;
    case "genre_exclude":
      next.genres_exclude = next.genres_exclude.filter((v) => v !== chip.value);
      break;
    case "year":
      next.year_range = null;
      break;
    case "min_rating":
      next.min_rating = null;
      break;
  }
  return next;
}

export function isScalarChip(kind: ChipKind): boolean {
  return YEAR_KINDS.has(kind);
}

/** True when the spec carries no structured filters worth showing chips for. */
export function specIsEmpty(spec: QuerySpec): boolean {
  return specToChips(spec).length === 0;
}
