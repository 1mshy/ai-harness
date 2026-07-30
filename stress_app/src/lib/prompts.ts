import type { Difficulty, PromptItem } from "./types";

interface RawFile {
  difficulty?: string;
  prompts?: Array<{
    id: string;
    title?: string;
    category?: string;
    text: string;
    difficulty?: string;
    targetTokens?: number;
  }>;
}

/**
 * The corpus is authored as JSON under src/data/prompts and bundled at build
 * time. Globbing rather than naming each file keeps the loader stable as the
 * corpus is split or extended.
 */
const modules = import.meta.glob<RawFile>("../data/prompts/*.json", {
  eager: true,
  import: "default",
});

function normalise(d: string | undefined): Difficulty {
  if (d === "easy" || d === "medium" || d === "hard") return d;
  return "medium";
}

function load(): PromptItem[] {
  const out: PromptItem[] = [];
  const seen = new Set<string>();

  for (const [path, mod] of Object.entries(modules)) {
    const fileDifficulty = normalise(mod?.difficulty);
    for (const p of mod?.prompts ?? []) {
      if (!p?.id || !p?.text) continue;
      if (seen.has(p.id)) {
        console.warn(`[prompts] duplicate id ${p.id} in ${path} — skipped`);
        continue;
      }
      seen.add(p.id);
      out.push({
        id: p.id,
        title: p.title?.trim() || p.id,
        category: p.category?.trim() || "uncategorised",
        text: p.text,
        difficulty: normalise(p.difficulty ?? fileDifficulty),
        targetTokens: p.targetTokens ?? 0,
      });
    }
  }

  const rank: Record<Difficulty, number> = { easy: 0, medium: 1, hard: 2 };
  out.sort(
    (a, b) => rank[a.difficulty] - rank[b.difficulty] || a.id.localeCompare(b.id),
  );
  return out;
}

export const ALL_PROMPTS: PromptItem[] = load();

export const PROMPT_COUNTS: Record<Difficulty, number> = {
  easy: ALL_PROMPTS.filter((p) => p.difficulty === "easy").length,
  medium: ALL_PROMPTS.filter((p) => p.difficulty === "medium").length,
  hard: ALL_PROMPTS.filter((p) => p.difficulty === "hard").length,
};

export const CATEGORIES: string[] = Array.from(
  new Set(ALL_PROMPTS.map((p) => p.category)),
).sort();
