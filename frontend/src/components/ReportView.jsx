import { useEffect, useState } from "react";

function ScoreRing({ score }) {
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    const raf = requestAnimationFrame(() => setAnimated(score));
    return () => cancelAnimationFrame(raf);
  }, [score]);

  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const fraction = Math.min(1, Math.max(0, animated / 10));
  const hue = score >= 7 ? "stroke-emerald-400" : score >= 4 ? "stroke-amber-400" : "stroke-red-400";

  return (
    <div className="relative h-36 w-36">
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle cx="64" cy="64" r={radius} fill="none" strokeWidth="10" className="stroke-slate-800" />
        <circle
          cx="64"
          cy="64"
          r={radius}
          fill="none"
          strokeWidth="10"
          strokeLinecap="round"
          className={`${hue} transition-[stroke-dashoffset] duration-1000 ease-out`}
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - fraction)}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-extrabold text-white">{score.toFixed(1)}</span>
        <span className="text-xs text-slate-500">/ 10</span>
      </div>
    </div>
  );
}

function Bars({ entries }) {
  const [grown, setGrown] = useState(false);
  useEffect(() => {
    const raf = requestAnimationFrame(() => setGrown(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="space-y-3">
      {entries.map(([name, score]) => (
        <div key={name} className="flex items-center gap-3">
          <span className="w-48 shrink-0 text-sm leading-tight text-slate-400 capitalize">
            {name.replaceAll("_", " ")}
          </span>
          <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-400 transition-[width] duration-1000 ease-out"
              style={{ width: grown ? `${Math.min(100, score * 10)}%` : "0%" }}
            />
          </div>
          <span className="w-9 shrink-0 text-right text-sm font-semibold text-slate-300">
            {score.toFixed(1)}
          </span>
        </div>
      ))}
    </div>
  );
}

function ChipCard({ title, items, emptyText, tone }) {
  const toneClass =
    tone === "good"
      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
      : tone === "bad"
        ? "border-red-500/30 bg-red-500/10 text-red-300"
        : "border-slate-600/40 bg-slate-700/20 text-slate-300";
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur">
      <h2 className="mb-3 text-sm font-semibold tracking-wide text-slate-400 uppercase">{title}</h2>
      {!items || items.length === 0 ? (
        <p className="text-sm text-slate-600 italic">{emptyText}</p>
      ) : (
        <ul className="flex flex-wrap gap-2">
          {items.map((item) => (
            <li
              key={item}
              className={`rounded-full border px-3 py-1 text-xs font-medium ${toneClass}`}
            >
              {item}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function ReportView({ report, profile, mode, onRestart }) {
  return (
    <div className="animate-rise space-y-6 pb-10">
      <header className="pt-6 text-center">
        <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
          Interview report
        </h1>
        <p className="mt-2 text-sm text-slate-400">
          {mode?.display_name ?? report.interview_type} · {report.questions_answered} answered ·{" "}
          {Math.round(report.duration_seconds / 60)} min
        </p>
      </header>

      <section className="flex flex-col items-center gap-8 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur sm:flex-row">
        <ScoreRing score={report.overall_score} />
        <div className="w-full flex-1">
          <Bars entries={Object.entries(report.dimension_scores)} />
        </div>
      </section>

      <div className="grid gap-4 sm:grid-cols-2">
        <ChipCard title="Strengths" items={report.strengths} emptyText="none recorded" tone="good" />
        <ChipCard title="Weaknesses" items={report.weaknesses} emptyText="none recorded" tone="bad" />
        <ChipCard title="Missed concepts" items={report.missed_concepts} emptyText="nothing missed" />
        <ChipCard
          title="Role skills not reached"
          items={report.unaddressed_target_skills}
          emptyText="all planned skills reached"
        />
      </div>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur">
        <h2 className="mb-4 text-sm font-semibold tracking-wide text-slate-400 uppercase">
          Question-by-question evidence
        </h2>
        <ol className="space-y-3">
          {report.evidence.map((line, index) => {
            const [head, ...rest] = line.split(" — ");
            return (
              <li
                key={index}
                className="rounded-xl border border-slate-800/70 bg-slate-950/50 px-4 py-3 text-sm leading-relaxed"
              >
                <strong className="text-indigo-300">{head}</strong>
                {rest.length > 0 && <span className="text-slate-400"> — {rest.join(" — ")}</span>}
              </li>
            );
          })}
        </ol>
      </section>

      {profile && profile.topics.length > 0 && (
        <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur">
          <h2 className="mb-4 text-sm font-semibold tracking-wide text-slate-400 uppercase">
            Skill profile <span className="font-normal text-slate-600 normal-case">across all interviews</span>
          </h2>
          <Bars entries={profile.topics.map((entry) => [entry.concept, entry.score])} />
          {profile.recommended_focus.length > 0 && (
            <p className="mt-4 text-sm text-slate-500">
              Recommended focus next: {profile.recommended_focus.slice(0, 5).join(", ")}
            </p>
          )}
        </section>
      )}

      <div className="flex justify-center pt-2">
        <button
          type="button"
          onClick={onRestart}
          className="rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 px-10 py-3.5 text-base font-semibold text-white shadow-lg shadow-indigo-500/25 transition hover:from-indigo-400 hover:to-violet-400 hover:shadow-indigo-500/40"
        >
          New interview
        </button>
      </div>
    </div>
  );
}
