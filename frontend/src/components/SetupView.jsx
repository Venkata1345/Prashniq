import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

function PdfUploadButton({ label, onText, toast }) {
  const inputRef = useRef(null);
  const [busy, setBusy] = useState(false);

  const upload = async (file) => {
    if (!file) return;
    setBusy(true);
    try {
      const body = new FormData();
      body.append("file", file);
      // no JSON content-type here: the browser sets the multipart boundary
      const response = await fetch("/candidate-contexts/extract-pdf", {
        method: "POST",
        body,
      });
      if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
          const data = await response.json();
          if (typeof data.detail === "string") detail = data.detail;
        } catch {
          /* keep the status text */
        }
        throw new Error(detail);
      }
      const extracted = await response.json();
      onText(extracted.text);
      toast(
        `Loaded ${file.name} (${extracted.pages} page${extracted.pages === 1 ? "" : "s"})`,
        "success"
      );
    } catch (error) {
      toast(error.message);
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,application/pdf"
        className="hidden"
        onChange={(event) => upload(event.target.files?.[0])}
      />
      <button
        type="button"
        disabled={busy}
        onClick={() => inputRef.current?.click()}
        className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:border-indigo-500 hover:text-indigo-300 disabled:opacity-50"
      >
        <svg viewBox="0 0 16 16" className="h-3.5 w-3.5 fill-current" aria-hidden="true">
          <path d="M8 10.5a.75.75 0 0 1-.75-.75V3.56L5.53 5.28a.75.75 0 0 1-1.06-1.06l3-3a.75.75 0 0 1 1.06 0l3 3a.75.75 0 1 1-1.06 1.06L8.75 3.56v6.19a.75.75 0 0 1-.75.75Z" />
          <path d="M2.75 10a.75.75 0 0 1 .75.75v1.5c0 .41.34.75.75.75h7.5c.41 0 .75-.34.75-.75v-1.5a.75.75 0 0 1 1.5 0v1.5A2.25 2.25 0 0 1 11.75 14.5h-7.5A2.25 2.25 0 0 1 2 12.25v-1.5a.75.75 0 0 1 .75-.75Z" />
        </svg>
        {busy ? "Extracting…" : label}
      </button>
    </>
  );
}

function ModelBadge({ health }) {
  if (!health?.llm_model) return null;
  const limits = health.llm_rate_limits;
  return (
    <div className="pb-8 text-center text-xs leading-relaxed text-slate-600">
      <p>
        Powered by <span className="text-slate-400">{health.llm_model}</span> via{" "}
        <span className="capitalize text-slate-400">{health.llm_provider}</span>
        {health.retrieval_ready && " · grounded in a 16k-chunk ML textbook corpus"}
      </p>
      {limits && (
        <p className="mt-1">
          Free demo — shared limits of {limits.requests_per_minute} requests and{" "}
          {limits.tokens_per_minute.toLocaleString()} tokens per minute. Under load, the
          interviewer may take a little longer between questions.
        </p>
      )}
    </div>
  );
}

export default function SetupView({ modes, onStarted, toast, health }) {
  const [selectedKey, setSelectedKey] = useState(null);
  const [candidateId, setCandidateId] = useState("");
  const [resume, setResume] = useState("");
  const [jd, setJd] = useState("");
  const [contextOpen, setContextOpen] = useState(false);
  const [starting, setStarting] = useState(false);

  const selectedMode = modes.find((mode) => mode.key === selectedKey) ?? null;

  useEffect(() => {
    if (!selectedKey && modes.length) setSelectedKey(modes[0].key);
  }, [modes, selectedKey]);

  const selectMode = (mode) => {
    setSelectedKey(mode.key);
    // Context-driven modes cannot run without a resume/JD; open the section.
    if (mode.topics.length === 0) setContextOpen(true);
  };

  const start = async () => {
    if (!selectedMode) {
      toast("Pick an interview type first");
      return;
    }
    setStarting(true);
    try {
      const cid = candidateId.trim() || null;
      let contextId = null;
      if (resume.trim() || jd.trim()) {
        const context = await api("/candidate-contexts", {
          method: "POST",
          body: JSON.stringify({
            candidate_id: cid,
            resume_text: resume.trim() || null,
            job_description_text: jd.trim() || null,
          }),
        });
        contextId = context.context_id;
      }
      const interview = await api("/interviews", {
        method: "POST",
        body: JSON.stringify({
          interview_type: selectedMode.key,
          candidate_id: cid,
          context_id: contextId,
        }),
      });
      onStarted({
        interviewId: interview.interview_id,
        mode: selectedMode,
        candidateId: cid,
      });
    } catch (error) {
      toast(error.message);
      setStarting(false);
    }
  };

  return (
    <div className="animate-rise space-y-6">
      <header className="pt-6 pb-2 text-center">
        <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-indigo-500/30 bg-indigo-500/10 px-4 py-1.5 text-xs font-medium tracking-wide text-indigo-300">
          <span className="h-1.5 w-1.5 rounded-full bg-indigo-400" />
          Adaptive · RAG-grounded · Feedback at the end
        </div>
        <h1 className="bg-gradient-to-r from-white via-indigo-100 to-indigo-300 bg-clip-text text-4xl font-extrabold tracking-tight text-transparent sm:text-5xl">
          Prashniq
        </h1>
        <p className="mx-auto mt-3 text-sm tracking-wide text-indigo-300/80">
          prashna (प्रश्न) · Sanskrit for “question”
        </p>
        <p className="mx-auto mt-4 max-w-xl text-base text-slate-400">
          An adaptive AI interviewer for AI/ML roles. It probes every answer and
          holds the feedback until the end — just like a real interview.
        </p>
      </header>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur">
        <h2 className="mb-4 text-sm font-semibold tracking-wide text-slate-400 uppercase">
          <span className="mr-2 text-indigo-400">1</span>Interview type
        </h2>
        <div role="radiogroup" aria-label="Interview type" className="grid gap-3 sm:grid-cols-2">
          {modes.map((mode) => {
            const selected = mode.key === selectedKey;
            return (
              <button
                key={mode.key}
                type="button"
                role="radio"
                aria-checked={selected}
                onClick={() => selectMode(mode)}
                className={`group rounded-xl border p-4 text-left transition-all duration-150 ${
                  selected
                    ? "border-indigo-500 bg-indigo-500/15 shadow-lg shadow-indigo-500/10"
                    : "border-slate-800 bg-slate-900/40 hover:border-slate-600 hover:bg-slate-800/60"
                }`}
              >
                <div className={`font-semibold ${selected ? "text-indigo-200" : "text-slate-200"}`}>
                  {mode.display_name}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  up to {mode.max_questions} questions
                  {mode.topics.length === 0 && " · needs a resume or JD"}
                </div>
              </button>
            );
          })}
          {modes.length === 0 && (
            <div className="col-span-full py-6 text-center text-sm text-slate-500">
              Loading interview types…
            </div>
          )}
        </div>
      </section>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur">
        <h2 className="mb-1 text-sm font-semibold tracking-wide text-slate-400 uppercase">
          <span className="mr-2 text-indigo-400">2</span>Who is interviewing?{" "}
          <span className="font-normal normal-case text-slate-600">(optional)</span>
        </h2>
        <p className="mb-4 text-sm text-slate-500">
          With a candidate id, results accumulate into a skill profile and future
          interviews steer toward your weak areas.
        </p>
        <input
          type="text"
          value={candidateId}
          onChange={(event) => setCandidateId(event.target.value)}
          placeholder="candidate id, e.g. abhishek"
          autoComplete="off"
          spellCheck={false}
          className="w-full rounded-xl border border-slate-700 bg-slate-950/70 px-4 py-3 text-slate-200 placeholder-slate-600 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/30"
        />
      </section>

      <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur">
        <h2 className="mb-1 text-sm font-semibold tracking-wide text-slate-400 uppercase">
          <span className="mr-2 text-indigo-400">3</span>Target a role{" "}
          <span className="font-normal normal-case text-slate-600">(optional)</span>
        </h2>
        <p className="mb-4 text-sm text-slate-500">
          Paste a resume and/or job description and the interview is planned around
          the role's must-haves and your own claims.
        </p>
        <button
          type="button"
          onClick={() => setContextOpen((open) => !open)}
          className="flex items-center gap-2 text-sm font-medium text-indigo-400 transition hover:text-indigo-300"
        >
          <span
            className={`inline-block transition-transform duration-200 ${contextOpen ? "rotate-90" : ""}`}
          >
            ▶
          </span>
          Add resume / job description
        </button>
        {contextOpen && (
          <div className="animate-rise mt-4 space-y-4">
            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <label htmlFor="resume-text" className="text-sm font-medium text-slate-400">
                  Resume (paste text or upload a PDF)
                </label>
                <PdfUploadButton
                  label="Upload PDF"
                  toast={toast}
                  onText={(text) => setResume(text)}
                />
              </div>
              <textarea
                id="resume-text"
                rows={6}
                value={resume}
                onChange={(event) => setResume(event.target.value)}
                placeholder="Built a RAG system using FAISS and FastAPI serving 200 rps..."
                className="w-full resize-y rounded-xl border border-slate-700 bg-slate-950/70 px-4 py-3 text-sm text-slate-200 placeholder-slate-600 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/30"
              />
            </div>
            <div>
              <label htmlFor="jd-text" className="mb-1.5 block text-sm font-medium text-slate-400">
                Job description
              </label>
              <textarea
                id="jd-text"
                rows={5}
                value={jd}
                onChange={(event) => setJd(event.target.value)}
                placeholder="Must have: production RAG experience, retrieval evaluation..."
                className="w-full resize-y rounded-xl border border-slate-700 bg-slate-950/70 px-4 py-3 text-sm text-slate-200 placeholder-slate-600 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/30"
              />
            </div>
          </div>
        )}
      </section>

      <div className="flex justify-center pt-2 pb-8">
        <button
          type="button"
          onClick={start}
          disabled={starting || !selectedMode}
          className="rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 px-10 py-3.5 text-base font-semibold text-white shadow-lg shadow-indigo-500/25 transition hover:from-indigo-400 hover:to-violet-400 hover:shadow-indigo-500/40 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {starting ? "Preparing your interview…" : "Start interview"}
        </button>
      </div>

      <ModelBadge health={health} />
    </div>
  );
}
