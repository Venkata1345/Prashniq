import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

function DifficultyDots({ level }) {
  return (
    <span className="flex items-center gap-1" title={`difficulty ${level}/5`}>
      {[1, 2, 3, 4, 5].map((dot) => (
        <span
          key={dot}
          className={`h-1.5 w-1.5 rounded-full ${
            dot <= level ? "bg-amber-400" : "bg-slate-700"
          }`}
        />
      ))}
    </span>
  );
}

function ThinkingBubble() {
  return (
    <div className="animate-rise flex justify-start">
      <div className="flex items-center gap-1.5 rounded-2xl rounded-bl-md border border-slate-800 bg-slate-900/80 px-5 py-4">
        {[0, 1, 2].map((dot) => (
          <span key={dot} className="thinking-dot h-2 w-2 rounded-full bg-indigo-400" />
        ))}
        <span className="ml-2 text-xs text-slate-500">interviewer is thinking</span>
      </div>
    </div>
  );
}

export default function InterviewView({ session, onFinished, toast }) {
  const [messages, setMessages] = useState([]); // {role: 'q'|'a', text, meta?}
  const [question, setQuestion] = useState(null);
  const [answer, setAnswer] = useState("");
  const [thinking, setThinking] = useState(true);
  const [answered, setAnswered] = useState(0);
  const [ending, setEnding] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const startedRef = useRef(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, thinking]);

  useEffect(() => {
    if (startedRef.current) return; // StrictMode double-mount guard
    startedRef.current = true;
    api(`/interviews/${session.interviewId}/start`, { method: "POST" })
      .then((firstQuestion) => {
        setQuestion(firstQuestion);
        setMessages([{ role: "q", text: firstQuestion.text, meta: firstQuestion }]);
      })
      .catch((error) => toast(error.message))
      .finally(() => {
        setThinking(false);
        inputRef.current?.focus();
      });
  }, [session.interviewId, toast]);

  const submit = async () => {
    const text = answer.trim();
    if (!text || thinking) return;
    setAnswer("");
    setMessages((prev) => [...prev, { role: "a", text }]);
    setThinking(true);
    try {
      const result = await api(`/interviews/${session.interviewId}/answers`, {
        method: "POST",
        body: JSON.stringify({ answer: text }),
      });
      setAnswered((count) => count + 1);
      if (result.interview_complete || !result.next_question) {
        await onFinished(session.interviewId);
        return;
      }
      setQuestion(result.next_question);
      setMessages((prev) => [
        ...prev,
        { role: "q", text: result.next_question.text, meta: result.next_question },
      ]);
    } catch (error) {
      toast(error.message);
      setAnswer(text); // let the candidate retry rather than lose the text
    } finally {
      setThinking(false);
      inputRef.current?.focus();
    }
  };

  const endInterview = async () => {
    if (
      answered === 0 &&
      !window.confirm("End before answering anything? There will be no report scores.")
    ) {
      return;
    }
    setEnding(true);
    try {
      await api(`/interviews/${session.interviewId}/complete`, { method: "POST" });
      await onFinished(session.interviewId);
    } catch (error) {
      toast(error.message);
      setEnding(false);
    }
  };

  return (
    <div className="animate-rise flex min-h-[calc(100vh-4rem)] flex-col">
      <header className="sticky top-0 z-10 -mx-4 mb-4 border-b border-slate-800/80 bg-slate-950/85 px-4 py-3 backdrop-blur sm:-mx-6 sm:px-6">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <span className="shrink-0 rounded-full border border-indigo-500/40 bg-indigo-500/15 px-3 py-1 text-xs font-semibold text-indigo-300">
              {session.mode.display_name}
            </span>
            <span className="truncate text-sm font-medium text-slate-300">
              {question?.topic ?? ""}
            </span>
          </div>
          <div className="flex shrink-0 items-center gap-4">
            {question && <DifficultyDots level={question.difficulty} />}
            {question && (
              <span className="text-xs text-slate-500">
                Q{question.index}
                {session.mode.max_questions ? ` of ≤${session.mode.max_questions}` : ""}
              </span>
            )}
          </div>
        </div>
      </header>

      <div className="flex-1 space-y-4 pb-4" aria-live="polite">
        {messages.map((message, index) =>
          message.role === "q" ? (
            <div key={index} className="animate-rise flex justify-start">
              <div className="max-w-[85%] rounded-2xl rounded-bl-md border border-slate-800 bg-slate-900/80 px-5 py-4 shadow-lg">
                <div className="mb-1.5 text-[11px] font-semibold tracking-wide text-indigo-400 uppercase">
                  Q{message.meta.index} · {message.meta.topic}
                </div>
                <div className="leading-relaxed whitespace-pre-wrap text-slate-200">
                  {message.text}
                </div>
              </div>
            </div>
          ) : (
            <div key={index} className="animate-rise flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-br-md bg-gradient-to-br from-indigo-600 to-violet-600 px-5 py-4 leading-relaxed whitespace-pre-wrap text-white shadow-lg shadow-indigo-900/40">
                {message.text}
              </div>
            </div>
          )
        )}
        {thinking && <ThinkingBubble />}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
        className="sticky bottom-0 -mx-4 border-t border-slate-800/80 bg-slate-950/85 px-4 py-4 backdrop-blur sm:-mx-6 sm:px-6"
      >
        <div className="mx-auto max-w-3xl">
          <textarea
            ref={inputRef}
            rows={4}
            value={answer}
            onChange={(event) => setAnswer(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder="Type your answer… (Ctrl+Enter to submit)"
            className="w-full resize-y rounded-xl border border-slate-700 bg-slate-900/80 px-4 py-3 text-slate-200 placeholder-slate-600 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/30"
          />
          <div className="mt-3 flex items-center justify-between">
            <button
              type="button"
              onClick={endInterview}
              disabled={ending}
              className="rounded-lg border border-slate-700 px-4 py-2 text-sm font-medium text-slate-400 transition hover:border-slate-500 hover:text-slate-200 disabled:opacity-50"
            >
              {ending ? "Wrapping up…" : "End interview"}
            </button>
            <button
              type="submit"
              disabled={thinking || !answer.trim()}
              className="rounded-lg bg-gradient-to-r from-indigo-500 to-violet-500 px-6 py-2 text-sm font-semibold text-white shadow-md shadow-indigo-500/25 transition hover:from-indigo-400 hover:to-violet-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Submit answer
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
