import { useCallback, useEffect, useRef, useState } from "react";
import SetupView from "./components/SetupView.jsx";
import InterviewView from "./components/InterviewView.jsx";
import ReportView from "./components/ReportView.jsx";
import { api } from "./api.js";

// ?preview=report renders the report screen with sample data (design review
// without running a paid interview). Not linked from anywhere in the UI.
const SAMPLE_REPORT = {
  interview_type: "ml_fundamentals",
  questions_answered: 8,
  duration_seconds: 1560,
  overall_score: 8.0,
  dimension_scores: {
    technical_correctness: 8.4,
    technical_depth: 7.6,
    communication: 8.5,
    reasoning: 8.1,
    tradeoff_awareness: 7.4,
  },
  strengths: [
    "regularization intuition",
    "clear structured answers",
    "connects theory to production",
  ],
  weaknesses: ["calibration metrics", "hand-wavy on class imbalance"],
  missed_concepts: ["Brier score", "stratified cross-validation"],
  unaddressed_target_skills: ["feature stores"],
  evidence: [
    "Q1 · bias-variance tradeoff — strong: derived the decomposition and tied it to model selection.",
    "Q2 · regularization — strong: contrasted L1 and L2 geometrically, mentioned elastic net.",
    "Q3 · model evaluation metrics — adequate: chose ROC-AUC correctly but could not define calibration.",
    "Q4 · overfitting and validation strategy — strong: proposed nested CV for small data.",
    "Q5 · feature engineering — adequate: solid pipeline hygiene, vague on target leakage detection.",
    "Q6 · model evaluation metrics — weak: did not know the Brier score; guessed at calibration curves.",
    "Q7 · regularization — strong: explained dropout as an ensemble approximation.",
    "Q8 · bias-variance tradeoff — strong: reasoned about double descent when pressed.",
  ],
};

const SAMPLE_PROFILE = {
  topics: [
    { concept: "regularization", score: 8.6 },
    { concept: "bias-variance tradeoff", score: 8.2 },
    { concept: "overfitting and validation strategy", score: 7.9 },
    { concept: "feature engineering", score: 6.8 },
    { concept: "model evaluation metrics", score: 5.9 },
  ],
  recommended_focus: ["model evaluation metrics", "feature engineering"],
};

const SAMPLE_MODE = { display_name: "AI/ML Fundamentals" };

const isReportPreview =
  new URLSearchParams(window.location.search).get("preview") === "report";

export default function App() {
  const [view, setView] = useState(isReportPreview ? "report" : "setup"); // setup | interview | report
  const [modes, setModes] = useState([]);
  const [toastMessage, setToastMessage] = useState(null);
  const toastTimer = useRef(null);

  // Session shared across views
  const [session, setSession] = useState(isReportPreview ? { mode: SAMPLE_MODE } : null);
  const [report, setReport] = useState(isReportPreview ? SAMPLE_REPORT : null);
  const [profile, setProfile] = useState(isReportPreview ? SAMPLE_PROFILE : null);

  const toast = useCallback((message, tone = "error") => {
    setToastMessage({ message, tone });
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToastMessage(null), 6000);
  }, []);

  const [health, setHealth] = useState(null);

  useEffect(() => {
    api("/interview-types")
      .then(setModes)
      .catch((error) => toast(`Could not load interview types: ${error.message}`));
    api("/health")
      .then(setHealth)
      .catch(() => {}); // the badge is informational; never block the app on it
  }, [toast]);

  const handleStarted = (newSession) => {
    setSession(newSession);
    setView("interview");
    window.scrollTo({ top: 0 });
  };

  const handleFinished = async (interviewId) => {
    const reportData = await api(`/interviews/${interviewId}/report`);
    setReport(reportData);
    let profileData = null;
    if (session?.candidateId) {
      try {
        profileData = await api(
          `/candidates/${encodeURIComponent(session.candidateId)}/profile`
        );
      } catch {
        /* the profile is a bonus; never block the report on it */
      }
    }
    setProfile(profileData);
    setView("report");
    window.scrollTo({ top: 0 });
  };

  const handleRestart = () => {
    if (isReportPreview) window.history.replaceState(null, "", window.location.pathname);
    setSession(null);
    setReport(null);
    setProfile(null);
    setView("setup");
    window.scrollTo({ top: 0 });
  };

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
      {view === "setup" && (
        <SetupView modes={modes} onStarted={handleStarted} toast={toast} health={health} />
      )}
      {view === "interview" && session && (
        <InterviewView session={session} onFinished={handleFinished} toast={toast} />
      )}
      {view === "report" && report && (
        <ReportView
          report={report}
          profile={profile}
          mode={session?.mode}
          onRestart={handleRestart}
        />
      )}

      {toastMessage && (
        <div
          role="alert"
          className={`animate-rise fixed bottom-6 left-1/2 z-50 -translate-x-1/2 rounded-xl border px-5 py-3 text-sm font-medium shadow-2xl backdrop-blur ${
            toastMessage.tone === "success"
              ? "border-emerald-500/40 bg-emerald-950/90 text-emerald-200"
              : "border-red-500/40 bg-red-950/90 text-red-200"
          }`}
        >
          {toastMessage.message}
        </div>
      )}
    </div>
  );
}
