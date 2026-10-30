import React, { useState, useEffect, useRef } from 'react';
import {
  HelpCircle,
  CheckCircle2,
  XCircle,
  Award,
  Sparkles,
  RefreshCw,
  Clock,
  Lightbulb,
  Zap,
  Flame,
  Trophy,
  SlidersHorizontal,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import confetti from 'canvas-confetti';
import { MODULE_QUIZ_DATA, QUIZ_DIFFICULTIES } from '../data/quizData.js';

export default function StudentQuiz({ currentModule = '00' }) {
  // Retrieve all questions for current module or fallback to '00'
  const allModuleQuestions = MODULE_QUIZ_DATA[currentModule] || MODULE_QUIZ_DATA['00'] || [];

  // Quiz state
  const [selectedDifficulty, setSelectedDifficulty] = useState('all');
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [expandedHints, setExpandedHints] = useState({});
  const [modeType, setModeType] = useState('exam'); // 'exam' or 'instant'
  
  // Speedrun timer state
  const [timerEnabled, setTimerEnabled] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const timerRef = useRef(null);

  // Filter questions based on chosen difficulty
  const questions = React.useMemo(() => {
    if (selectedDifficulty === 'all') {
      return allModuleQuestions;
    }
    return allModuleQuestions.filter(q => q.difficulty === selectedDifficulty);
  }, [allModuleQuestions, selectedDifficulty]);

  // Difficulty counts for badge pills
  const countsByDifficulty = React.useMemo(() => {
    const counts = { all: allModuleQuestions.length, basic: 0, hard: 0, pro: 0 };
    allModuleQuestions.forEach(q => {
      if (counts[q.difficulty] !== undefined) {
        counts[q.difficulty]++;
      }
    });
    return counts;
  }, [allModuleQuestions]);

  // Reset quiz state whenever module or difficulty changes
  const resetQuiz = () => {
    setSelectedAnswers({});
    setSubmitted(false);
    setExpandedHints({});
    setElapsedSeconds(0);
  };

  useEffect(() => {
    resetQuiz();
  }, [currentModule, selectedDifficulty]);

  // Stopwatch timer logic
  useEffect(() => {
    if (timerEnabled && !submitted) {
      timerRef.current = setInterval(() => {
        setElapsedSeconds(prev => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [timerEnabled, submitted]);

  const handleSelectOption = (qId, optionIdx) => {
    // In submitted exam mode, locks answers
    if (submitted && modeType === 'exam') return;
    
    // In instant mode, lock after first selection for that question
    if (modeType === 'instant' && selectedAnswers[qId] !== undefined) return;

    const nextAnswers = { ...selectedAnswers, [qId]: optionIdx };
    setSelectedAnswers(nextAnswers);

    // If instant mode, check if all are answered
    if (modeType === 'instant') {
      const allDone = questions.every(q => nextAnswers[q.id] !== undefined);
      if (allDone) {
        setSubmitted(true);
        triggerCompletionCelebration(calculateScore(nextAnswers));
      }
    }
  };

  const toggleHint = (qId) => {
    setExpandedHints(prev => ({
      ...prev,
      [qId]: !prev[qId]
    }));
  };

  const calculateScore = (answers = selectedAnswers) => {
    let correct = 0;
    questions.forEach(q => {
      if (answers[q.id] !== undefined && q.options[answers[q.id]]?.correct) {
        correct++;
      }
    });
    return correct;
  };

  const triggerCompletionCelebration = (scoreVal) => {
    if (scoreVal === questions.length && questions.length > 0) {
      confetti({
        particleCount: 100,
        spread: 80,
        origin: { y: 0.6 }
      });
    }
  };

  const handleSubmit = () => {
    setSubmitted(true);
    const scoreVal = calculateScore();
    triggerCompletionCelebration(scoreVal);
  };

  const score = calculateScore();
  const allAnswered = questions.length > 0 && questions.every(q => selectedAnswers[q.id] !== undefined);
  const answeredCount = questions.filter(q => selectedAnswers[q.id] !== undefined).length;

  const formatTime = (totalSec) => {
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  // Playful achievement title generator
  const getPlayfulRank = () => {
    const pct = questions.length > 0 ? (score / questions.length) : 0;
    if (pct === 1) {
      if (selectedDifficulty === 'pro') return { title: '👑 Galaxy Brain SOTA Supreme', desc: 'You speak fluent gradient tensors! NeurIPS review committee has nothing on you.' };
      if (selectedDifficulty === 'hard') return { title: '🔥 Gradient Bender Elite', desc: 'No anisotropic ravine or vanished gradient can stop your backpropagation!' };
      if (selectedDifficulty === 'basic') return { title: '🌱 Neuron Whisperer', desc: 'Flawless foundation! You are ready to crush the Hard and Pro gauntlets.' };
      return { title: '⚡ Master of the Neural Multiverse', desc: 'Clean sweep across all difficulties! Pure 100% computational graph mastery.' };
    }
    if (pct >= 0.75) {
      return { title: '🚀 Tensor Titan', desc: 'Rock solid intuition! Just a tiny hyperparameter tuning away from perfection.' };
    }
    if (pct >= 0.5) {
      return { title: '🛠️ Local Minima Survivor', desc: 'Good fight! A little more momentum and you will escape that saddle point.' };
    }
    return { title: '☕ Gradient Divergence (Grab Coffee)', desc: 'Learning rate was a bit spicy! Review the pedagogical insights and spin again.' };
  };

  const rank = getPlayfulRank();

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center space-y-3 relative overflow-hidden">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-48 h-48 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 -ml-16 -mb-16 w-48 h-48 bg-fuchsia-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-mono">
          <HelpCircle className="w-3.5 h-3.5" />
          <span>Interactive Knowledge Check</span>
        </div>
        
        <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
          Module {currentModule} Mastery Assessment
        </h2>
        
        <p className="text-xs sm:text-sm text-slate-400 max-w-2xl mx-auto leading-relaxed">
          Select your challenge tier below. Whether you want gentle visual intuitions or bleeding-edge SOTA research internals, pick your mode and test your neural wits.
        </p>

        {/* Difficulty Selection Pills */}
        <div className="pt-2">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 max-w-2xl mx-auto">
            {QUIZ_DIFFICULTIES.map(diff => {
              const isActive = selectedDifficulty === diff.id;
              const count = countsByDifficulty[diff.id] || 0;

              let activeBorder = 'border-slate-800 bg-slate-950/70 text-slate-400 hover:border-slate-700 hover:text-slate-200';
              if (isActive) {
                if (diff.id === 'basic') activeBorder = 'border-emerald-500 bg-emerald-950/40 text-emerald-300 ring-2 ring-emerald-500/40';
                else if (diff.id === 'hard') activeBorder = 'border-amber-500 bg-amber-950/40 text-amber-300 ring-2 ring-amber-500/40';
                else if (diff.id === 'pro') activeBorder = 'border-fuchsia-500 bg-fuchsia-950/40 text-fuchsia-300 ring-2 ring-fuchsia-500/40';
                else activeBorder = 'border-cyan-500 bg-cyan-950/40 text-cyan-300 ring-2 ring-cyan-500/40';
              }

              return (
                <button
                  key={diff.id}
                  onClick={() => setSelectedDifficulty(diff.id)}
                  className={`p-3 rounded-xl border text-left transition-all relative flex flex-col justify-between group ${activeBorder}`}
                >
                  <div className="flex items-center justify-between w-full mb-1">
                    <span className="text-base">{diff.icon}</span>
                    <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300">
                      {count} Qs
                    </span>
                  </div>
                  <div className="font-bold text-xs sm:text-sm text-white group-hover:text-cyan-300 transition">
                    {diff.label}
                  </div>
                  <div className="text-[10px] text-slate-500 leading-tight line-clamp-1 mt-0.5">
                    {diff.id === 'basic' ? 'Gentle & Intuitive' : diff.id === 'hard' ? 'Math & Mechanics' : diff.id === 'pro' ? 'Galaxy Brain SOTA' : 'Full Challenge'}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Control Bar: Mode Toggle & Timer */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2 text-slate-300 font-medium">
          <span className="text-slate-500">Active Tier:</span>
          <span className="capitalize font-bold text-white flex items-center gap-1.5">
            {selectedDifficulty === 'basic' && <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />}
            {selectedDifficulty === 'hard' && <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />}
            {selectedDifficulty === 'pro' && <span className="w-2 h-2 rounded-full bg-fuchsia-400 animate-pulse" />}
            {selectedDifficulty === 'all' && <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />}
            {selectedDifficulty} Mode ({questions.length} questions)
          </span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400 font-mono">
            {answeredCount}/{questions.length} Answered
          </span>
        </div>

        <div className="flex items-center gap-4">
          {/* Practice vs Exam mode toggle */}
          <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
            <button
              onClick={() => { setModeType('exam'); resetQuiz(); }}
              className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition ${
                modeType === 'exam'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              📝 Exam
            </button>
            <button
              onClick={() => { setModeType('instant'); resetQuiz(); }}
              className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition ${
                modeType === 'instant'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              ⚡ Instant
            </button>
          </div>

          {/* Speedrun timer toggle */}
          <button
            onClick={() => setTimerEnabled(!timerEnabled)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border transition ${
              timerEnabled
                ? 'bg-amber-500/10 border-amber-500/40 text-amber-300 font-mono'
                : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-300'
            }`}
          >
            <Clock className="w-3.5 h-3.5 text-amber-400" />
            <span>{timerEnabled ? formatTime(elapsedSeconds) : 'Timer'}</span>
          </button>
        </div>
      </div>

      {/* Questions List */}
      <div className="space-y-4">
        {questions.length === 0 ? (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center text-slate-400">
            No questions found for this mode in this module yet.
          </div>
        ) : (
          questions.map((q, idx) => {
            const selectedIdx = selectedAnswers[q.id];
            const isAnswered = selectedIdx !== undefined;
            const isInstant = modeType === 'instant';
            const showFeedback = submitted || (isInstant && isAnswered);
            const isHintOpen = Boolean(expandedHints[q.id]);

            // Question difficulty badge
            let diffBadge = 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30';
            if (q.difficulty === 'basic') diffBadge = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
            if (q.difficulty === 'hard') diffBadge = 'bg-amber-500/10 text-amber-400 border-amber-500/30';
            if (q.difficulty === 'pro') diffBadge = 'bg-fuchsia-500/10 text-fuchsia-400 border-fuchsia-500/30';

            return (
              <div
                key={q.id}
                className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 space-y-4 transition"
              >
                {/* Question Header & Difficulty Tag */}
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3">
                    <span className="w-6 h-6 rounded-lg bg-cyan-500/20 text-cyan-300 font-mono font-bold text-xs flex items-center justify-center flex-shrink-0 mt-0.5">
                      {idx + 1}
                    </span>
                    <div className="font-bold text-white text-sm sm:text-base leading-snug">
                      {q.question}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-wider font-bold border ${diffBadge}`}>
                      {q.difficulty}
                    </span>
                  </div>
                </div>

                {/* Optional Hint Toggle */}
                {q.hint && (
                  <div className="pl-9">
                    <button
                      onClick={() => toggleHint(q.id)}
                      className="inline-flex items-center gap-1.5 text-xs text-amber-400/90 hover:text-amber-300 font-medium transition cursor-pointer"
                    >
                      <Lightbulb className="w-3.5 h-3.5" />
                      <span>{isHintOpen ? 'Hide Hint' : 'Need a hint?'}</span>
                      {isHintOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                    </button>
                    {isHintOpen && (
                      <div className="mt-2 p-3 rounded-xl bg-amber-950/20 border border-amber-500/20 text-xs text-amber-200/90 animate-fadeIn">
                        <strong className="text-amber-400 font-semibold mr-1">Hint:</strong>
                        {q.hint}
                      </div>
                    )}
                  </div>
                )}

                {/* Options List */}
                <div className="space-y-2 pl-9">
                  {q.options.map((opt, oIdx) => {
                    const isSelected = selectedIdx === oIdx;
                    let optStyle = "bg-slate-950/60 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-850";

                    if (showFeedback) {
                      if (opt.correct) {
                        optStyle = "bg-emerald-950/40 border-emerald-500/60 text-emerald-200 font-medium";
                      } else if (isSelected && !opt.correct) {
                        optStyle = "bg-rose-950/40 border-rose-500/60 text-rose-200";
                      } else {
                        optStyle = "bg-slate-950/30 border-slate-900 text-slate-600";
                      }
                    } else if (isSelected) {
                      optStyle = "bg-cyan-950/40 border-cyan-500/80 text-cyan-200 ring-1 ring-cyan-500/50";
                    }

                    return (
                      <div
                        key={oIdx}
                        onClick={() => handleSelectOption(q.id, oIdx)}
                        className={`p-3.5 rounded-xl border text-xs sm:text-sm font-medium transition cursor-pointer flex items-center justify-between ${optStyle}`}
                      >
                        <span className="leading-relaxed">{opt.text}</span>
                        {showFeedback && opt.correct && (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 ml-2" />
                        )}
                        {showFeedback && isSelected && !opt.correct && (
                          <XCircle className="w-4 h-4 text-rose-400 flex-shrink-0 ml-2" />
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Feedback Explanation */}
                {showFeedback && (
                  <div className="mt-4 ml-9 p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 leading-relaxed animate-fadeIn">
                    <span className="font-bold text-cyan-400 block mb-1 flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5" />
                      <span>Pedagogical Insight:</span>
                    </span>
                    {q.explanation}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Action Footer & Results */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        {!submitted ? (
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="text-xs text-slate-400">
              {allAnswered ? (
                <span className="text-emerald-400 font-semibold flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4" /> All {questions.length} questions answered! Ready to grade.
                </span>
              ) : (
                <span>
                  Answer all {questions.length} questions to submit ({answeredCount}/{questions.length} complete).
                </span>
              )}
            </div>

            <button
              onClick={handleSubmit}
              disabled={!allAnswered}
              className={`w-full sm:w-auto px-6 py-2.5 rounded-xl font-bold text-xs sm:text-sm transition flex items-center justify-center gap-2 ${
                allAnswered
                  ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/20 hover:scale-102 cursor-pointer"
                  : "bg-slate-800 text-slate-500 cursor-not-allowed"
              }`}
            >
              <Sparkles className="w-4 h-4" />
              <span>Submit Answers & Grade Quiz</span>
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-xl bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center flex-shrink-0">
                  <Trophy className="w-6 h-6 text-cyan-400" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-black text-white">
                      Score: {score} / {questions.length}
                    </span>
                    <span className="px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono text-xs font-bold">
                      {Math.round((score / questions.length) * 100)}%
                    </span>
                    {timerEnabled && (
                      <span className="px-2 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 font-mono text-xs flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatTime(elapsedSeconds)}
                      </span>
                    )}
                  </div>
                  <div className="text-xs font-semibold text-cyan-300 mt-0.5">
                    {rank.title}
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">
                    {rank.desc}
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2 w-full sm:w-auto">
                <button
                  onClick={resetQuiz}
                  className="flex-1 sm:flex-initial px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 transition flex items-center justify-center gap-1.5"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Retry Quiz</span>
                </button>

                {selectedDifficulty === 'basic' && (
                  <button
                    onClick={() => setSelectedDifficulty('hard')}
                    className="flex-1 sm:flex-initial px-4 py-2.5 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-xs font-bold text-amber-300 transition flex items-center justify-center gap-1.5"
                  >
                    <span>Try Hard Mode</span>
                    <Flame className="w-3.5 h-3.5" />
                  </button>
                )}

                {selectedDifficulty === 'hard' && (
                  <button
                    onClick={() => setSelectedDifficulty('pro')}
                    className="flex-1 sm:flex-initial px-4 py-2.5 rounded-xl bg-fuchsia-500/20 hover:bg-fuchsia-500/30 border border-fuchsia-500/40 text-xs font-bold text-fuchsia-300 transition flex items-center justify-center gap-1.5"
                  >
                    <span>Try Pro Mode</span>
                    <Zap className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>

            {/* Quick Difficulty Switcher Footer */}
            <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-xs text-slate-400">
              <span>Switch difficulty tier:</span>
              <div className="flex items-center gap-1.5">
                {['all', 'basic', 'hard', 'pro'].map(tier => (
                  <button
                    key={tier}
                    onClick={() => setSelectedDifficulty(tier)}
                    className={`px-2.5 py-1 rounded-lg text-[11px] font-mono capitalize transition ${
                      selectedDifficulty === tier
                        ? 'bg-cyan-500 text-black font-bold'
                        : 'bg-slate-950 text-slate-400 hover:text-white border border-slate-800'
                    }`}
                  >
                    {tier}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
