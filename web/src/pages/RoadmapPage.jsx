import React from 'react';
import HistoryTimeline from '../components/HistoryTimeline.jsx';
import { ArrowLeft, BookOpen, Compass } from 'lucide-react';

export default function RoadmapPage({ onNavigate }) {
  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Banner */}
      <div className="flex items-center justify-between pb-2">
        <button
          onClick={() => onNavigate('dashboard')}
          className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs font-semibold text-slate-300 hover:text-white transition"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Dashboard</span>
        </button>

        <span className="text-xs font-mono text-cyan-400">80 Years of Deep Learning History</span>
      </div>

      <HistoryTimeline />
    </div>
  );
}
