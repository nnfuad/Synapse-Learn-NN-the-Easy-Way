import React, { useState } from 'react';
import { Zap, ShieldAlert, Cpu, Activity, Info, Check, RefreshCw } from 'lucide-react';

export default function McCullochPittsSim() {
  const [gateType, setGateType] = useState('AND');
  const [x1, setX1] = useState(1);
  const [x2, setX2] = useState(1);
  const [inhibitory, setInhibitory] = useState(0);
  const [customThreshold, setCustomThreshold] = useState(2);

  // Derive threshold based on selected gate
  const threshold = gateType === 'AND' ? 2 : gateType === 'OR' ? 1 : gateType === 'NOT' ? 0 : customThreshold;

  // Neuron computation logic
  const isVetoed = inhibitory === 1;
  const excitatorySum = (gateType === 'NOT' ? 0 : x1) + (gateType === 'NOT' ? 0 : x2);
  const isFired = !isVetoed && excitatorySum >= threshold;

  const truthTables = {
    AND: [
      { x1: 0, x2: 0, inh: 0, out: 0 },
      { x1: 0, x2: 1, inh: 0, out: 0 },
      { x1: 1, x2: 0, inh: 0, out: 0 },
      { x1: 1, x2: 1, inh: 0, out: 1 },
    ],
    OR: [
      { x1: 0, x2: 0, inh: 0, out: 0 },
      { x1: 0, x2: 1, inh: 0, out: 1 },
      { x1: 1, x2: 0, inh: 0, out: 1 },
      { x1: 1, x2: 1, inh: 0, out: 1 },
    ],
    NOT: [
      { x1: '-', x2: '-', inh: 0, out: 1 },
      { x1: '-', x2: '-', inh: 1, out: 0 },
    ]
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-sm space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-semibold mb-1">
            <Cpu className="w-3.5 h-3.5" />
            1943 Biological Model
          </div>
          <h3 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
            McCulloch-Pitts Logic Gate Simulator
          </h3>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Toggle excitatory inputs and emergency inhibitory vetoes to see how biological thresholds compute formal logic.
          </p>
        </div>

        {/* Gate Selector Pills */}
        <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800">
          {['AND', 'OR', 'NOT', 'CUSTOM'].map((g) => (
            <button
              key={g}
              onClick={() => {
                setGateType(g);
                if (g === 'NOT') setInhibitory(0);
                else setInhibitory(0);
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                gateType === g
                  ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {g}
            </button>
          ))}
        </div>
      </div>

      {/* Main Interactive Diagram */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* Left: Input Switches */}
        <div className="lg:col-span-4 bg-slate-950/70 border border-slate-800/80 rounded-xl p-5 space-y-4">
          <div className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between">
            <span>Dendrite Inputs</span>
            <span className="text-[10px] text-cyan-400">Click to Toggle</span>
          </div>

          {gateType !== 'NOT' && (
            <>
              {/* Excitatory Input 1 */}
              <div
                onClick={() => setX1(x1 === 1 ? 0 : 1)}
                className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition select-none ${
                  x1 === 1
                    ? 'bg-cyan-950/40 border-cyan-500/60 text-cyan-300'
                    : 'bg-slate-900 border-slate-800 text-slate-400'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${x1 === 1 ? 'bg-cyan-400 animate-pulse' : 'bg-slate-700'}`} />
                  <div>
                    <div className="text-xs font-mono font-bold">Input x₁ (Excitatory)</div>
                    <div className="text-[11px] text-slate-500">Weight: +1</div>
                  </div>
                </div>
                <div className="text-sm font-mono font-bold px-2.5 py-1 rounded bg-slate-800/80">
                  {x1}
                </div>
              </div>

              {/* Excitatory Input 2 */}
              <div
                onClick={() => setX2(x2 === 1 ? 0 : 1)}
                className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition select-none ${
                  x2 === 1
                    ? 'bg-cyan-950/40 border-cyan-500/60 text-cyan-300'
                    : 'bg-slate-900 border-slate-800 text-slate-400'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${x2 === 1 ? 'bg-cyan-400 animate-pulse' : 'bg-slate-700'}`} />
                  <div>
                    <div className="text-xs font-mono font-bold">Input x₂ (Excitatory)</div>
                    <div className="text-[11px] text-slate-500">Weight: +1</div>
                  </div>
                </div>
                <div className="text-sm font-mono font-bold px-2.5 py-1 rounded bg-slate-800/80">
                  {x2}
                </div>
              </div>
            </>
          )}

          {/* Inhibitory Veto Input */}
          <div
            onClick={() => setInhibitory(inhibitory === 1 ? 0 : 1)}
            className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition select-none ${
              inhibitory === 1
                ? 'bg-rose-950/40 border-rose-500/60 text-rose-300'
                : 'bg-slate-900 border-slate-800 text-slate-400'
            }`}
          >
            <div className="flex items-center gap-3">
              <ShieldAlert className={`w-4 h-4 ${inhibitory === 1 ? 'text-rose-400 animate-pulse' : 'text-slate-600'}`} />
              <div>
                <div className="text-xs font-mono font-bold">Inhibitory Input (Veto)</div>
                <div className="text-[11px] text-slate-500">Instant brake (cancels firing)</div>
              </div>
            </div>
            <div className="text-sm font-mono font-bold px-2.5 py-1 rounded bg-slate-800/80">
              {inhibitory}
            </div>
          </div>

          {/* Custom Threshold Slider if custom */}
          {gateType === 'CUSTOM' && (
            <div className="pt-2">
              <div className="flex justify-between text-xs font-mono text-slate-400 mb-1">
                <span>Threshold θ</span>
                <span className="text-cyan-400 font-bold">{customThreshold}</span>
              </div>
              <input
                type="range"
                min="0"
                max="3"
                step="1"
                value={customThreshold}
                onChange={(e) => setCustomThreshold(parseInt(e.target.value))}
                className="w-full accent-cyan-500 cursor-pointer"
              />
            </div>
          )}
        </div>

        {/* Center: The Biological Cell & Axon Hillock */}
        <div className="lg:col-span-5 flex flex-col items-center justify-center p-6 bg-slate-950/40 rounded-xl border border-slate-800/60 relative">
          <div className="text-[11px] font-mono text-slate-400 mb-4 uppercase tracking-wider">
            Soma & Axon Hillock
          </div>

          {/* The Neuron Body */}
          <div className={`w-36 h-36 rounded-full border-4 flex flex-col items-center justify-center relative transition-all duration-300 ${
            isVetoed
              ? 'border-rose-500 bg-rose-950/30 text-rose-300 shadow-lg shadow-rose-500/20'
              : isFired
              ? 'border-cyan-400 bg-cyan-950/40 text-cyan-200 shadow-xl shadow-cyan-400/30 scale-105'
              : 'border-slate-700 bg-slate-900/80 text-slate-400'
          }`}>
            <div className="text-xs font-mono text-slate-400">Sum: Σx</div>
            <div className="text-2xl font-extrabold font-mono mt-0.5">
              {excitatorySum}
            </div>
            <div className="text-[11px] font-mono mt-1 text-slate-400">
              θ = {threshold}
            </div>

            {/* Firing Status Badge */}
            <div className="absolute -bottom-3">
              <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold font-mono tracking-wide ${
                isVetoed
                  ? 'bg-rose-500 text-white'
                  : isFired
                  ? 'bg-cyan-400 text-slate-950'
                  : 'bg-slate-800 text-slate-400'
              }`}>
                {isVetoed ? 'VETOED' : isFired ? 'SPIKE (1)' : 'SILENT (0)'}
              </span>
            </div>
          </div>

          {/* Explanation Text */}
          <div className="mt-6 text-center text-xs text-slate-400 max-w-xs leading-relaxed">
            {isVetoed ? (
              <span className="text-rose-400 font-medium">
                Emergency brake active! Any inhibitory signal prevents firing, regardless of charge.
              </span>
            ) : isFired ? (
              <span className="text-cyan-300 font-medium">
                Action potential triggered! Total excitation ({excitatorySum}) &ge; Threshold ({threshold}).
              </span>
            ) : (
              <span>
                Sub-threshold: Total excitation ({excitatorySum}) is less than threshold ({threshold}). Cell stays quiet.
              </span>
            )}
          </div>
        </div>

        {/* Right: Truth Table & Output */}
        <div className="lg:col-span-3 bg-slate-950/70 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between">
          <div>
            <div className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider mb-2">
              {gateType} Truth Table
            </div>

            <div className="w-full text-xs font-mono">
              <div className="grid grid-cols-4 pb-1 border-b border-slate-800 text-slate-500 font-semibold text-center">
                <span>x₁</span>
                <span>x₂</span>
                <span>Inh</span>
                <span>Out</span>
              </div>
              <div className="divide-y divide-slate-800/50">
                {(truthTables[gateType] || truthTables.AND).map((row, i) => {
                  const isActiveRow =
                    (row.x1 === '-' || row.x1 === x1) &&
                    (row.x2 === '-' || row.x2 === x2) &&
                    row.inh === inhibitory;
                  return (
                    <div
                      key={i}
                      className={`grid grid-cols-4 py-1.5 text-center transition ${
                        isActiveRow
                          ? 'bg-cyan-500/20 text-cyan-300 font-bold rounded'
                          : 'text-slate-400'
                      }`}
                    >
                      <span>{row.x1}</span>
                      <span>{row.x2}</span>
                      <span>{row.inh}</span>
                      <span className={row.out === 1 ? 'text-emerald-400 font-bold' : ''}>
                        {row.out}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Big Output Pill */}
          <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">Neuron Output (y):</span>
            <span className={`text-xl font-mono font-black px-3 py-0.5 rounded-lg ${
              isFired
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
                : 'bg-slate-900 text-slate-500 border border-slate-800'
            }`}>
              {isFired ? 1 : 0}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
