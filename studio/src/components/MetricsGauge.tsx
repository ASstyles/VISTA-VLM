import React from 'react';
import { ReliabilityIndicators, InspectionStatus } from '../types';
import { AlertTriangle, CheckCircle2, AlertOctagon, HelpCircle } from 'lucide-react';

interface MetricsGaugeProps {
  score: number;
  status: InspectionStatus;
  reliability: ReliabilityIndicators;
}

export const MetricsGauge: React.FC<MetricsGaugeProps> = ({ score, status, reliability }) => {
  const threshold = reliability.calibrated_threshold;
  const delta = 0.03; // selective prediction uncertainty band width
  const lowerBand = Math.max(0, threshold - delta);
  const upperBand = Math.min(1, threshold + delta);

  // Compute percentage positions for 0 to 1 scale
  const scorePercent = Math.min(100, Math.max(0, score * 100));
  const thresholdPercent = threshold * 100;
  const lowerPercent = lowerBand * 100;
  const upperPercent = upperBand * 100;

  const getStatusColor = () => {
    switch (status) {
      case 'NORMAL':
        return { text: 'text-emerald-400', bg: 'bg-emerald-500', border: 'border-emerald-500/30' };
      case 'ANOMALOUS':
        return { text: 'text-rose-400', bg: 'bg-rose-500', border: 'border-rose-500/30' };
      case 'UNCERTAIN':
        return { text: 'text-amber-400', bg: 'bg-amber-500', border: 'border-amber-500/30' };
      case 'OOD':
        return { text: 'text-purple-400', bg: 'bg-purple-500', border: 'border-purple-500/30' };
      default:
        return { text: 'text-slate-400', bg: 'bg-slate-500', border: 'border-slate-500/30' };
    }
  };

  const colors = getStatusColor();

  return (
    <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Conformal Score & Threshold Gauge
          </h4>
          <p className="text-[11px] text-slate-500 font-mono">Split Conformal Non-Conformity Evaluation (α = {reliability.target_false_alarm_rate})</p>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-slate-400 font-mono">Score:</span>
          <span className={`text-xl font-extrabold font-mono ${colors.text}`}>
            {score.toFixed(4)}
          </span>
        </div>
      </div>

      {/* Linear Track Gauge */}
      <div className="relative pt-6 pb-2">
        {/* Background Track */}
        <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden border border-slate-800 relative">
          {/* Normal Zone */}
          <div 
            className="absolute top-0 bottom-0 left-0 bg-emerald-950/60 border-r border-emerald-700/50"
            style={{ width: `${lowerPercent}%` }}
          />
          {/* Uncertainty Band */}
          <div 
            className="absolute top-0 bottom-0 bg-amber-950/70 border-r border-amber-700/50"
            style={{ left: `${lowerPercent}%`, width: `${upperPercent - lowerPercent}%` }}
          />
          {/* Anomaly Zone */}
          <div 
            className="absolute top-0 bottom-0 bg-rose-950/60"
            style={{ left: `${upperPercent}%`, right: 0 }}
          />
        </div>

        {/* Calibrated Threshold Marker */}
        <div 
          className="absolute top-2 -translate-x-1/2 flex flex-col items-center pointer-events-none"
          style={{ left: `${thresholdPercent}%` }}
        >
          <span className="text-[10px] font-mono text-cyan-400 font-bold">τ = {threshold.toFixed(3)}</span>
          <div className="w-0.5 h-6 bg-cyan-400 shadow-sm shadow-cyan-400"></div>
        </div>

        {/* Current Score Pin */}
        <div 
          className="absolute top-5 -translate-x-1/2 flex flex-col items-center transition-all duration-500"
          style={{ left: `${scorePercent}%` }}
        >
          <div className={`w-4 h-4 rounded-full ${colors.bg} border-2 border-white shadow-lg flex items-center justify-center`}>
            <div className="w-1.5 h-1.5 bg-slate-900 rounded-full"></div>
          </div>
        </div>
      </div>

      {/* Threshold & Reliability Details Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-slate-800/80 text-xs">
        <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800">
          <span className="text-[11px] text-slate-500 block">Threshold (τ_c)</span>
          <span className="font-mono font-bold text-slate-200">{threshold.toFixed(4)}</span>
        </div>

        <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800">
          <span className="text-[11px] text-slate-500 block">Conformal p-value</span>
          <span className="font-mono font-bold text-cyan-300">{reliability.conformal_p_value.toFixed(4)}</span>
        </div>

        <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800">
          <span className="text-[11px] text-slate-500 block">Uncertainty Band</span>
          <span className="font-mono text-slate-300">
            [{lowerBand.toFixed(3)}, {upperBand.toFixed(3)}]
          </span>
        </div>

        <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800">
          <span className="text-[11px] text-slate-500 block">OOD Distance</span>
          <span className="font-mono text-slate-300">
            {reliability.ood_distance.toFixed(3)} / {reliability.ood_threshold.toFixed(3)}
          </span>
        </div>
      </div>

      {/* Selective Prediction Advisory */}
      <div className={`p-3 rounded-xl border flex items-start space-x-3 ${
        reliability.human_review_recommended
          ? 'bg-amber-950/30 border-amber-800/60 text-amber-300'
          : 'bg-emerald-950/30 border-emerald-800/60 text-emerald-300'
      }`}>
        {reliability.human_review_recommended ? (
          <AlertTriangle className="w-4 h-4 mt-0.5 text-amber-400 shrink-0" />
        ) : (
          <CheckCircle2 className="w-4 h-4 mt-0.5 text-emerald-400 shrink-0" />
        )}
        <div className="text-xs">
          <span className="font-bold block">
            {reliability.human_review_recommended ? 'Selective Prediction: Human Review Recommended' : 'Automated Verification: High Confidence'}
          </span>
          <span className="text-slate-400 text-[11px]">
            {reliability.review_reason}
          </span>
        </div>
      </div>
    </div>
  );
};
