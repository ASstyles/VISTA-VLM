import React from 'react';
import { ReliabilityIndicators, InspectionStatus } from '../types';
import { AlertTriangle, CheckCircle2, ShieldAlert, Sparkles, Scale, Info } from 'lucide-react';

interface MetricsGaugeProps {
  score: number;
  status: InspectionStatus;
  reliability: ReliabilityIndicators;
}

export const MetricsGauge: React.FC<MetricsGaugeProps> = ({ score, status, reliability }) => {
  const threshold = reliability.calibrated_threshold;
  const delta = 0.04; // selective prediction uncertainty band width
  const lowerBand = Math.max(0, threshold - delta);
  const upperBand = Math.min(1, threshold + delta);

  // Position percentage calculations
  const scorePercent = Math.min(100, Math.max(0, score * 100));
  const thresholdPercent = threshold * 100;
  const lowerPercent = lowerBand * 100;
  const upperPercent = upperBand * 100;

  const getStatusTheme = () => {
    switch (status) {
      case 'NORMAL':
        return {
          text: 'text-emerald-400',
          gradientText: 'from-emerald-400 to-teal-300',
          badgeBg: 'bg-emerald-950/60 text-emerald-300 border-emerald-700/60',
          pinBg: 'bg-emerald-400 ring-emerald-400/30',
          label: 'Conformal Pass (Nominal)'
        };
      case 'ANOMALOUS':
        return {
          text: 'text-rose-400',
          gradientText: 'from-rose-400 to-red-300',
          badgeBg: 'bg-rose-950/60 text-rose-300 border-rose-700/60',
          pinBg: 'bg-rose-500 ring-rose-500/30',
          label: 'Conformal Defect Detected'
        };
      case 'UNCERTAIN':
        return {
          text: 'text-amber-400',
          gradientText: 'from-amber-400 to-yellow-300',
          badgeBg: 'bg-amber-950/60 text-amber-300 border-amber-700/60',
          pinBg: 'bg-amber-400 ring-amber-400/30',
          label: 'Uncertain: In Critical Band'
        };
      case 'OOD':
        return {
          text: 'text-purple-400',
          gradientText: 'from-purple-400 to-indigo-300',
          badgeBg: 'bg-purple-950/60 text-purple-300 border-purple-700/60',
          pinBg: 'bg-purple-500 ring-purple-500/30',
          label: 'Out-of-Distribution Shift'
        };
      default:
        return {
          text: 'text-slate-400',
          gradientText: 'from-slate-400 to-slate-200',
          badgeBg: 'bg-slate-900 text-slate-300 border-slate-700',
          pinBg: 'bg-slate-400 ring-slate-400/30',
          label: 'Unspecified'
        };
    }
  };

  const theme = getStatusTheme();

  return (
    <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4 shadow-xl">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-cyan-950/60 border border-cyan-800/60 text-cyan-400">
            <Scale className="w-4 h-4" />
          </div>
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center space-x-1.5">
              <span>Split Conformal Non-Conformity Evaluation</span>
              <span className="text-[10px] font-mono text-cyan-400 font-semibold px-2 py-0.2 rounded bg-cyan-950 border border-cyan-800">
                α = {reliability.target_false_alarm_rate}
              </span>
            </h4>
            <p className="text-[11px] text-slate-400">
              Guarantees false-alarm rate &le; 5% on independent normal calibration distributions
            </p>
          </div>
        </div>

        <div className="flex items-baseline space-x-2 bg-slate-950/80 px-3.5 py-1.5 rounded-xl border border-slate-800">
          <span className="text-xs text-slate-400 font-mono">Anomaly Score:</span>
          <span className={`text-2xl font-black font-mono tracking-tight bg-gradient-to-r ${theme.gradientText} bg-clip-text text-transparent`}>
            {score.toFixed(4)}
          </span>
        </div>
      </div>

      {/* Horizontal Multi-Zone Linear Track Gauge */}
      <div className="space-y-1.5 pt-1">
        <div className="flex justify-between text-[11px] font-mono text-slate-400">
          <span className="text-emerald-400 font-medium">0.0 (Nominal)</span>
          <span className="text-cyan-400 font-bold">τ_c = {threshold.toFixed(4)}</span>
          <span className="text-rose-400 font-medium">1.0 (Anomalous)</span>
        </div>

        <div className="relative pt-4 pb-2">
          {/* Main Track */}
          <div className="w-full h-4 bg-slate-900 rounded-full overflow-hidden border border-slate-700/80 relative shadow-inner flex">
            {/* Safe Nominal Band */}
            <div 
              className="h-full bg-gradient-to-r from-emerald-950/80 via-emerald-900/60 to-emerald-800/40 border-r border-emerald-600/40"
              style={{ width: `${lowerPercent}%` }}
              title="Autonomous Normal Region"
            />
            {/* Conformal Uncertainty Interval */}
            <div 
              className="h-full bg-gradient-to-r from-amber-900/60 to-amber-800/60 border-r border-amber-600/50"
              style={{ width: `${Math.max(1, upperPercent - lowerPercent)}%` }}
              title="Selective Prediction: Human Review Uncertainty Band"
            />
            {/* Defect Anomaly Band */}
            <div 
              className="h-full bg-gradient-to-r from-rose-950/70 via-rose-900/70 to-red-900/80 flex-1"
              title="Autonomous Anomaly Region"
            />
          </div>

          {/* Calibrated Threshold Marker */}
          <div 
            className="absolute top-0 -translate-x-1/2 flex flex-col items-center pointer-events-none transition-all"
            style={{ left: `${thresholdPercent}%` }}
          >
            <span className="text-[10px] font-mono text-cyan-300 font-bold bg-cyan-950/90 px-1.5 py-0.2 rounded border border-cyan-700 shadow-md">
              τ_c
            </span>
            <div className="w-0.5 h-6 bg-cyan-400 shadow-[0_0_8px_rgba(56,189,248,0.8)]"></div>
          </div>

          {/* Score Indicator Pin with Pulse */}
          <div 
            className="absolute top-3 -translate-x-1/2 flex flex-col items-center transition-all duration-700 z-10"
            style={{ left: `${scorePercent}%` }}
          >
            <div className={`w-5 h-5 rounded-full ${theme.pinBg} border-2 border-white shadow-xl flex items-center justify-center ring-4 transition-transform hover:scale-110`}>
              <div className="w-2 h-2 bg-slate-950 rounded-full"></div>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono px-1">
          <div className="flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            <span>Nominal (&lt; {lowerBand.toFixed(3)})</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-400"></span>
            <span>Uncertainty Band [{lowerBand.toFixed(3)}, {upperBand.toFixed(3)}]</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-rose-500"></span>
            <span>Defect (&gt; {upperBand.toFixed(3)})</span>
          </div>
        </div>
      </div>

      {/* Grid of Analytical Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-2 text-xs">
        <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/80 shadow-sm">
          <span className="text-[10px] text-slate-500 block uppercase font-bold tracking-wider">Conformal τ_c</span>
          <span className="font-mono font-extrabold text-cyan-300 text-sm">{threshold.toFixed(4)}</span>
          <span className="text-[10px] text-slate-500 block mt-0.5">Quantile (1 - α)</span>
        </div>

        <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/80 shadow-sm">
          <span className="text-[10px] text-slate-500 block uppercase font-bold tracking-wider">Conformal p-value</span>
          <span className="font-mono font-extrabold text-white text-sm">{reliability.conformal_p_value.toFixed(4)}</span>
          <span className="text-[10px] text-slate-500 block mt-0.5">{reliability.conformal_p_value < 0.05 ? 'Significant (< 0.05)' : 'Nominal (> 0.05)'}</span>
        </div>

        <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/80 shadow-sm">
          <span className="text-[10px] text-slate-500 block uppercase font-bold tracking-wider">OOD Prototype Dist</span>
          <span className="font-mono font-extrabold text-slate-200 text-sm">
            {reliability.ood_distance.toFixed(3)}
          </span>
          <span className="text-[10px] text-slate-500 block mt-0.5">Threshold: {reliability.ood_threshold.toFixed(3)}</span>
        </div>

        <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/80 shadow-sm">
          <span className="text-[10px] text-slate-500 block uppercase font-bold tracking-wider">Review Gating</span>
          <span className={`font-mono font-bold text-xs ${reliability.human_review_recommended ? 'text-amber-400' : 'text-emerald-400'}`}>
            {reliability.human_review_recommended ? 'HUMAN REVIEW' : 'AUTONOMOUS'}
          </span>
          <span className="text-[10px] text-slate-500 block mt-0.5">{reliability.in_uncertainty_band ? 'In Uncertainty Band' : 'Clear Decision'}</span>
        </div>
      </div>

      {/* Advisory Callout Card */}
      <div className={`p-3.5 rounded-xl border flex items-start space-x-3 transition-all ${
        reliability.human_review_recommended
          ? 'bg-amber-950/30 border-amber-700/60 text-amber-200 shadow-md shadow-amber-950/20'
          : 'bg-emerald-950/30 border-emerald-700/60 text-emerald-200 shadow-md shadow-emerald-950/20'
      }`}>
        {reliability.human_review_recommended ? (
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5 animate-pulse" />
        ) : (
          <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
        )}
        <div className="text-xs space-y-0.5">
          <span className="font-bold block tracking-tight">
            {reliability.human_review_recommended
              ? 'Selective Prediction Advisory: Escalation to Quality Inspector Recommended'
              : 'Automated Certification: High Confidence Assessment'}
          </span>
          <p className="text-slate-300 text-[11px] leading-relaxed">
            {reliability.review_reason}
          </p>
        </div>
      </div>
    </div>
  );
};
