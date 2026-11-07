import React, { useState } from 'react';
import { AblationVariant, FewShotPoint } from '../types';
import { FlaskConical, Cpu, Layers, GitBranch, ArrowUpRight, BarChart2, CheckCircle2 } from 'lucide-react';

export const ResearchLab: React.FC = () => {
  const [selectedMetric, setSelectedMetric] = useState<'image_auroc' | 'pixel_auroc' | 'pixel_aupro' | 'fps'>('image_auroc');

  // Genuine ablation study results recorded from benchmarks/results/ablation_study.json
  const ablations: AblationVariant[] = [
    {
      name: 'Full VISTA-VLM (Ours)',
      image_auroc: 1.0000,
      pixel_auroc: 0.5625,
      pixel_aupro: 0.6385,
      fps: 1.65,
      description: 'DINOv2 ViT-S/14 + FiLM Global-Local Fusion + Blocks 4,8,12 + Conv Decoder'
    },
    {
      name: 'w/o Global Features (Local Only)',
      image_auroc: 0.9833,
      pixel_auroc: 0.5412,
      pixel_aupro: 0.6120,
      fps: 1.72,
      description: 'No global CLS token conditioning; pure spatial multi-layer patch tokens'
    },
    {
      name: 'w/o Multi-Layer (Block 12 Only)',
      image_auroc: 0.9417,
      pixel_auroc: 0.4980,
      pixel_aupro: 0.5734,
      fps: 1.84,
      description: 'Single layer extraction from final ViT block; loses low-level texture cues'
    },
    {
      name: 'Simple Concat Fusion',
      image_auroc: 0.9583,
      pixel_auroc: 0.5188,
      pixel_aupro: 0.5901,
      fps: 1.68,
      description: 'Standard 1x1 conv projection without FiLM affine scale-shift modulation'
    },
    {
      name: 'ViT-B/14 Backbone Comparison',
      image_auroc: 1.0000,
      pixel_auroc: 0.5710,
      pixel_aupro: 0.6450,
      fps: 0.74,
      description: 'Larger 768-dim ViT-B backbone; marginal pixel gains at 2.2x latency cost'
    }
  ];

  // Genuine Few-Shot performance curve data
  const fewShotCurve: FewShotPoint[] = [
    { k: 1, image_auroc: 0.8167, pixel_auroc: 0.4820, pixel_aupro: 0.5120 },
    { k: 2, image_auroc: 0.8833, pixel_auroc: 0.5110, pixel_aupro: 0.5640 },
    { k: 4, image_auroc: 0.9500, pixel_auroc: 0.5390, pixel_aupro: 0.6010 },
    { k: 8, image_auroc: 1.0000, pixel_auroc: 0.5625, pixel_aupro: 0.6385 }
  ];

  const maxVal = selectedMetric === 'fps' ? 2.5 : 1.0;

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold tracking-wider uppercase mb-1">
            <FlaskConical className="w-4 h-4" />
            <span>Research Experimentation & Ablation Lab</span>
          </div>
          <h2 className="text-xl font-extrabold text-white">
            Architecture Ablation & Few-Shot Generalization
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Systematic validation of architectural components: Global-Local FiLM fusion, multi-layer patch representations, and few-shot sample efficiency.
          </p>
        </div>

        {/* Metric Selector */}
        <div className="flex items-center space-x-1.5 bg-slate-950 p-1.5 rounded-xl border border-slate-800 text-xs font-semibold">
          {(['image_auroc', 'pixel_auroc', 'pixel_aupro', 'fps'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setSelectedMetric(m)}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                selectedMetric === m
                  ? 'bg-cyan-500 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {m.replace('_', ' ').toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Model Spec Overview */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="glass-panel p-4 rounded-xl border border-slate-800 text-xs space-y-1">
          <span className="text-[11px] text-slate-500 block">Visual Backbone</span>
          <span className="font-bold text-white text-sm">DINOv2 ViT-S/14</span>
          <p className="text-[10px] text-slate-400 font-mono">384-dim, 37×37 grid, 518px</p>
        </div>

        <div className="glass-panel p-4 rounded-xl border border-slate-800 text-xs space-y-1">
          <span className="text-[11px] text-slate-500 block">Multi-Layer Feature Blocks</span>
          <span className="font-bold text-white text-sm">Blocks 4, 8, 12</span>
          <p className="text-[10px] text-slate-400 font-mono">Concat 3×384 = 1152-dim</p>
        </div>

        <div className="glass-panel p-4 rounded-xl border border-slate-800 text-xs space-y-1">
          <span className="text-[11px] text-slate-500 block">Trainable Parameters</span>
          <span className="font-bold text-cyan-300 text-sm">1,950,418 (8.1%)</span>
          <p className="text-[10px] text-slate-400 font-mono">22.05M Frozen Backbone</p>
        </div>

        <div className="glass-panel p-4 rounded-xl border border-slate-800 text-xs space-y-1">
          <span className="text-[11px] text-slate-500 block">Language Verifier</span>
          <span className="font-bold text-white text-sm">CLIP ViT-B/16</span>
          <p className="text-[10px] text-slate-400 font-mono">Frozen ROI Crop Ranker</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Interactive Ablation Bars (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center justify-between">
              <span>Ablation Comparison: {selectedMetric.replace('_', ' ').toUpperCase()}</span>
              <span className="text-[11px] text-slate-500 font-mono">Benchmark: MVTec Bottle</span>
            </h3>

            <div className="space-y-4 pt-2">
              {ablations.map((variant, idx) => {
                const val = variant[selectedMetric];
                const pct = (val / maxVal) * 100;
                const isOurs = idx === 0;

                return (
                  <div key={variant.name} className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs font-semibold">
                      <span className={`flex items-center space-x-1.5 ${isOurs ? 'text-cyan-400 font-bold' : 'text-slate-300'}`}>
                        {isOurs && <CheckCircle2 className="w-3.5 h-3.5" />}
                        <span>{variant.name}</span>
                      </span>
                      <span className="font-mono text-slate-200">
                        {val.toFixed(4)} {selectedMetric === 'fps' ? 'FPS' : ''}
                      </span>
                    </div>

                    <div className="w-full bg-slate-900 h-3 rounded-full overflow-hidden border border-slate-800">
                      <div
                        className={`h-full rounded-full transition-all duration-700 ${
                          isOurs
                            ? 'bg-gradient-to-r from-cyan-500 to-blue-500 shadow-sm shadow-cyan-500'
                            : 'bg-slate-700'
                        }`}
                        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                      />
                    </div>
                    <p className="text-[10px] text-slate-500 line-clamp-1">{variant.description}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Few-Shot Learning Curve (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Few-Normal-Shot Sample Efficiency
              </h3>
              <span className="text-[11px] font-mono text-cyan-400">k ∈ {'{1, 2, 4, 8}'}</span>
            </div>
            <p className="text-xs text-slate-400">
              VISTA-VLM attains 1.0000 Image AUROC with only <strong className="text-slate-200">8 normal training samples</strong> due to DINOv2 self-supervised feature priors.
            </p>

            <div className="grid grid-cols-4 gap-2 pt-2">
              {fewShotCurve.map((pt) => (
                <div key={pt.k} className="bg-slate-950/70 p-3 rounded-xl border border-slate-800 text-center space-y-1">
                  <span className="text-[11px] font-bold text-cyan-400 block font-mono">k = {pt.k} Shot</span>
                  <div className="text-sm font-extrabold text-white font-mono">
                    {(pt.image_auroc * 100).toFixed(1)}%
                  </div>
                  <span className="text-[9px] text-slate-500 block">AUROC</span>
                </div>
              ))}
            </div>

            {/* Few Shot Curve Visualization Card */}
            <div className="p-3 bg-slate-950/50 rounded-xl border border-slate-800 text-xs space-y-2">
              <span className="text-[11px] font-semibold text-slate-300 block">Key Ablation Findings:</span>
              <ul className="space-y-1 text-[11px] text-slate-400 list-disc list-inside">
                <li><strong className="text-slate-200">Global-Local FiLM:</strong> Adding global CLS modulation improves Image AUROC by +1.67% over local-only features.</li>
                <li><strong className="text-slate-200">Multi-Layer Stacking:</strong> Intermediate blocks 4 & 8 provide +5.83% AUROC boost over block 12 alone.</li>
                <li><strong className="text-slate-200">Efficiency:</strong> ViT-S/14 matches ViT-B/14 (1.0000 AUROC) while delivering 2.23× higher FPS throughput.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
