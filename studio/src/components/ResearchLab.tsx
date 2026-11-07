import React, { useState } from 'react';
import { AblationVariant, FewShotPoint } from '../types';
import { FlaskConical, Cpu, Layers, GitBranch, ArrowUpRight, BarChart2, CheckCircle2, Zap } from 'lucide-react';

export const ResearchLab: React.FC = () => {
  const [selectedMetric, setSelectedMetric] = useState<'image_auroc' | 'pixel_auroc' | 'pixel_aupro' | 'fps'>('image_auroc');

  const ablations: AblationVariant[] = [
    {
      name: 'Full VISTA-VLM (Ours)',
      image_auroc: 1.0000,
      pixel_auroc: 0.5625,
      pixel_aupro: 0.6385,
      fps: 1.65,
      description: 'DINOv2 ViT-S/14 + FiLM Global-Local Fusion + Blocks 4,8,12 + Progressive Conv Decoder'
    },
    {
      name: 'w/o Global Features (Local Only)',
      image_auroc: 0.9833,
      pixel_auroc: 0.5412,
      pixel_aupro: 0.6120,
      fps: 1.72,
      description: 'Omits global CLS token conditioning; pure spatial multi-layer patch tokens (-1.67% AUROC)'
    },
    {
      name: 'w/o Multi-Layer (Block 12 Only)',
      image_auroc: 0.9417,
      pixel_auroc: 0.4980,
      pixel_aupro: 0.5734,
      fps: 1.84,
      description: 'Single layer extraction from final ViT block; loses low-level texture cues (-5.83% AUROC)'
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
      description: 'Larger 768-dim ViT-B backbone; marginal pixel gains (+0.85%) at 2.23x latency cost'
    }
  ];

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
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-xl">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold tracking-wider uppercase mb-1">
            <FlaskConical className="w-4 h-4 text-cyan-400" />
            <span>Research Experimentation & Ablation Lab</span>
          </div>
          <h2 className="text-xl sm:text-2xl font-black text-white tracking-tight">
            Architecture Ablation & Few-Shot Generalization
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Systematic experimental validation of FiLM modulation, multi-layer intermediate representations, and few-shot efficiency.
          </p>
        </div>

        {/* Metric Selector Pills */}
        <div className="flex items-center space-x-1.5 bg-slate-950 p-1.5 rounded-xl border border-slate-800 text-xs font-semibold shadow-inner">
          {(['image_auroc', 'pixel_auroc', 'pixel_aupro', 'fps'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setSelectedMetric(m)}
              className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
                selectedMetric === m
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/25'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              {m.replace('_', ' ').toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Model Spec Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="glass-panel p-4 rounded-xl border border-slate-800 text-xs space-y-1 shadow-md">
          <span className="text-[10px] text-slate-500 block uppercase font-bold tracking-wider">Visual Backbone</span>
          <span className="font-extrabold text-white text-sm">DINOv2 ViT-S/14</span>
          <p className="text-[10px] text-cyan-400 font-mono">384-dim • 37×37 Grid • 518px</p>
        </div>

        <div className="glass-panel p-4 rounded-xl border border-slate-800 text-xs space-y-1 shadow-md">
          <span className="text-[10px] text-slate-500 block uppercase font-bold tracking-wider">Feature Extraction</span>
          <span className="font-extrabold text-white text-sm">Blocks 4, 8, 12</span>
          <p className="text-[10px] text-slate-400 font-mono">Concat 3×384 = 1152-dim</p>
        </div>

        <div className="glass-panel p-4 rounded-xl border border-slate-800 text-xs space-y-1 shadow-md">
          <span className="text-[10px] text-slate-500 block uppercase font-bold tracking-wider">Trainable Parameters</span>
          <span className="font-extrabold text-cyan-300 text-sm">1,950,418 (8.1%)</span>
          <p className="text-[10px] text-slate-400 font-mono">22.05M Frozen Backbone</p>
        </div>

        <div className="glass-panel p-4 rounded-xl border border-slate-800 text-xs space-y-1 shadow-md">
          <span className="text-[10px] text-slate-500 block uppercase font-bold tracking-wider">Multimodal Verifier</span>
          <span className="font-extrabold text-white text-sm">CLIP ViT-B/16</span>
          <p className="text-[10px] text-slate-400 font-mono">Frozen ROI Crop Ranker</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Interactive Ablation Bars (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center space-x-2">
                <BarChart2 className="w-4 h-4 text-cyan-400" />
                <span>Ablation Comparison: {selectedMetric.replace('_', ' ').toUpperCase()}</span>
              </h3>
              <span className="text-[11px] text-slate-500 font-mono">MVTec AD Bottle</span>
            </div>

            <div className="space-y-4 pt-1">
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
                      <span className="font-mono text-slate-100 font-bold">
                        {val.toFixed(4)} {selectedMetric === 'fps' ? 'FPS' : ''}
                      </span>
                    </div>

                    <div className="w-full bg-slate-900 h-3 rounded-full overflow-hidden border border-slate-800 shadow-inner">
                      <div
                        className={`h-full rounded-full transition-all duration-700 ${
                          isOurs
                            ? 'bg-gradient-to-r from-cyan-500 to-blue-500 shadow-md shadow-cyan-500/40'
                            : 'bg-slate-700'
                        }`}
                        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                      />
                    </div>
                    <p className="text-[11px] text-slate-400">{variant.description}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Few-Shot Curve (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Few-Normal-Shot Sample Efficiency
              </h3>
              <span className="text-[10px] font-mono text-cyan-400 font-semibold px-2 py-0.5 rounded bg-cyan-950 border border-cyan-800">
                k &isin; {'{1, 2, 4, 8}'}
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              VISTA-VLM achieves <strong className="text-cyan-300 font-mono font-bold">1.0000 Image AUROC</strong> with only <strong className="text-white">8 normal training samples</strong>, leveraging DINOv2 self-supervised visual priors to eliminate large domain data requirements.
            </p>

            <div className="grid grid-cols-4 gap-2 pt-1">
              {fewShotCurve.map((pt) => (
                <div key={pt.k} className="bg-slate-950/80 p-3 rounded-xl border border-slate-800 text-center space-y-1 shadow-sm">
                  <span className="text-[10px] font-bold text-cyan-400 block font-mono">k = {pt.k} Shot</span>
                  <div className="text-sm font-black text-white font-mono">
                    {(pt.image_auroc * 100).toFixed(1)}%
                  </div>
                  <span className="text-[9px] text-slate-500 block uppercase font-mono">AUROC</span>
                </div>
              ))}
            </div>

            {/* Architectural Insights Card */}
            <div className="p-3.5 bg-slate-950/70 rounded-xl border border-slate-800 text-xs space-y-2 shadow-sm">
              <span className="text-[11px] font-bold text-slate-200 block uppercase tracking-wide">
                Key Research Takeaways:
              </span>
              <ul className="space-y-1.5 text-[11px] text-slate-300 list-disc list-inside">
                <li><strong className="text-cyan-300">FiLM Modulation:</strong> Global context conditioning yields +1.67% Image AUROC improvement over unconditioned local patch models.</li>
                <li><strong className="text-cyan-300">Multi-Layer Concatenation:</strong> Intermediate blocks 4 & 8 add essential low-level texture cues (+5.83% AUROC vs block 12 alone).</li>
                <li><strong className="text-cyan-300">Edge ViT-S Efficiency:</strong> ViT-S/14 matches ViT-B/14 (1.0000 AUROC) while delivering 2.23× higher FPS throughput.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
