import React, { useState, useRef } from 'react';
import { InspectionResponse, InspectionStatus } from '../types';
import { MetricsGauge } from './MetricsGauge';
import { 
  Upload, 
  Image as ImageIcon, 
  CheckCircle, 
  AlertTriangle, 
  AlertOctagon, 
  HelpCircle, 
  Sparkles, 
  ArrowRight,
  ShieldAlert,
  Clock,
  Layers
} from 'lucide-react';

interface VisualInspectionProps {
  inspection: InspectionResponse | null;
  onInspect: (file: File, category: string) => Promise<void>;
  loading: boolean;
  selectedCategory: string;
  onViewSpatial: () => void;
  onAskVISTA: () => void;
}

export const VisualInspection: React.FC<VisualInspectionProps> = ({
  inspection,
  onInspect,
  loading,
  selectedCategory,
  onViewSpatial,
  onAskVISTA
}) => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Preset Benchmark Samples
  const presets = [
    { name: 'Broken Large (Defect)', category: 'bottle', type: 'defect', desc: 'Severe structural fracture at mouth', color: 'from-rose-500/20 to-red-600/10' },
    { name: 'Contamination (Defect)', category: 'bottle', type: 'defect', desc: 'Foreign chemical residue in interior', color: 'from-amber-500/20 to-orange-600/10' },
    { name: 'Normal Sample (Clean)', category: 'bottle', type: 'normal', desc: 'Pristine bottle with nominal geometry', color: 'from-emerald-500/20 to-teal-600/10' },
    { name: 'Cable Cut Inner (Defect)', category: 'cable', type: 'defect', desc: 'Severed internal conductor wire', color: 'from-indigo-500/20 to-purple-600/10' }
  ];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const loadPreset = (presetName: string) => {
    // Generate a lightweight canvas preview for testing
    const canvas = document.createElement('canvas');
    canvas.width = 518;
    canvas.height = 518;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, 518, 518);
      // Draw bottle silhouette
      ctx.fillStyle = '#475569';
      ctx.beginPath();
      ctx.ellipse(259, 300, 100, 160, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillRect(219, 90, 80, 80); // neck

      if (presetName.includes('Broken')) {
        // Red defect artifact
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(280, 120, 25, 0, Math.PI * 2);
        ctx.fill();
      } else if (presetName.includes('Contamination')) {
        // Amber stain artifact
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.arc(240, 320, 30, 0, Math.PI * 2);
        ctx.fill();
      }

      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], `${presetName.toLowerCase().replace(/\s+/g, '_')}.png`, { type: 'image/png' });
          setSelectedFile(file);
          setPreviewUrl(canvas.toDataURL());
        }
      });
    }
  };

  const triggerInspect = () => {
    if (selectedFile) {
      onInspect(selectedFile, selectedCategory);
    }
  };

  const renderStatusBadge = (status: InspectionStatus) => {
    switch (status) {
      case 'NORMAL':
        return (
          <span className="badge-normal px-3 py-1 rounded-full text-xs font-bold inline-flex items-center space-x-1.5 shadow-sm">
            <CheckCircle className="w-3.5 h-3.5" />
            <span>NORMAL — NOMINAL</span>
          </span>
        );
      case 'ANOMALOUS':
        return (
          <span className="badge-anomalous px-3 py-1 rounded-full text-xs font-bold inline-flex items-center space-x-1.5 shadow-sm">
            <AlertOctagon className="w-3.5 h-3.5" />
            <span>ANOMALOUS — DEFECT DETECTED</span>
          </span>
        );
      case 'UNCERTAIN':
        return (
          <span className="badge-uncertain px-3 py-1 rounded-full text-xs font-bold inline-flex items-center space-x-1.5 shadow-sm">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>UNCERTAIN — CONFORMAL BAND</span>
          </span>
        );
      case 'OOD':
        return (
          <span className="badge-ood px-3 py-1 rounded-full text-xs font-bold inline-flex items-center space-x-1.5 shadow-sm">
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>OUT-OF-DISTRIBUTION (OOD)</span>
          </span>
        );
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner / Pipeline Description */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 relative overflow-hidden">
        <div className="absolute -right-16 -top-16 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 relative z-10">
          <div>
            <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold tracking-wider uppercase mb-1">
              <Sparkles className="w-4 h-4" />
              <span>Industrial Vision Intelligence Pipeline</span>
            </div>
            <h1 className="text-2xl font-extrabold text-white tracking-tight">
              Real-Time Visual Anomaly Inspection
            </h1>
            <p className="text-sm text-slate-400 max-w-2xl mt-1">
              Evaluates surface geometry using frozen DINOv2 ViT-S/14 patch features, Global-Local FiLM fusion, and split conformal calibration guaranteeing false-alarm control.
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={triggerInspect}
              disabled={!selectedFile || loading}
              className={`px-5 py-2.5 rounded-xl text-xs font-bold flex items-center space-x-2 shadow-lg transition-all ${
                !selectedFile || loading
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:from-cyan-400 hover:to-blue-500 shadow-cyan-500/25 active:scale-95'
              }`}
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  <span>Analyzing Feature Grid...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Run VISTA Inspection</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Input Image Dropzone & Presets (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div 
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[320px] ${
              previewUrl
                ? 'border-cyan-500/40 bg-slate-900/60'
                : 'border-slate-800 hover:border-slate-700 bg-slate-950/40'
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              className="hidden"
            />
            {previewUrl ? (
              <div className="relative group w-full flex flex-col items-center">
                <img
                  src={previewUrl}
                  alt="Inspection target"
                  className="max-h-[260px] object-contain rounded-xl shadow-md border border-slate-700"
                />
                <span className="text-xs text-slate-400 mt-2 block font-mono">
                  {selectedFile?.name || 'custom_upload.png'} ({(selectedFile?.size ? (selectedFile.size / 1024).toFixed(1) : '24.5')} KB)
                </span>
                <span className="text-[11px] text-cyan-400 group-hover:underline mt-1">Click to replace image</span>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="w-12 h-12 rounded-2xl bg-slate-900 flex items-center justify-center mx-auto text-slate-400 border border-slate-800">
                  <Upload className="w-6 h-6 text-cyan-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-200">
                    Drop sample image or click to browse
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    Accepts PNG, JPG, BMP up to 20MB (518×518 resized)
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Quick Presets */}
          <div className="glass-panel p-4 rounded-2xl border border-slate-800">
            <h4 className="text-xs font-semibold uppercase text-slate-400 tracking-wider mb-2.5">
              Quick Benchmark Presets
            </h4>
            <div className="grid grid-cols-2 gap-2">
              {presets.map((p) => (
                <button
                  key={p.name}
                  onClick={() => loadPreset(p.name)}
                  className={`text-left p-2.5 rounded-xl border border-slate-800 hover:border-slate-700 bg-slate-950/60 transition-all text-xs`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-slate-200 truncate">{p.name}</span>
                    <span className={`w-2 h-2 rounded-full ${p.type === 'defect' ? 'bg-rose-400' : 'bg-emerald-400'}`}></span>
                  </div>
                  <p className="text-[10px] text-slate-400 line-clamp-1">{p.desc}</p>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Inspection Findings & Metrics (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          {inspection ? (
            <>
              {/* Status Header Card */}
              <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <span className="text-[11px] text-slate-500 font-mono">INSPECTION ID: {inspection.inspection_id}</span>
                    <div className="flex items-center space-x-3">
                      {renderStatusBadge(inspection.status)}
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-[11px] text-slate-500 block">Target Category</span>
                    <span className="text-xs font-bold text-slate-200 uppercase font-mono">{inspection.category}</span>
                  </div>
                </div>

                {/* Score vs Threshold Gauge Component */}
                <MetricsGauge
                  score={inspection.image_anomaly_score}
                  status={inspection.status}
                  reliability={inspection.reliability}
                />

                {/* Natural Language Interpretation */}
                <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 text-xs">
                  <span className="text-[10px] uppercase tracking-wider font-bold text-slate-400 block mb-1">
                    Multimodal Interpretation Summary
                  </span>
                  <p className="text-slate-200 leading-relaxed font-medium">
                    {inspection.textual_interpretation}
                  </p>
                </div>

                {/* Action Buttons to Spatial & Multimodal Tabs */}
                <div className="flex items-center justify-end space-x-3 pt-2">
                  <button
                    onClick={onViewSpatial}
                    className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 text-slate-200 hover:bg-slate-700 flex items-center space-x-2 transition-all"
                  >
                    <Layers className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Explore Spatial Heatmaps</span>
                  </button>

                  <button
                    onClick={onAskVISTA}
                    className="px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-cyan-600 to-blue-600 text-white hover:from-cyan-500 hover:to-blue-500 flex items-center space-x-2 transition-all shadow-md shadow-cyan-500/20"
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>View Grounded Defect Q&A</span>
                    <ArrowRight className="w-3.5 h-3.5 ml-1" />
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="glass-panel p-10 rounded-2xl border border-slate-800 text-center flex flex-col items-center justify-center min-h-[380px] space-y-4">
              <div className="w-16 h-16 rounded-3xl bg-slate-900 border border-slate-800 flex items-center justify-center text-cyan-400 shadow-inner">
                <ShieldAlert className="w-8 h-8 opacity-60" />
              </div>
              <div className="max-w-md space-y-1">
                <h3 className="text-base font-bold text-slate-200">No Active Inspection</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Select an image or choose one of the benchmark sample presets above, then click <strong className="text-cyan-300">Run VISTA Inspection</strong> to execute DINOv2 feature extraction, progressive decoding, and conformal reliability checks.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
