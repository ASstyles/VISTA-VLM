import React, { useState } from 'react';
import { InspectionResponse, RegionFinding } from '../types';
import { Layers, Eye, Sliders, Maximize2, ShieldAlert, Crosshair } from 'lucide-react';

interface SpatialViewerProps {
  inspection: InspectionResponse | null;
}

export const SpatialViewer: React.FC<SpatialViewerProps> = ({ inspection }) => {
  const [opacity, setOpacity] = useState<number>(0.65);
  const [activeTab, setActiveTab] = useState<'overlay' | 'side_by_side' | 'mask'>('overlay');
  const [hoveredRegionIndex, setHoveredRegionIndex] = useState<number | null>(null);

  if (!inspection) {
    return (
      <div className="glass-panel p-12 rounded-2xl border border-slate-800 text-center flex flex-col items-center justify-center min-h-[450px] space-y-4">
        <div className="w-16 h-16 rounded-3xl bg-slate-900 border border-slate-800 flex items-center justify-center text-cyan-400">
          <Layers className="w-8 h-8 opacity-60" />
        </div>
        <div>
          <h3 className="text-base font-bold text-slate-200">No Spatial Data Available</h3>
          <p className="text-xs text-slate-400 mt-1 max-w-sm">
            Please run an inspection first from the Inspection tab to generate dense anomaly heatmaps, segmentation masks, and bounding boxes.
          </p>
        </div>
      </div>
    );
  }

  const regions: RegionFinding[] = inspection.region_findings || [];

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold tracking-wider uppercase mb-1">
            <Layers className="w-4 h-4" />
            <span>Dense Spatial Anomaly Representation</span>
          </div>
          <h2 className="text-xl font-extrabold text-white">
            518×518 Pixel-Level Anomaly Localization
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Decoded from multi-layer DINOv2 ViT-S patch tokens (37×37 grid) via progressive depthwise-separable conv layers.
          </p>
        </div>

        {/* View Mode Controls */}
        <div className="flex items-center space-x-2 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
          <button
            onClick={() => setActiveTab('overlay')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              activeTab === 'overlay'
                ? 'bg-cyan-500 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Heatmap Overlay
          </button>
          <button
            onClick={() => setActiveTab('side_by_side')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              activeTab === 'side_by_side'
                ? 'bg-cyan-500 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Side-by-Side
          </button>
          <button
            onClick={() => setActiveTab('mask')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              activeTab === 'mask'
                ? 'bg-cyan-500 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Binary Mask
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Main Canvas Viewer (8 Cols) */}
        <div className="lg:col-span-8 space-y-4">
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 flex flex-col items-center justify-center relative min-h-[480px]">
            {activeTab === 'overlay' && (
              <div className="relative max-w-[500px] w-full aspect-square rounded-xl overflow-hidden border border-slate-700 shadow-2xl bg-slate-950">
                {/* Heatmap Layer */}
                {inspection.heatmap_base64 && (
                  <img
                    src={`data:image/png;base64,${inspection.heatmap_base64}`}
                    alt="Anomaly Heatmap Overlay"
                    className="w-full h-full object-contain"
                    style={{ opacity: opacity }}
                  />
                )}

                {/* Interactive Bounding Boxes Overlay */}
                {regions.map((region, idx) => {
                  const [x1, y1, x2, y2] = region.box;
                  // Scale coordinates from 518 to % (assuming 518 max dim)
                  const left = (x1 / 518) * 100;
                  const top = (y1 / 518) * 100;
                  const width = ((x2 - x1) / 518) * 100;
                  const height = ((y2 - y1) / 518) * 100;
                  const isHovered = hoveredRegionIndex === idx;

                  return (
                    <div
                      key={idx}
                      onMouseEnter={() => setHoveredRegionIndex(idx)}
                      onMouseLeave={() => setHoveredRegionIndex(null)}
                      className={`absolute border-2 transition-all cursor-pointer ${
                        isHovered
                          ? 'border-yellow-400 bg-yellow-400/20 shadow-lg shadow-yellow-400/30 z-20'
                          : 'border-cyan-400 bg-cyan-400/10 z-10'
                      }`}
                      style={{
                        left: `${left}%`,
                        top: `${top}%`,
                        width: `${Math.max(width, 4)}%`,
                        height: `${Math.max(height, 4)}%`
                      }}
                    >
                      <div className="absolute -top-6 left-0 bg-slate-900/90 text-white text-[10px] font-mono px-1.5 py-0.5 rounded border border-cyan-500 whitespace-nowrap shadow">
                        ROI #{idx + 1}: {region.top_defect_label} ({region.spatial_anomaly_score.toFixed(2)})
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {activeTab === 'side_by_side' && (
              <div className="grid grid-cols-2 gap-4 w-full max-w-[640px]">
                <div className="space-y-1.5 text-center">
                  <span className="text-[11px] font-mono text-slate-400">Jet Anomaly Map</span>
                  <div className="aspect-square rounded-xl overflow-hidden border border-slate-700 bg-slate-950">
                    {inspection.heatmap_base64 && (
                      <img
                        src={`data:image/png;base64,${inspection.heatmap_base64}`}
                        alt="JET Heatmap"
                        className="w-full h-full object-contain"
                      />
                    )}
                  </div>
                </div>

                <div className="space-y-1.5 text-center">
                  <span className="text-[11px] font-mono text-slate-400">Segmented Binary Mask</span>
                  <div className="aspect-square rounded-xl overflow-hidden border border-slate-700 bg-slate-950">
                    {inspection.mask_base64 && (
                      <img
                        src={`data:image/png;base64,${inspection.mask_base64}`}
                        alt="Binary Mask"
                        className="w-full h-full object-contain"
                      />
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'mask' && (
              <div className="max-w-[480px] w-full aspect-square rounded-xl overflow-hidden border border-slate-700 shadow-xl bg-slate-950">
                {inspection.mask_base64 && (
                  <img
                    src={`data:image/png;base64,${inspection.mask_base64}`}
                    alt="Binary Anomaly Mask"
                    className="w-full h-full object-contain"
                  />
                )}
              </div>
            )}

            {/* Opacity Slider Control */}
            {activeTab === 'overlay' && (
              <div className="w-full max-w-sm mt-4 flex items-center space-x-3 bg-slate-950/80 px-4 py-2 rounded-xl border border-slate-800">
                <Sliders className="w-4 h-4 text-slate-400" />
                <span className="text-xs text-slate-400 font-mono">Heatmap Opacity</span>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
                <span className="text-xs font-mono font-bold text-cyan-400">
                  {Math.round(opacity * 100)}%
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Localized ROIs & Defect Table (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
                <Crosshair className="w-4 h-4 text-cyan-400" />
                <span>Detected ROIs ({regions.length})</span>
              </h3>
              <span className="text-[11px] font-mono text-slate-500">Connected Components</span>
            </div>

            {regions.length > 0 ? (
              <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
                {regions.map((r, idx) => (
                  <div
                    key={idx}
                    onMouseEnter={() => setHoveredRegionIndex(idx)}
                    onMouseLeave={() => setHoveredRegionIndex(null)}
                    className={`p-3 rounded-xl border transition-all text-xs space-y-2 cursor-pointer ${
                      hoveredRegionIndex === idx
                        ? 'bg-slate-900 border-yellow-400/80 shadow-md shadow-yellow-400/10'
                        : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-200">ROI #{idx + 1}</span>
                      <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-rose-950/50 text-rose-400 border border-rose-800">
                        Peak: {r.spatial_anomaly_score.toFixed(3)}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-1 text-[11px] text-slate-400 font-mono">
                      <div>Area: <span className="text-slate-200">{r.area} px</span></div>
                      <div>Margin: <span className="text-slate-200">{r.confidence_margin.toFixed(3)}</span></div>
                      <div className="col-span-2">
                        Box: [{r.box.join(', ')}]
                      </div>
                    </div>

                    <div className="pt-1 border-t border-slate-800/80 flex items-center justify-between">
                      <span className="text-[11px] text-slate-400">CLIP Top Class:</span>
                      <span className="font-semibold text-cyan-300 capitalize">{r.top_defect_label}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-6 text-center text-xs text-slate-500 bg-slate-950/40 rounded-xl border border-slate-800">
                No connected anomaly regions exceed spatial threshold.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
