import React from 'react';
import { 
  ShieldCheck, 
  Layers, 
  MessageSquareCode, 
  FlaskConical, 
  BarChart3, 
  Cpu, 
  Sparkles 
} from 'lucide-react';

export type ActiveTab = 'inspection' | 'spatial' | 'ask_vista' | 'research' | 'benchmarks';

interface NavbarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  selectedCategory: string;
  setSelectedCategory: (cat: string) => void;
  systemHealthy: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  selectedCategory,
  setSelectedCategory,
  systemHealthy
}) => {
  const categories = [
    { id: 'bottle', name: 'MVTec: Bottle' },
    { id: 'cable', name: 'MVTec: Cable' },
    { id: 'leather', name: 'MVTec: Leather' },
    { id: 'capsule', name: 'MVTec: Capsule' },
    { id: 'tile', name: 'MVTec: Tile' },
    { id: 'candle', name: 'VisA: Candle' },
    { id: 'pcb1', name: 'VisA: PCB1' }
  ];

  return (
    <header className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 px-6 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-500 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-cyan-300 bg-clip-text text-transparent">
                VISTA-VLM
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-cyan-950/80 text-cyan-400 border border-cyan-800">
                v1.0-ViT-S/14
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium">Vision Intelligence through Spatial & Textual Alignment</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1 bg-slate-950/80 p-1.5 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveTab('inspection')}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'inspection'
                ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/25'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <ShieldCheck className="w-4 h-4" />
            <span>Inspection</span>
          </button>

          <button
            onClick={() => setActiveTab('spatial')}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'spatial'
                ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/25'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Layers className="w-4 h-4" />
            <span>Spatial Viewer</span>
          </button>

          <button
            onClick={() => setActiveTab('ask_vista')}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'ask_vista'
                ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/25'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <MessageSquareCode className="w-4 h-4" />
            <span>Ask VISTA</span>
          </button>

          <button
            onClick={() => setActiveTab('research')}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'research'
                ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/25'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <FlaskConical className="w-4 h-4" />
            <span>Research Lab</span>
          </button>

          <button
            onClick={() => setActiveTab('benchmarks')}
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'benchmarks'
                ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/25'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            <span>Benchmark Center</span>
          </button>
        </nav>

        {/* Right Controls: Category & Status */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5 text-xs text-slate-400 bg-slate-950/60 px-3 py-1.5 rounded-lg border border-slate-800">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="bg-transparent text-slate-200 font-medium focus:outline-none cursor-pointer"
            >
              {categories.map((c) => (
                <option key={c.id} value={c.id} className="bg-slate-900 text-slate-100">
                  {c.name}
                </option>
              ))}
            </select>
          </div>

          <div className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono border ${
            systemHealthy 
              ? 'bg-emerald-950/50 text-emerald-400 border-emerald-800/80' 
              : 'bg-amber-950/50 text-amber-400 border-amber-800/80'
          }`}>
            <span className={`w-2 h-2 rounded-full ${systemHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
            <span>{systemHealthy ? 'API ONLINE' : 'STANDALONE MODE'}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
