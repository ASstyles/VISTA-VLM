import React from 'react';
import { 
  ShieldCheck, 
  Layers, 
  MessageSquareCode, 
  FlaskConical, 
  BarChart3, 
  Cpu, 
  Sparkles,
  Activity,
  CheckCircle2,
  AlertCircle
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

  const navItems = [
    { id: 'inspection' as ActiveTab, label: 'Inspection', icon: ShieldCheck, badge: 'Live' },
    { id: 'spatial' as ActiveTab, label: 'Spatial Viewer', icon: Layers, badge: '518px' },
    { id: 'ask_vista' as ActiveTab, label: 'Ask VISTA', icon: MessageSquareCode, badge: 'CLIP' },
    { id: 'research' as ActiveTab, label: 'Research Lab', icon: FlaskConical, badge: 'Ablations' },
    { id: 'benchmarks' as ActiveTab, label: 'Benchmark Center', icon: BarChart3, badge: '7 Models' },
  ];

  return (
    <header className="sticky top-0 z-50 bg-slate-950/85 backdrop-blur-xl border-b border-slate-800/80 px-4 sm:px-6 py-2.5 transition-all">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
        {/* Brand & System Tag */}
        <div className="flex items-center space-x-3 w-full md:w-auto justify-between md:justify-start">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 ring-1 ring-cyan-400/40">
                <Sparkles className="w-5 h-5 text-white animate-pulse-subtle" />
              </div>
              <span className="absolute -bottom-1 -right-1 w-3.5 h-3.5 bg-emerald-500 rounded-full border-2 border-slate-950 flex items-center justify-center">
                <span className="w-1.5 h-1.5 bg-white rounded-full"></span>
              </span>
            </div>

            <div>
              <div className="flex items-center space-x-2">
                <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-100 to-cyan-300 bg-clip-text text-transparent">
                  VISTA-VLM
                </span>
                <span className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full font-mono bg-cyan-950/80 text-cyan-300 border border-cyan-800/80 font-bold">
                  V7.0 Industrial Studio
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-medium">Vision Intelligence through Spatial & Textual Alignment</p>
            </div>
          </div>

          {/* Mobile Status Dot */}
          <div className="md:hidden flex items-center">
            <span className={`w-2.5 h-2.5 rounded-full ${systemHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800/80 shadow-inner overflow-x-auto max-w-full">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`relative flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all whitespace-nowrap cursor-pointer ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/30'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                <span>{item.label}</span>
                {item.badge && (
                  <span className={`text-[9px] font-mono px-1.5 py-0.2 rounded-md ${
                    isActive 
                      ? 'bg-white/20 text-white' 
                      : 'bg-slate-800 text-slate-400'
                  }`}>
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Right Controls: Category & Status */}
        <div className="flex items-center space-x-2.5 w-full md:w-auto justify-end">
          <div className="flex items-center space-x-1.5 text-xs text-slate-300 bg-slate-900/90 px-3 py-1.5 rounded-xl border border-slate-800/80 shadow-sm">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="bg-transparent text-slate-200 font-medium focus:outline-none cursor-pointer pr-1"
            >
              {categories.map((c) => (
                <option key={c.id} value={c.id} className="bg-slate-900 text-slate-100">
                  {c.name}
                </option>
              ))}
            </select>
          </div>

          <div className={`hidden sm:flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-[11px] font-mono border ${
            systemHealthy 
              ? 'bg-emerald-950/40 text-emerald-300 border-emerald-700/60 shadow-sm shadow-emerald-900/20' 
              : 'bg-amber-950/40 text-amber-300 border-amber-700/60 shadow-sm shadow-amber-900/20'
          }`}>
            <span className={`w-2 h-2 rounded-full ${systemHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
            <span className="font-semibold">{systemHealthy ? 'API ONLINE (8000)' : 'STANDALONE MODE'}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
