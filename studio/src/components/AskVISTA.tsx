import React, { useState } from 'react';
import { InspectionResponse, RegionFinding, CandidateRanking } from '../types';
import { 
  MessageSquareCode, 
  Send, 
  AlertTriangle, 
  CheckCircle2, 
  HelpCircle, 
  ShieldCheck, 
  Sparkles,
  UserCheck
} from 'lucide-react';

interface AskVISTAProps {
  inspection: InspectionResponse | null;
  onSendFeedback: (feedback: {
    inspection_id: string;
    category: string;
    user_label: string;
    comments: string;
    correct_defect_type?: string;
  }) => Promise<void>;
}

export const AskVISTA: React.FC<AskVISTAProps> = ({ inspection, onSendFeedback }) => {
  const [userQuery, setUserQuery] = useState<string>('');
  const [chatLog, setChatLog] = useState<Array<{ sender: 'user' | 'vista'; text: string; time: string }>>([
    {
      sender: 'vista',
      text: 'Hello, I am VISTA Multimodal Inspector. I analyze visual anomalies using frozen CLIP ViT-B/16 text embeddings aligned with local cropped defect features. How can I assist with this inspection?',
      time: '12:00'
    }
  ]);

  const [feedbackLabel, setFeedbackLabel] = useState<string>('normal');
  const [correctedType, setCorrectedType] = useState<string>('');
  const [feedbackComments, setFeedbackComments] = useState<string>('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<boolean>(false);

  const handleAsk = (e: React.FormEvent) => {
    e.preventDefault();
    if (!userQuery.trim()) return;

    const q = userQuery.trim().toLowerCase();
    const newChat = [...chatLog, { sender: 'user' as const, text: userQuery, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }];

    let response = '';
    if (!inspection) {
      response = 'Please run an inspection first so I can ground my responses in the extracted patch features and defect candidates.';
    } else if (q.includes('defect') || q.includes('what is') || q.includes('type')) {
      const topDefects = inspection.region_findings?.map(r => r.top_defect_label).join(', ') || 'none';
      response = `Based on CLIP crop feature alignment, localized ROIs show characteristics of: ${topDefects}. Confidence margin: ${inspection.region_findings?.[0]?.confidence_margin.toFixed(3) || 'N/A'}.`;
    } else if (q.includes('confidence') || q.includes('p-value') || q.includes('conformal')) {
      response = `Split conformal calibration was applied with non-conformity threshold τ = ${inspection.reliability.calibrated_threshold.toFixed(4)}. Empirical conformal p-value is ${inspection.reliability.conformal_p_value.toFixed(4)}.`;
    } else if (q.includes('ood') || q.includes('out of distribution')) {
      response = inspection.reliability.is_ood 
        ? `Yes, the input is flagged as Out-of-Distribution (distance ${inspection.reliability.ood_distance.toFixed(3)} exceeds threshold ${inspection.reliability.ood_threshold.toFixed(3)}). Human review is strongly recommended.`
        : `No, global CLS embeddings are in-distribution relative to learned ${inspection.category} prototypes.`;
    } else {
      response = `VISTA inspection status is ${inspection.status} with image score ${inspection.image_anomaly_score.toFixed(4)} (Threshold: ${inspection.threshold.toFixed(4)}). ${inspection.textual_interpretation}`;
    }

    newChat.push({
      sender: 'vista',
      text: response,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    });

    setChatLog(newChat);
    setUserQuery('');
  };

  const handleFeedbackSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inspection) return;

    await onSendFeedback({
      inspection_id: inspection.inspection_id,
      category: inspection.category,
      user_label: feedbackLabel,
      comments: feedbackComments,
      correct_defect_type: correctedType || undefined
    });

    setFeedbackSubmitted(true);
    setTimeout(() => setFeedbackSubmitted(false), 4000);
  };

  const regions = inspection?.region_findings || [];
  const topRegion = regions[0];

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold tracking-wider uppercase mb-1">
            <MessageSquareCode className="w-4 h-4" />
            <span>Multimodal Verification & Defect Grounding</span>
          </div>
          <h2 className="text-xl font-extrabold text-white">
            Ask VISTA Multimodal Intelligence
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Cross-modal semantic verification using CLIP ViT-B/16 image crops and candidate textual defect templates.
          </p>
        </div>
        <div className="flex items-center space-x-2 text-xs font-mono bg-slate-950 px-3 py-1.5 rounded-xl border border-slate-800 text-slate-300">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>CLIP ViT-B/16 (Frozen)</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Interactive Chat & QA (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col h-[520px]">
            {/* Chat Messages Log */}
            <div className="flex-1 overflow-y-auto space-y-3 pr-2 mb-4">
              {chatLog.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
                >
                  <div
                    className={`max-w-[85%] p-3.5 rounded-2xl text-xs leading-relaxed ${
                      msg.sender === 'user'
                        ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white rounded-br-none shadow-md shadow-cyan-500/20'
                        : 'bg-slate-950/80 border border-slate-800 text-slate-200 rounded-bl-none'
                    }`}
                  >
                    <div className="flex items-center space-x-1.5 mb-1 opacity-70 text-[10px] font-mono">
                      <span>{msg.sender === 'user' ? 'Operator' : 'VISTA-VLM'}</span>
                      <span>•</span>
                      <span>{msg.time}</span>
                    </div>
                    {msg.text}
                  </div>
                </div>
              ))}
            </div>

            {/* Input Bar */}
            <form onSubmit={handleAsk} className="flex items-center space-x-2 pt-2 border-t border-slate-800">
              <input
                type="text"
                value={userQuery}
                onChange={(e) => setUserQuery(e.target.value)}
                placeholder="Ask about defect type, confidence margin, or conformal p-value..."
                className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500"
              />
              <button
                type="submit"
                className="bg-cyan-500 hover:bg-cyan-400 text-white p-2.5 rounded-xl transition-all shadow-md shadow-cyan-500/20"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </div>

        {/* Right Column: CLIP Candidate Rankings & Feedback (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Candidate Rankings Card */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                CLIP Candidate Rankings (ROI #1)
              </h3>
              <span className="text-[11px] font-mono text-cyan-400">Cosine Similarity</span>
            </div>

            {topRegion && topRegion.candidate_rankings.length > 0 ? (
              <div className="space-y-2">
                {topRegion.candidate_rankings.map((c, i) => (
                  <div
                    key={i}
                    className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800 text-xs space-y-1.5"
                  >
                    <div className="flex items-center justify-between font-semibold">
                      <span className="text-slate-200 capitalize">#{i + 1} {c.label}</span>
                      <span className="font-mono text-cyan-300 font-bold">{c.score.toFixed(4)}</span>
                    </div>
                    <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full rounded-full"
                        style={{ width: `${Math.min(100, Math.max(0, c.score * 100))}%` }}
                      />
                    </div>
                    <p className="text-[10px] text-slate-500 font-mono truncate">
                      Prompt: "{c.prompt}"
                    </p>
                  </div>
                ))}

                {/* Ambiguity Alert */}
                {topRegion.is_ambiguous && (
                  <div className="p-2.5 rounded-xl bg-amber-950/40 border border-amber-800/80 text-amber-300 text-xs flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                    <span>Ambiguity Alert: Confidence margin &lt; 0.05 between top candidates.</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="p-4 rounded-xl bg-slate-950/40 border border-slate-800 text-center text-xs text-slate-500">
                No defect crop candidates to rank. Run inspection on an anomalous sample.
              </div>
            )}
          </div>

          {/* Operator Ground-Truth Feedback Form */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-400">
              <UserCheck className="w-4 h-4 text-emerald-400" />
              <span>Inspector Feedback & Correction</span>
            </div>
            <p className="text-[11px] text-slate-400">
              Submit ground-truth labels for selective prediction re-calibration (stored separately).
            </p>

            <form onSubmit={handleFeedbackSubmit} className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setFeedbackLabel('normal')}
                  className={`p-2 rounded-xl border font-semibold transition-all ${
                    feedbackLabel === 'normal'
                      ? 'bg-emerald-950/50 border-emerald-500 text-emerald-300'
                      : 'bg-slate-950 border-slate-800 text-slate-400'
                  }`}
                >
                  Confirm Normal
                </button>
                <button
                  type="button"
                  onClick={() => setFeedbackLabel('anomalous')}
                  className={`p-2 rounded-xl border font-semibold transition-all ${
                    feedbackLabel === 'anomalous'
                      ? 'bg-rose-950/50 border-rose-500 text-rose-300'
                      : 'bg-slate-950 border-slate-800 text-slate-400'
                  }`}
                >
                  Confirm Defect
                </button>
              </div>

              {feedbackLabel === 'anomalous' && (
                <div>
                  <label className="text-[11px] text-slate-400 block mb-1">True Defect Class (Optional)</label>
                  <input
                    type="text"
                    value={correctedType}
                    onChange={(e) => setCorrectedType(e.target.value)}
                    placeholder="e.g. broken_large, contamination, scratch"
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-cyan-500"
                  />
                </div>
              )}

              <div>
                <label className="text-[11px] text-slate-400 block mb-1">Inspector Notes</label>
                <textarea
                  rows={2}
                  value={feedbackComments}
                  onChange={(e) => setFeedbackComments(e.target.value)}
                  placeholder="Surface crack verified under microscope..."
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-cyan-500 resize-none"
                />
              </div>

              <button
                type="submit"
                disabled={!inspection}
                className={`w-full py-2.5 rounded-xl font-bold transition-all ${
                  !inspection
                    ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                    : 'bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-md shadow-emerald-500/20'
                }`}
              >
                {feedbackSubmitted ? 'Recorded in Calibration Log!' : 'Submit Inspector Verification'}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};
