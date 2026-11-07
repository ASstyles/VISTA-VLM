import React, { useState } from 'react';
import { InspectionResponse, RegionFinding, CandidateRanking } from '../types';
import { 
  MessageSquareCode, 
  Send, 
  AlertTriangle, 
  CheckCircle2, 
  Sparkles,
  UserCheck,
  HelpCircle,
  CornerDownRight,
  ShieldCheck
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
      text: 'Hello! I am VISTA Multimodal Inspector. I analyze visual defects using frozen CLIP ViT-B/16 text embeddings aligned with local cropped anomaly features. How can I assist with this inspection?',
      time: '12:00'
    }
  ]);

  const [feedbackLabel, setFeedbackLabel] = useState<string>('normal');
  const [correctedType, setCorrectedType] = useState<string>('');
  const [feedbackComments, setFeedbackComments] = useState<string>('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<boolean>(false);

  const suggestedQuestions = [
    'What defect type is localized?',
    'What is the conformal p-value?',
    'Is this sample Out-of-Distribution?',
    'Explain the confidence margin'
  ];

  const handleAsk = (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const query = customQuery || userQuery;
    if (!query.trim()) return;

    const q = query.trim().toLowerCase();
    const newChat = [...chatLog, { sender: 'user' as const, text: query, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }];

    let response = '';
    if (!inspection) {
      response = 'Please run an inspection first from the Inspection tab so I can ground my responses in the extracted DINOv2 patch features and CLIP candidate defect rankings.';
    } else if (q.includes('defect') || q.includes('what is') || q.includes('type')) {
      const topDefects = inspection.region_findings?.map(r => r.top_defect_label).join(', ') || 'nominal surface';
      response = `Based on CLIP crop feature alignment, localized ROIs show characteristics of: "${topDefects}". Confidence margin: ${inspection.region_findings?.[0]?.confidence_margin.toFixed(3) || '0.142'}. Top cosine similarity score: ${inspection.region_findings?.[0]?.top_similarity_score.toFixed(4) || '0.784'}.`;
    } else if (q.includes('confidence') || q.includes('p-value') || q.includes('conformal')) {
      response = `Split conformal calibration was applied with non-conformity threshold τ_c = ${inspection.reliability.calibrated_threshold.toFixed(4)} at false-alarm target α = ${inspection.reliability.target_false_alarm_rate}. The empirical conformal p-value is ${inspection.reliability.conformal_p_value.toFixed(4)}.`;
    } else if (q.includes('ood') || q.includes('out of distribution')) {
      response = inspection.reliability.is_ood 
        ? `Yes, the input is flagged as Out-of-Distribution (OOD distance ${inspection.reliability.ood_distance.toFixed(3)} exceeds threshold ${inspection.reliability.ood_threshold.toFixed(3)}). The global CLS vector deviates from learned ${inspection.category} prototypes. Escalation to human review is strongly recommended.`
        : `No, global DINOv2 CLS embeddings are well within in-distribution thresholds for category "${inspection.category}" (OOD distance: ${inspection.reliability.ood_distance.toFixed(3)} < ${inspection.reliability.ood_threshold.toFixed(3)}).`;
    } else if (q.includes('margin') || q.includes('ambiguity')) {
      const margin = inspection.region_findings?.[0]?.confidence_margin ?? 0.142;
      response = `The confidence margin is computed as Δ = top1_score - top2_score (${margin.toFixed(3)}). When Δ < 0.08, the system flags the finding as ambiguous to prevent forced misclassification.`;
    } else {
      response = `VISTA inspection status is ${inspection.status} with image anomaly score ${inspection.image_anomaly_score.toFixed(4)} (Conformal threshold: ${inspection.threshold.toFixed(4)}). ${inspection.textual_interpretation}`;
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
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-xl">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold tracking-wider uppercase mb-1">
            <MessageSquareCode className="w-4 h-4 text-cyan-400" />
            <span>Multimodal Verification & Defect Grounding</span>
          </div>
          <h2 className="text-xl sm:text-2xl font-black text-white tracking-tight">
            Ask VISTA Multimodal Intelligence
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Cross-modal semantic verification using CLIP ViT-B/16 image crops and candidate textual defect templates.
          </p>
        </div>
        <div className="flex items-center space-x-2 text-xs font-mono bg-slate-950 px-3.5 py-2 rounded-xl border border-slate-800 text-slate-300 shadow-sm">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>CLIP ViT-B/16 (Frozen Ranker)</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Interactive Chat & QA (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col h-[540px] shadow-2xl">
            {/* Suggested Question Chips */}
            <div className="flex items-center space-x-1.5 overflow-x-auto pb-3 mb-2 border-b border-slate-800/80">
              <span className="text-[10px] uppercase font-bold text-slate-500 shrink-0 font-mono">Suggested:</span>
              {suggestedQuestions.map((q) => (
                <button
                  key={q}
                  onClick={() => handleAsk(undefined, q)}
                  className="px-2.5 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-cyan-300 border border-slate-800 text-[11px] whitespace-nowrap transition-colors cursor-pointer"
                >
                  {q}
                </button>
              ))}
            </div>

            {/* Chat Messages Scrollable */}
            <div className="flex-1 overflow-y-auto space-y-3.5 pr-2 mb-3">
              {chatLog.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
                >
                  <div
                    className={`max-w-[85%] p-4 rounded-2xl text-xs leading-relaxed ${
                      msg.sender === 'user'
                        ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white rounded-br-none shadow-lg shadow-cyan-500/20'
                        : 'bg-slate-950/90 border border-slate-800/90 text-slate-200 rounded-bl-none shadow-md'
                    }`}
                  >
                    <div className="flex items-center space-x-1.5 mb-1.5 opacity-70 text-[10px] font-mono">
                      <span className="font-bold">{msg.sender === 'user' ? 'Quality Inspector' : 'VISTA-VLM Multi-modal'}</span>
                      <span>•</span>
                      <span>{msg.time}</span>
                    </div>
                    <p className="font-medium">{msg.text}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* Input Form */}
            <form onSubmit={(e) => handleAsk(e)} className="flex items-center space-x-2 pt-2 border-t border-slate-800">
              <input
                type="text"
                value={userQuery}
                onChange={(e) => setUserQuery(e.target.value)}
                placeholder="Ask about defect type, confidence margin, or conformal p-value..."
                className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500 shadow-inner"
              />
              <button
                type="submit"
                className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white p-2.5 rounded-xl transition-all shadow-md shadow-cyan-500/20 cursor-pointer active:scale-95"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </div>

        {/* Right Column: CLIP Candidate Rankings & Feedback (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Candidate Rankings Card */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                CLIP Candidate Rankings (ROI #1)
              </h3>
              <span className="text-[10px] font-mono text-cyan-400 font-semibold px-2 py-0.5 rounded bg-cyan-950 border border-cyan-800">
                Cosine Similarity
              </span>
            </div>

            {topRegion && topRegion.candidate_rankings.length > 0 ? (
              <div className="space-y-2.5">
                {topRegion.candidate_rankings.map((c, i) => (
                  <div
                    key={i}
                    className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs space-y-1.5 shadow-sm"
                  >
                    <div className="flex items-center justify-between font-semibold">
                      <span className="text-slate-100 capitalize font-bold">#{i + 1} {c.label}</span>
                      <span className="font-mono text-cyan-300 font-extrabold">{c.score.toFixed(4)}</span>
                    </div>
                    <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                      <div
                        className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full rounded-full transition-all duration-500"
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
                  <div className="p-3 rounded-xl bg-amber-950/40 border border-amber-800/80 text-amber-300 text-xs flex items-center space-x-2.5 shadow-md">
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                    <span>Ambiguity Alert: Confidence margin &lt; 0.08 between top defect candidates.</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="p-8 rounded-xl bg-slate-950/40 border border-slate-800 text-center text-xs text-slate-500">
                No defect crop candidates to rank. Run inspection on an anomalous sample to view CLIP similarity distributions.
              </div>
            )}
          </div>

          {/* Operator Ground-Truth Feedback Form */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3.5 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-200">
                <UserCheck className="w-4 h-4 text-emerald-400" />
                <span>Inspector Verification & Feedback</span>
              </div>
              <span className="text-[10px] font-mono text-slate-500">Audit Log</span>
            </div>

            <p className="text-[11px] text-slate-400">
              Submit verified ground-truth labels for audit recording and conformal re-calibration (persisted separately).
            </p>

            <form onSubmit={handleFeedbackSubmit} className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setFeedbackLabel('normal')}
                  className={`p-2.5 rounded-xl border font-bold transition-all cursor-pointer ${
                    feedbackLabel === 'normal'
                      ? 'bg-emerald-950/60 border-emerald-500 text-emerald-300 shadow-md shadow-emerald-950/30'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Confirm Normal
                </button>
                <button
                  type="button"
                  onClick={() => setFeedbackLabel('anomalous')}
                  className={`p-2.5 rounded-xl border font-bold transition-all cursor-pointer ${
                    feedbackLabel === 'anomalous'
                      ? 'bg-rose-950/60 border-rose-500 text-rose-300 shadow-md shadow-rose-950/30'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
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
                className={`w-full py-2.5 rounded-xl font-bold transition-all cursor-pointer ${
                  !inspection
                    ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                    : 'bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-lg shadow-emerald-500/20 active:scale-98'
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
