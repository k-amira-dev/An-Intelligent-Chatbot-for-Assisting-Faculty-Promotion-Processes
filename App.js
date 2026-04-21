import { useState, useEffect, useRef } from "react";

const API_URL = "https://ranger-orientation-pierce-ala.trycloudflare.com/chat";
const METRIC_LABELS = {
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer Relevancy",
  context_precision: "Context Precision",
  context_recall: "Context Recall",
};

const METRIC_DESCRIPTIONS = {
  faithfulness: "How factually consistent the answer is with the retrieved context.",
  answer_relevancy: "How relevant the answer is to the question asked.",
  context_precision: "How precisely the retrieved context relates to the question.",
  context_recall: "How much relevant information was captured. Requires a ground truth answer.",
};

function scoreColor(v) {
  if (v === null || v === undefined) return "text-gray-400";
  return v >= 0.7 ? "text-emerald-600" : v >= 0.4 ? "text-amber-500" : "text-red-500";
}

function scoreBg(v) {
  if (v === null || v === undefined) return "bg-gray-200";
  return v >= 0.7 ? "bg-emerald-500" : v >= 0.4 ? "bg-amber-400" : "bg-red-400";
}

// ── Evaluation panel ─────────────────────────────────────────────────────────

function EvaluationPanel({ evaluation, groundTruthUsed }) {
  const [open, setOpen] = useState(false);
  const [hovered, setHovered] = useState(null);

  const values = Object.values(evaluation).filter((v) => v !== null && v !== undefined);
  const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;

  return (
    <div className="mt-3 border border-gray-200 rounded-xl overflow-hidden text-xs bg-white">

      {/* Header row */}
      <button
        onClick={() => setOpen((p) => !p)}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="font-semibold text-gray-600">Quality Metrics</span>
          {groundTruthUsed && (
            <span className="bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded text-[10px] font-medium">GT used</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {avg !== null && <span className={`font-mono font-bold ${scoreColor(avg)}`}>{avg.toFixed(2)} avg</span>}
          <svg className={`w-3.5 h-3.5 text-gray-400 transition-transform ${open ? "rotate-180" : ""}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Expanded metrics */}
      {open && (
        <div className="px-3 pb-3 pt-2 space-y-2 border-t border-gray-100">
          {Object.entries(evaluation).map(([key, value]) => (
            <div key={key} className="relative">
              <div
                className="flex items-center gap-2 cursor-default"
                onMouseEnter={() => setHovered(key)}
                onMouseLeave={() => setHovered(null)}
              >
                <span className="w-32 text-gray-500 shrink-0">{METRIC_LABELS[key]}</span>
                <div className="flex-1 bg-gray-200 rounded-full h-1.5 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${scoreBg(value)}`}
                    style={{ width: `${value !== null && value !== undefined ? Math.round(value * 100) : 0}%` }}
                  />
                </div>
                <span className={`w-10 text-right font-mono font-semibold ${scoreColor(value)}`}>
                  {value !== null && value !== undefined ? value.toFixed(2) : "N/A"}
                </span>
              </div>

              {hovered === key && (
                <div className="absolute z-10 left-0 bottom-full mb-1.5 w-64 bg-gray-800 text-white rounded-lg px-3 py-2 shadow-lg pointer-events-none">
                  {METRIC_DESCRIPTIONS[key]}
                  {(value === null || value === undefined) && (
                    <span className="block mt-1 text-gray-400 italic">No ground truth for this query.</span>
                  )}
                </div>
              )}
            </div>
          ))}

          {!groundTruthUsed && (
            <p className="text-gray-400 italic text-[10px] pt-1">
              Context Recall was skipped — add this question to <code>ground_truth.json</code> to enable it.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [error, setError] = useState(null);

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  const currentChat = chats.find((c) => c.id === currentChatId) ?? null;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chats, currentChatId]);

  function createNewChat() {
    const id = Date.now();
    setChats((prev) => [{ id, title: "New chat", messages: [] }, ...prev]);
    setCurrentChatId(id);
    setError(null);
    return id;
  }

  async function sendMessage() {
    const text = input.trim();
    if (!text || loading) return;

    const chatId = currentChatId ?? createNewChat();
    setInput("");
    setLoading(true);
    setError(null);

    setChats((prev) =>
      prev.map((chat) =>
        chat.id === chatId
          ? { ...chat, messages: [...chat.messages,
              { role: "user", content: text },
              { role: "assistant", content: "", loading: true }
            ]}
          : chat
      )
    );

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `Server error ${res.status}`);
      }

      const data = await res.json();

      setChats((prev) =>
        prev.map((chat) => {
          if (chat.id !== chatId) return chat;
          const messages = [...chat.messages];
          messages[messages.length - 1] = {
            role: "assistant",
            content: data.response,
            sources: data.sources ?? [],
            evaluation: data.evaluation ?? null,
            groundTruthUsed: data.ground_truth_used ?? false,
          };
          return {
            ...chat,
            messages,
            title: chat.title === "New chat" ? text.slice(0, 30) : chat.title,
          };
        })
      );
    } catch (err) {
      setError(err.message);
      setChats((prev) =>
        prev.map((chat) => {
          if (chat.id !== chatId) return chat;
          const messages = [...chat.messages];
          messages[messages.length - 1] = { role: "assistant", content: "Something went wrong. Please try again." };
          return { ...chat, messages };
        })
      );
    } finally {
      setLoading(false);
      textareaRef.current?.focus();
    }
  }

  return (
    <div className="flex h-screen bg-gray-50 font-sans">

      {/* ── Sidebar ── */}
      <div className={`bg-white border-r border-gray-200 flex flex-col h-screen transition-all duration-300
        ${sidebarOpen ? "w-64 p-3" : "w-14 p-2 items-center"}`}>

        <div className={`flex mb-5 items-center ${sidebarOpen ? "justify-between" : "flex-col gap-3"}`}>
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 bg-gray-900 rounded-lg flex items-center justify-center flex-shrink-0">
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-3 3v-3z" />
              </svg>
            </div>
            {sidebarOpen && <span className="font-semibold text-sm text-gray-900">Faculty AI</span>}
          </div>
          <button onClick={() => setSidebarOpen((p) => !p)}
            className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 transition-colors">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>

        <button onClick={createNewChat}
          className={`flex items-center gap-2 px-2 py-2 rounded-lg text-sm font-medium hover:bg-gray-100 text-gray-700 transition-colors w-full ${!sidebarOpen ? "justify-center" : ""}`}>
          <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          {sidebarOpen && <span>New chat</span>}
        </button>

        <div className="flex-1 overflow-y-auto mt-3">
          {sidebarOpen && chats.length > 0 && (
            <>
              <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide px-2 mb-1.5">Recent</p>
              <div className="space-y-0.5">
                {chats.map((chat) => (
                  <button key={chat.id} onClick={() => setCurrentChatId(chat.id)}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm truncate transition-colors
                      ${chat.id === currentChatId ? "bg-gray-100 font-medium text-gray-900" : "text-gray-600 hover:bg-gray-50"}`}>
                    {chat.title}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>

        <div className="border-t border-gray-200 pt-3">
          <div className={`flex items-center gap-2.5 px-2 py-2 rounded-lg hover:bg-gray-100 cursor-pointer ${!sidebarOpen ? "justify-center" : ""}`}>
            <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center text-xs font-bold flex-shrink-0">AK</div>
            {sidebarOpen && (
              <div>
                <p className="text-sm font-medium text-gray-800 leading-tight">Amira Khezzar</p>
                <p className="text-[11px] text-gray-400">Faculty Member</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Main area ── */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* Header */}
        <div className="flex items-center justify-center py-3 border-b border-gray-200 bg-white">
          <img src="/uos_logo.png" alt="University of Sharjah" className="h-10 opacity-90" />
        </div>

        {/* Error banner */}
        {error && (
          <div className="bg-red-50 border-b border-red-200 px-4 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Messages */}
        {!currentChat || currentChat.messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center px-4">
            <div className="w-14 h-14 bg-gray-900 rounded-2xl flex items-center justify-center">
              <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-3 3v-3z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-800">Faculty Affairs Assistant</h1>
              <p className="text-sm text-gray-500 mt-1">Ask anything about UoS faculty policies and promotion guidelines.</p>
            </div>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto space-y-6 py-6 px-4">
              {currentChat.messages.map((msg, i) => (
                <div key={i} className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>

                  {/* Avatar */}
                  <div className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-xs font-semibold mt-0.5
                    ${msg.role === "user" ? "bg-green-500 text-white" : "bg-gray-800 text-white"}`}>
                    {msg.role === "user" ? "AK" : "AI"}
                  </div>

                  {/* Bubble */}
                  <div className={`max-w-xl flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}>
                    <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap
                      ${msg.role === "user"
                        ? "bg-green-500 text-white rounded-tr-sm"
                        : "bg-white border border-gray-200 shadow-sm text-gray-800 rounded-tl-sm"}`}>
                      {msg.loading ? (
                        <div className="flex items-center gap-1 py-1">
                          {[0, 1, 2].map((i) => (
                            <span key={i} className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"
                              style={{ animationDelay: `${i * 0.15}s` }} />
                          ))}
                        </div>
                      ) : msg.content}
                    </div>

                    {/* Sources */}
                    {!msg.loading && msg.role === "assistant" && msg.sources?.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {msg.sources.map((s, idx) => (
                          <span key={idx} className="inline-flex items-center gap-1 bg-gray-100 text-gray-600 px-2 py-0.5 rounded-md text-[11px] font-medium">
                            <svg className="w-3 h-3 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414A1 1 0 0119 9.414V19a2 2 0 01-2 2z" />
                            </svg>
                            {s.file_name}{s.page ? ` · p.${s.page}` : ""}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* RAGAS evaluation */}
                    {!msg.loading && msg.role === "assistant" && msg.evaluation && (
                      <div className="w-full max-w-xl">
                        <EvaluationPanel evaluation={msg.evaluation} groundTruthUsed={msg.groundTruthUsed ?? false} />
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
          </div>
        )}

        {/* Input */}
        <div className="p-4 bg-white border-t border-gray-200">
          <div className="max-w-3xl mx-auto flex items-end gap-3 bg-white border border-gray-300 rounded-2xl px-4 py-3 shadow-sm focus-within:border-gray-400 transition-colors">
            <textarea
              ref={textareaRef}
              className="flex-1 outline-none resize-none text-sm text-gray-800 placeholder-gray-400 leading-relaxed max-h-32"
              rows={1}
              placeholder="Ask about faculty policies, promotion criteria…"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 128)}px`;
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
              }}
              disabled={loading}
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className={`flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-all
                ${loading || !input.trim() ? "bg-gray-200 text-gray-400 cursor-not-allowed" : "bg-gray-900 text-white hover:bg-gray-700"}`}>
              {loading ? (
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              )}
            </button>
          </div>
          <p className="text-center text-[11px] text-gray-400 mt-2">Answers are grounded in UoS promotion documents only.</p>
        </div>
      </div>
    </div>
  );
}
