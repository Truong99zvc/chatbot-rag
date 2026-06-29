import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, 
  Send, 
  Trash2, 
  BookOpen, 
  Compass, 
  CheckCircle, 
  HelpCircle, 
  Sparkles, 
  Layers,
  ChevronDown,
  ChevronUp,
  FileText
} from 'lucide-react';
import confetti from 'canvas-confetti';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: string;
}

interface ChatSession {
  session_id: string;
  name: string;
}

interface AgentStep {
  name: string;
  status: 'idle' | 'active' | 'completed';
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'lookup'>('chat');
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>('default');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [expandedSourcesIdx, setExpandedSourcesIdx] = useState<number | null>(null);

  // Article Lookup state
  const [lookupArticle, setLookupArticle] = useState<string>('');
  const [lookupResult, setLookupResult] = useState<{ article: string; content: string; sources: string } | null>(null);
  const [lookupLoading, setLookupLoading] = useState<boolean>(false);
  const [lookupError, setLookupError] = useState<string>('');

  // Agent steps tracing
  const [steps, setSteps] = useState<AgentStep[]>([
    { name: 'Phân loại ý định (Router)', status: 'idle' },
    { name: 'Viết lại câu hỏi (Rewriter)', status: 'idle' },
    { name: 'Truy vấn lai (Qdrant + BM25)', status: 'idle' },
    { name: 'Kiểm định tài liệu (Doc Grader)', status: 'idle' },
    { name: 'Sinh phản hồi & Chống ảo giác', status: 'idle' }
  ]);

  const messageEndRef = useRef<HTMLDivElement>(null);

  // Load sessions on startup
  useEffect(() => {
    // Default session list
    const saved = localStorage.getItem('uit_chat_sessions');
    if (saved) {
      const parsed = JSON.parse(saved);
      setSessions(parsed);
      if (parsed.length > 0) {
        setActiveSessionId(parsed[0].session_id);
      }
    } else {
      const initial = [{ session_id: 'default', name: 'Phiên tư vấn mặc định' }];
      setSessions(initial);
      localStorage.setItem('uit_chat_sessions', JSON.stringify(initial));
    }
  }, []);

  // Fetch history when active session changes
  useEffect(() => {
    if (activeSessionId) {
      fetchSessionHistory(activeSessionId);
    }
  }, [activeSessionId]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const fetchSessionHistory = async (sessionId: string) => {
    try {
      const res = await fetch(`/api/v1/rag/sessions/${sessionId}`);
      if (res.ok) {
        const data = await res.json();
        const history: Message[] = [];
        data.turns.forEach((turn: any) => {
          history.push({ role: 'user', content: turn.question });
          history.push({ role: 'assistant', content: turn.answer, sources: turn.sources || undefined });
        });
        setMessages(history);
      } else {
        setMessages([]);
      }
    } catch (e) {
      console.error('Failed to fetch session history:', e);
      setMessages([]);
    }
  };

  const handleCreateSession = () => {
    const newId = 'session_' + Date.now();
    const newSession = {
      session_id: newId,
      name: `Phiên thảo luận #${sessions.length + 1}`
    };
    const updated = [newSession, ...sessions];
    setSessions(updated);
    localStorage.setItem('uit_chat_sessions', JSON.stringify(updated));
    setActiveSessionId(newId);
  };

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    // Call delete API
    try {
      await fetch(`/api/v1/rag/sessions/${sessionId}`, { method: 'DELETE' });
    } catch (err) {
      console.error('Failed to delete session history on server:', err);
    }

    const filtered = sessions.filter(s => s.session_id !== sessionId);
    setSessions(filtered);
    localStorage.setItem('uit_chat_sessions', JSON.stringify(filtered));

    if (activeSessionId === sessionId) {
      if (filtered.length > 0) {
        setActiveSessionId(filtered[0].session_id);
      } else {
        const fallback = { session_id: 'default', name: 'Phiên tư vấn mặc định' };
        setSessions([fallback]);
        localStorage.setItem('uit_chat_sessions', JSON.stringify([fallback]));
        setActiveSessionId('default');
      }
    }
  };

  const handleClearAllSessions = async () => {
    if (!confirm('Bạn có chắc chắn muốn xóa toàn bộ lịch sử tư vấn?')) return;
    
    // Delete all from server
    for (const session of sessions) {
      try {
        await fetch(`/api/v1/rag/sessions/${session.session_id}`, { method: 'DELETE' });
      } catch (err) {
        console.error(err);
      }
    }

    const fallback = { session_id: 'default', name: 'Phiên tư vấn mặc định' };
    setSessions([fallback]);
    localStorage.setItem('uit_chat_sessions', JSON.stringify([fallback]));
    setActiveSessionId('default');
  };

  // Simulate step-by-step progress logging of Agent nodes
  const runAgentStepsAnimation = () => {
    setSteps([
      { name: 'Phân loại ý định (Router)', status: 'active' },
      { name: 'Viết lại câu hỏi (Rewriter)', status: 'idle' },
      { name: 'Truy vấn lai (Qdrant + BM25)', status: 'idle' },
      { name: 'Kiểm định tài liệu (Doc Grader)', status: 'idle' },
      { name: 'Sinh phản hồi & Chống ảo giác', status: 'idle' }
    ]);

    // Router done, Rewriter active
    setTimeout(() => {
      setSteps(prev => [
        { ...prev[0], status: 'completed' },
        { ...prev[1], status: 'active' },
        ...prev.slice(2)
      ]);
    }, 1200);

    // Rewriter done, Qdrant search active
    setTimeout(() => {
      setSteps(prev => [
        prev[0],
        { ...prev[1], status: 'completed' },
        { ...prev[2], status: 'active' },
        ...prev.slice(3)
      ]);
    }, 2400);

    // Qdrant done, Grader active
    setTimeout(() => {
      setSteps(prev => [
        prev[0],
        prev[1],
        { ...prev[2], status: 'completed' },
        { ...prev[3], status: 'active' },
        prev[4]
      ]);
    }, 3800);

    // Grader done, LLM generation + Hallucination check active
    setTimeout(() => {
      setSteps(prev => [
        prev[0],
        prev[1],
        prev[2],
        { ...prev[3], status: 'completed' },
        { ...prev[4], status: 'active' }
      ]);
    }, 5000);
  };

  const handleSend = async (textToSend?: string) => {
    const queryText = textToSend || input;
    if (!queryText.trim() || loading) return;

    setInput('');
    setLoading(true);
    setExpandedSourcesIdx(null);
    setMessages(prev => [...prev, { role: 'user', content: queryText }]);

    // Trigger step indicators
    runAgentStepsAnimation();

    try {
      const res = await fetch('/api/v1/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: queryText, session_id: activeSessionId })
      });

      if (res.ok) {
        const data = await res.json();
        // Update steps to all completed
        setSteps(prev => prev.map(s => ({ ...s, status: 'completed' })));
        
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: data.answer, 
          sources: data.sources || undefined 
        }]);

        // Throw micro-confetti on success
        confetti({
          particleCount: 40,
          spread: 60,
          origin: { y: 0.8 },
          colors: ['#8b5cf6', '#6366f1', '#ec4899']
        });
      } else {
        const errData = await res.json();
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `⚠️ Có lỗi xảy ra: ${errData.detail || 'Không thể xử lý phản hồi'}. Vui lòng thử lại.` 
        }]);
      }
    } catch (e) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: '⚠️ Lỗi kết nối mạng đến Server. Vui lòng kiểm tra lại trạng thái Backend.' 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleLookup = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!lookupArticle.trim() || lookupLoading) return;

    setLookupLoading(true);
    setLookupError('');
    setLookupResult(null);

    try {
      const res = await fetch(`/api/v1/rag/search?article=${encodeURIComponent(lookupArticle.trim())}`);
      if (res.ok) {
        const data = await res.json();
        setLookupResult(data);
      } else {
        const err = await res.json();
        setLookupError(err.detail || 'Không thể tìm thấy Điều luật tương ứng.');
      }
    } catch (err) {
      setLookupError('Không thể kết nối đến máy chủ.');
    } finally {
      setLookupLoading(false);
    }
  };

  const sampleQuestions = [
    { text: 'Điều kiện xét tốt nghiệp đại học là gì?', desc: 'Tra cứu điều kiện cần và đủ để nhận bằng cử nhân' },
    { text: 'Cảnh cáo học vụ và buộc thôi học', desc: 'Quy định về học lực yếu, điểm trung bình tối thiểu' },
    { text: 'Quy trình đăng ký học vượt thế nào?', desc: 'Hướng dẫn rút ngắn thời gian đào tạo' },
    { text: 'Quy định về thi lại và cải thiện điểm số', desc: 'Thang điểm thi và quy tắc làm tròn điểm' }
  ];

  // Helper to parse source markdown formatted bullet strings into lists
  const renderCitationsList = (sourcesString: string) => {
    return sourcesString.split('\n').filter(s => s.trim()).map((source, i) => {
      // clean markdown asterisks
      const cleaned = source.replace(/[\*\-]/g, '').trim();
      return (
        <div key={i} className="citation-tag">
          <FileText size={13} />
          <span>{cleaned}</span>
        </div>
      );
    });
  };

  return (
    <div className="app-container">
      {/* Sidebar Panel */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-badge">
            <Sparkles size={20} className="text-white" />
          </div>
          <h1>UIT Academic RAG</h1>
        </div>

        <div className="sidebar-actions">
          <button className="btn-new-chat" onClick={handleCreateSession}>
            <MessageSquare size={16} />
            <span>Tạo cuộc hội thoại mới</span>
          </button>
        </div>

        <div className="session-list">
          {sessions.map(session => (
            <div 
              key={session.session_id} 
              className={`session-item ${session.session_id === activeSessionId ? 'active' : ''}`}
              onClick={() => setActiveSessionId(session.session_id)}
            >
              <div className="session-info">
                <MessageSquare size={16} />
                <span className="session-title">{session.name}</span>
              </div>
              <button 
                className="btn-delete-session"
                onClick={(e) => handleDeleteSession(session.session_id, e)}
                title="Xóa phiên này"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <button className="btn-clear-all" onClick={handleClearAllSessions}>
            <Trash2 size={14} />
            <span>Xóa sạch toàn bộ lịch sử</span>
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <header className="app-header">
          <div className="header-title-container">
            <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
              {activeTab === 'chat' ? 'Advisor Chatbot' : 'Tra cứu văn bản'}
            </span>
          </div>

          <nav className="nav-tabs">
            <button 
              className={`tab-button ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              <MessageSquare size={16} />
              <span>Tư vấn tự động</span>
            </button>
            <button 
              className={`tab-button ${activeTab === 'lookup' ? 'active' : ''}`}
              onClick={() => setActiveTab('lookup')}
            >
              <BookOpen size={16} />
              <span>Tra cứu số Điều</span>
            </button>
          </nav>
        </header>

        {activeTab === 'chat' ? (
          <div className="chat-area">
            <div className="message-list">
              {messages.length === 0 ? (
                <div className="welcome-screen">
                  <h2>Chào mừng bạn đến với UIT Advisor AI</h2>
                  <p className="welcome-subtitle">
                    Hệ thống Tác tử (Agentic RAG) hỗ trợ trả lời chuẩn quy chế đào tạo đại học chính quy của UIT.
                  </p>
                  
                  <div className="suggestions-grid">
                    {sampleQuestions.map((q, idx) => (
                      <div 
                        key={idx} 
                        className="suggestion-card"
                        onClick={() => handleSend(q.text)}
                      >
                        <Compass size={20} className="suggestion-icon" />
                        <h3>{q.text}</h3>
                        <p>{q.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                messages.map((msg, index) => (
                  <div key={index} className={`message-wrapper ${msg.role}`}>
                    <div className="message-avatar">
                      <HelpCircle size={20} className="text-white" />
                    </div>
                    <div className="message-bubble">
                      <div className="message-text" style={{ whiteSpace: 'pre-wrap' }}>
                        {msg.content}
                      </div>

                      {msg.sources && (
                        <div className="citations-container">
                          <div 
                            className="citations-header"
                            onClick={() => setExpandedSourcesIdx(expandedSourcesIdx === index ? null : index)}
                          >
                            <span>Tài liệu trích dẫn</span>
                            {expandedSourcesIdx === index ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                          </div>
                          {expandedSourcesIdx === index && (
                            <div className="citation-list">
                              {renderCitationsList(msg.sources)}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}

              {/* Agentic workflow loading tracing */}
              {loading && (
                <div className="agentic-status-container">
                  <div className="agentic-header">
                    <div className="pulse-dot"></div>
                    <span className="agentic-header-text">Luồng suy nghĩ của tác tử học vụ</span>
                    <div className="typing-dots">
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                    </div>
                  </div>
                  <div className="agentic-steps">
                    {steps.map((step, idx) => (
                      <div 
                        key={idx} 
                        className={`agentic-pill ${step.status === 'active' ? 'active' : step.status === 'completed' ? 'completed' : ''}`}
                      >
                        {step.status === 'completed' && <CheckCircle size={12} />}
                        {step.status === 'active' && <Layers size={12} className="animate-spin" />}
                        <span>{step.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              <div ref={messageEndRef} />
            </div>

            {/* Chat Input Field */}
            <div className="input-area">
              <div className="input-container-wrapper">
                <input 
                  type="text" 
                  className="chat-input"
                  placeholder="Nhập câu hỏi học vụ của bạn tại đây... (Ví dụ: điều kiện tốt nghiệp)"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  disabled={loading}
                />
                <button 
                  className="btn-send"
                  onClick={() => handleSend()}
                  disabled={loading || !input.trim()}
                >
                  <Send size={18} />
                </button>
              </div>
              <p className="input-disclaimer">
                Hệ thống tự động sử dụng LangGraph và Qdrant để lọc ảo giác. Nội dung được trích xuất từ quy chế chính thức ngày 15/03/2019.
              </p>
            </div>
          </div>
        ) : (
          /* Article Direct Lookup Tab */
          <div className="article-lookup-view">
            <div className="lookup-card">
              <h2>Tra cứu số Điều trực tiếp</h2>
              <p>
                Nếu bạn đã biết chính xác số Điều cần xem trong quy chế đào tạo của UIT (ví dụ: Điều 8, Điều 15), hãy nhập số Điều dưới đây để lấy trực tiếp nội dung đầy đủ từ cơ sở dữ liệu.
              </p>

              <form onSubmit={handleLookup} className="lookup-form">
                <input 
                  type="text" 
                  className="lookup-input"
                  placeholder="Nhập số điều (ví dụ: 15, 23)"
                  value={lookupArticle}
                  onChange={(e) => setLookupArticle(e.target.value)}
                />
                <button type="submit" className="btn-lookup" disabled={lookupLoading}>
                  {lookupLoading ? 'Đang tìm...' : 'Tra cứu'}
                </button>
              </form>

              {lookupError && (
                <div style={{ color: '#f87171', fontSize: '0.9rem', marginTop: '8px' }}>
                  {lookupError}
                </div>
              )}
            </div>

            {lookupResult && (
              <div className="lookup-result">
                <div className="markdown-body">
                  <h2>Điều {lookupResult.article}</h2>
                  <div style={{ whiteSpace: 'pre-wrap', fontSize: '0.95rem' }}>
                    {lookupResult.content}
                  </div>
                  {lookupResult.sources && (
                    <div style={{ marginTop: '16px', borderTop: '1px dashed rgba(255,255,255,0.08)', paddingTop: '12px' }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>TÀI LIỆU TRÍCH DẪN</span>
                      <div className="citation-list" style={{ marginTop: '8px' }}>
                        {renderCitationsList(lookupResult.sources)}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
