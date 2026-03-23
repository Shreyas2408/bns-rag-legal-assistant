import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null); // Ref for auto-scrolling

  // Auto-scroll to the bottom when a new message arrives
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { role: "user", text: input }];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: input }),
      });
      const data = await res.json();
      setMessages([...newMessages, { role: "bot", text: data.response }]);
    } catch (err) {
      setMessages([...newMessages, { role: "bot", text: "Error: Is your backend running?" }]);
    }
    setLoading(false);
  };

  return (
    <div className="rag-app">
      {/* 1. App Header */}
      <header className="rag-header">
        <div className="rag-title">⚖️ RAGLegal BNS Assistant</div>
        <div className="rag-status">Bharatiya Nyaya Sanhita (2023) • AI Grounded in Law</div>
      </header>

      {/* 2. Chat Window (Auto-scrolling) */}
      <main className="rag-chat-window">
        {messages.length === 0 && (
          <div className="rag-welcome">
            <h3>Ready to assist with BNS Legal Queries.</h3>
            <p>Example: "What is the penalty for medical negligence?"</p>
          </div>
        )}
        
        {messages.map((m, i) => (
          <div key={i} className={`rag-message-row ${m.role}`}>
            <div className="rag-avatar">{m.role === "user" ? "👤" : "🏛️"}</div>
            <div className="rag-message-bubble">
              {/* ReactMarkdown renders bolding, lists, and newlines correctly */}
              <div className="rag-markdown">
                <ReactMarkdown>{m.text}</ReactMarkdown>
              </div>
            </div>
          </div>
        ))}
        {loading && (
          <div className="rag-message-row bot loading">
            <div className="rag-avatar">🏛️</div>
            <div className="rag-message-bubble">
              <span className="dot-flashing"></span>
            </div>
          </div>
        )}
        {/* Dummy element for auto-scroll to focus on */}
        <div ref={messagesEndRef} />
      </main>

      {/* 3. Input Area (Fixed at the Bottom) */}
      <footer className="rag-input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter a legal scenario for analysis..."
          rows={input.split('\n').length > 3 ? 4 : input.split('\n').length || 1}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
        />
        <button className="rag-send-btn" onClick={handleSend} disabled={loading}>
          {loading ? "..." : "Send"}
        </button>
      </footer>
    </div>
  );
}

export default App;