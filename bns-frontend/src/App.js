import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null); // Ref for auto-scrolling
  const [showIntro, setShowIntro] = useState(true);

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
      setMessages([
        ...newMessages,
        {
          role: "bot",
          type: "result",
          data: data
        }
      ]);
    } catch (err) {
      setMessages([...newMessages, { role: "bot", text: "Error: Is your backend running?" }]);
    }
    setLoading(false);
  };

  return (
    <>
      {showIntro && (
        <div className="intro-overlay">
          <div className="intro-modal">
            
            <h2>👋 Welcome to RAGLegal BNS Assistant</h2>
            
            <p>
              This project uses <b>Retrieval-Augmented Generation (RAG)</b> to analyze
              legal scenarios based on the <b>Bharatiya Nyaya Sanhita (2023)</b>.
            </p>
      
            <p>
              🔍 It retrieves relevant legal sections <br />
              🤖 Uses AI to explain applicable laws <br />
              ⚖️ Helps you understand legal outcomes
            </p>
      
            <button onClick={() => setShowIntro(false)}>
              Try Now 🚀
            </button>
      
          </div>
        </div>
      )}

      <div className="rag-app">
        {/* 1. App Header */}
        <header className="rag-header">
          <div className="rag-title">⚖️ RAGLegal BNS Assistant</div>
          <div className="rag-status">Bharatiya Nyaya Sanhita (2023) • AI Grounded in Law</div>
        </header>

        {/* 2. Chat Window */}
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
                <div className="rag-markdown">
                  {m.type === "result" ? (
                    <div>

                      <h4>📚 Retrieved Legal Context</h4>
                      {m.data.retrieved_context.map((item, i) => (
                        <div key={i} style={{ marginBottom: "10px" }}>
                          <b>Section {item.section}</b>
                          <div>Score: {item.score.toFixed(3)}</div>
                          <div>{item.text}</div>
                        </div>
                      ))}

                      <h4>🤖 Analysis</h4>
                      <pre>{m.data.analysis.raw}</pre>

                    </div>
                  ) : (
                    <ReactMarkdown>{m.text}</ReactMarkdown>
                  )}
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

          <div ref={messagesEndRef} />
        </main>

        {/* 3. Input Area */}
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
    </>
  );
}

export default App;