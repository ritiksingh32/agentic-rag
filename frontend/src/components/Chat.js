'use client';

import { useState, useRef, useEffect } from 'react';
import { sendChatMessage } from '@/lib/api';

const TOOL_LABELS = {
  retrieve_documents: 'Searching your documents',
  web_search: 'Searching the web',
  compare_documents: 'Comparing documents',
  list_my_documents: 'Listing your documents',
};

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function handleSend() {
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput('');
    setError('');

    const conversationHistory = messages.map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, { role: 'user', content: question }]);
    setLoading(true);

    try {
      const result = await sendChatMessage(question, conversationHistory);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: result.answer,
          reasoningTrace: result.reasoning_trace,
          hitLimit: result.hit_iteration_limit,
        },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 flex flex-col h-[600px]">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <p className="text-sm text-gray-700">
            Ask a question — the agent will decide whether to search your documents, search the web, or both.
          </p>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={msg.role === 'user' ? 'text-right' : 'text-left'}>
            {msg.role === 'assistant' && msg.reasoningTrace?.length > 0 && (
              <div className="mb-1 flex flex-wrap gap-1">
                {msg.reasoningTrace.map((step, j) => (
                  <span
                    key={j}
                    className="text-xs bg-purple-50 text-purple-700 px-2 py-1 rounded-full border border-purple-200"
                  >
                    🔧 {TOOL_LABELS[step.tool] || step.tool}
                  </span>
                ))}
              </div>
            )}

            <div
              className={`inline-block max-w-[80%] px-3 py-2 rounded-lg text-sm whitespace-pre-wrap ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              {msg.content}
            </div>

            {msg.hitLimit && (
              <p className="text-xs text-amber-600 mt-1">
                (Reached the max tool-call limit — answered with what was found so far.)
              </p>
            )}
          </div>
        ))}

        {loading && (
          <div className="text-left">
            <div className="inline-block bg-gray-100 text-gray-500 px-3 py-2 rounded-lg text-sm">
              Thinking...
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {error && <p className="text-sm text-red-600 px-4">{error}</p>}

      <div className="border-t border-gray-200 p-3 flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask something..."
          rows={1}
          className="flex-1 resize-none border border-gray-300 rounded-md px-3 py-2 text-sm text-[#000000] focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          onClick={handleSend}
          disabled={loading}
          className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700 disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
}