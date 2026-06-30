import { supabase } from './supabaseClient';

const API_URL = process.env.NEXT_PUBLIC_API_URL;

/**
 * Makes an authenticated request to the backend, automatically
 * attaching the current Supabase session's access token.
 */
async function apiFetch(path, options = {}) {
  const { data: { session } } = await supabase.auth.getSession();

  if (!session) {
    throw new Error('Not logged in.');
  }

  const headers = {
    ...options.headers,
    Authorization: `Bearer ${session.access_token}`,
  };

  const response = await fetch(`${API_URL}${path}`, { ...options, headers });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Request failed: ${response.status}`);
  }

  return response.json();
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);

  const { data: { session } } = await supabase.auth.getSession();
  if (!session) throw new Error('Not logged in.');

  const response = await fetch(`${API_URL}/documents/upload`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${session.access_token}` },
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Upload failed: ${response.status}`);
  }

  return response.json();
}

export async function listDocuments() {
  return apiFetch('/documents');
}

export async function deleteDocument(docId) {
  return apiFetch(`/documents/${docId}`, { method: 'DELETE' });
}

export async function sendChatMessage(question, conversationHistory = []) {
  return apiFetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, conversation_history: conversationHistory }),
  });
}