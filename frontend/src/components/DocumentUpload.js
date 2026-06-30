'use client';

import { useState, useEffect, useCallback } from 'react';
import { uploadDocument, listDocuments, deleteDocument } from '@/lib/api';

export default function DocumentUpload() {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');

  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const docs = await listDocuments();
        if (!cancelled) setDocuments(docs);
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  async function handleFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setError('');

    try {
      await uploadDocument(file);
      await refreshDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      e.target.value = ''; // reset the file input
    }
  }

  async function handleDelete(docId) {
    try {
      await deleteDocument(docId);
      await refreshDocuments();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <h2 className="font-semibold mb-3 text-[#000000]">Your Documents</h2>

      <label className="block">
        <span className="sr-only">Upload document</span>
        <input
          type="file"
          accept=".pdf,.docx"
          onChange={handleFileChange}
          disabled={uploading}
          className="block w-full text-sm text-gray-600 mb-3 file:mr-3 file:py-2 file:px-3 file:rounded-md file:border-0 file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
      </label>

      {uploading && <p className="text-sm text-blue-600 mb-2">Uploading and processing...</p>}
      {error && <p className="text-sm text-red-600 mb-2">{error}</p>}

      <ul className="space-y-2">
        {documents.length === 0 && (
          <li className="text-sm text-gray-400">No documents uploaded yet.</li>
        )}
        {documents.map((doc) => (
          <li
            key={doc.doc_id}
            className="flex justify-between items-center text-sm bg-gray-50 px-3 py-2 rounded"
          >
            <span className="truncate text-[#000000]">
              {doc.filename}{' '}
              <span className="text-gray-400">({doc.status})</span>
            </span>
            <button
              onClick={() => handleDelete(doc.doc_id)}
              className="text-red-500 hover:underline ml-2 shrink-0"
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}