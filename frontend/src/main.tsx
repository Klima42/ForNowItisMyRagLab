import { StrictMode, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { FileText, Search, UploadCloud, Gauge, ChevronRight } from 'lucide-react';
import './styles.css';

type Result = { chunk_id: string; text: string; source: string; section?: string; score: number };
type Citation = { citation_id: string; source: string; page?: number; section?: string; quote: string };
type Health = { status: string; documents: number; chunks: number };
type Metrics = { cases: number; recall_at_k: number; precision_at_k: number; mrr: number; average_latency_ms: number; mode: string };

function App() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('hybrid');
  const [rerank, setRerank] = useState(false);
  const [rewrite, setRewrite] = useState(false);
  const [documentType, setDocumentType] = useState('');
  const [year, setYear] = useState('');
  const [results, setResults] = useState<Result[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [health, setHealth] = useState<Health>({ status: 'connecting', documents: 0, chunks: 0 });
  const [answer, setAnswer] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [metrics, setMetrics] = useState<Metrics | null>(null);

  useEffect(() => { fetch('/api/health').then((response) => response.json()).then(setHealth).catch(() => setHealth((current) => ({ ...current, status: 'offline' }))); }, [message]);

  async function ask() {
    if (!query.trim()) return;
    setBusy(true); setMessage('');
    const response = await fetch('/api/answer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, mode, rerank, rewrite, top_k: 5, document_type: documentType || null, year: year ? Number(year) : null }) });
    const data = await response.json(); setAnswer(data.answer + (data.rewritten_query ? `\n\nRewritten query: ${data.rewritten_query}` : '')); setResults(data.retrieved_chunks); setCitations(data.citations); setBusy(false);
  }

  async function upload(fileList: FileList | null) {
    if (!fileList?.length) return;
    const form = new FormData(); Array.from(fileList).forEach((file) => form.append('files', file));
    const response = await fetch('/api/documents', { method: 'POST', body: form });
    const data = await response.json(); setMessage(`${data.ingested.length} document(s) indexed`);
  }

  async function evaluate() {
    const response = await fetch('/api/evaluate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode, top_k: 5, rerank, rewrite, cases: [{ question: query || 'authentication token', relevant_sources: results.length ? [results[0].source] : ['__no_expected_source__'] }] }) });
    if (response.ok) setMetrics(await response.json());
  }

  return <main className="shell">
    <aside className="rail"><div className="mark">A</div><div className="rail-icon active"><Search size={19} /></div><div className="rail-icon"><FileText size={19} /></div><div className="rail-icon"><Gauge size={19} /></div></aside>
    <section className="workspace">
      <header className="topbar"><div><span className="kicker">ATLAS / RAG LAB</span><h1>Knowledge, with receipts.</h1></div><div className="status"><i className={health.status === 'ok' ? 'online' : ''} /> {health.status === 'ok' ? 'Local engine ready' : health.status}</div></header>
      <div className="content">
        <section className="hero"><p className="eyebrow">Document intelligence</p><h2>Ask your corpus<br /><em>anything.</em></h2><p className="subtext">Search across your indexed knowledge base with hybrid retrieval and traceable evidence.</p></section>
        <section className="query-panel"><div className="query-row"><Search size={20} /><input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => event.key === 'Enter' && ask()} placeholder="What would you like to know?" /><button onClick={ask} disabled={busy}>{busy ? 'Searching...' : 'Ask Atlas'} <ChevronRight size={16} /></button></div><div className="controls"><span>RETRIEVAL STRATEGY</span>{['vector', 'hybrid', 'lexical'].map((option) => <button className={mode === option ? 'selected' : ''} onClick={() => setMode(option)} key={option}>{option}</button>)}<label className="toggle"><input type="checkbox" checked={rerank} onChange={(event) => setRerank(event.target.checked)} /> rerank</label><label className="toggle"><input type="checkbox" checked={rewrite} onChange={(event) => setRewrite(event.target.checked)} /> rewrite</label><label className="filter">TYPE<select value={documentType} onChange={(event) => setDocumentType(event.target.value)}><option value="">all</option><option value="pdf">pdf</option><option value="docx">docx</option><option value="md">md</option><option value="txt">txt</option></select></label><label className="filter">YEAR<input type="number" min="2000" max="2100" placeholder="all" value={year} onChange={(event) => setYear(event.target.value)} /></label><span className="hint">Press Enter to search</span></div></section>
        {answer && <section className="answer"><div className="answer-label">ANSWER <span>{mode} retrieval</span></div><p>{answer}</p>{citations.length > 0 && <div className="citations"><div className="citation-title">CITATIONS <span>{citations.length} sources</span></div>{citations.map((citation) => <div className="citation" key={citation.citation_id}><b>{citation.citation_id}</b><div><strong>{citation.source}</strong><small>{citation.section || 'Indexed passage'}{citation.page ? ` · page ${citation.page}` : ''}</small><q>{citation.quote}</q></div></div>)}</div>}</section>}
        <section className="lower"><div className="sources"><div className="section-heading"><h3>Evidence trail</h3><span>{results.length} results</span></div>{results.length ? results.map((result) => <article className="result" key={result.chunk_id}><div className="result-top"><span><FileText size={15} /> {result.source}</span><b>{Math.round(result.score * 100)}%</b></div><p>{result.text}</p><small>{result.section || 'Indexed passage'}</small></article>) : <div className="empty">Your retrieved passages will appear here.</div>}</div><div className="index-card"><div className="section-heading"><h3>Corpus</h3><span className="live">LIVE</span></div><div className="metric"><strong>{health.documents}</strong><span>documents indexed</span></div><div className="metric"><strong>{health.chunks}</strong><span>passages available</span></div><label className="dropzone"><UploadCloud size={22} /><span>Drop files to index</span><small>PDF, DOCX, MD, TXT</small><input type="file" multiple accept=".pdf,.docx,.md,.txt" onChange={(event) => upload(event.target.files)} /></label>{message && <p className="upload-message">{message}</p>}<button className="benchmark" onClick={evaluate}>Run benchmark</button>{metrics && <div className="metrics"><div><strong>{Math.round(metrics.recall_at_k * 100)}%</strong><span>recall@5</span></div><div><strong>{Math.round(metrics.mrr * 100)}%</strong><span>MRR</span></div><div><strong>{metrics.average_latency_ms}ms</strong><span>latency</span></div></div>}</div></section>
      </div>
    </section>
  </main>;
}

export default function Root() { return <StrictMode><App /></StrictMode>; }

createRoot(document.getElementById('root')!).render(<Root />);
