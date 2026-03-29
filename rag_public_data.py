#!/usr/bin/env python3
"""Real paper RAG data provider for the public dashboard.

This script reads the active paper corpus from ChromaDB and exposes
JSON endpoints for stats, search, browse, and LLM answers.

Data sources:
- /root/.openclaw/workspace/.vector_db/papers
  - collection: papers
  - collection: papers_summary

The search strategy is intentionally simple and reliable:
- keyword scoring over titles, authors, summaries, and paper chunks
- real corpus data only (no synthetic fixtures)
- LLM answers via OpenRouter when an API key is available
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import chromadb
from requests.exceptions import RequestsDependencyWarning

warnings.filterwarnings('ignore', category=RequestsDependencyWarning)
import requests

WORKSPACE = Path('/root/.openclaw/workspace')
PAPERS_DB = WORKSPACE / '.vector_db' / 'papers'
PAPERS_COLLECTION = 'papers'
SUMMARIES_COLLECTION = 'papers_summary'
OBSIDIAN_PAPERS_DIR = Path('/data/obsidian/3. Resources/Papers')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'stepfun/step-3.5-flash:free')
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
TERM_EXPANSIONS = {
    'bouger': ['bouger', 'bouguer'],
    'bouguer': ['bouguer', 'bouger'],
    'gravimetri': ['gravimetri', 'gravity', 'gravity anomaly'],
    'gravity': ['gravity', 'gravimetri', 'gravity anomaly'],
    'anomaly': ['anomaly', 'anomali'],
    'simamora': ['simamora'],
}


def normalize_key(value: str | None) -> str:
    text = (value or '').strip().lower()
    text = re.sub(r'\.[a-z0-9]+$', '', text)
    text = text.replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def trim(text: str | None, limit: int = 320) -> str:
    if not text:
        return ''
    clean = ' '.join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + '…'


def safe_json_load(text: str, fallback: Any = None) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return fallback


SECTION_HEADINGS = ('abstract', 'method', 'results', 'limitations', 'next steps', 'references', 'keywords')
TITLE_NOISE_RE = re.compile(r'\s*(?:\(\d+\)\s*)+$')


def normalize_title_text(text: str | None) -> str:
    clean = ' '.join(str(text or '').replace('\u200b', ' ').split())
    clean = clean.strip(' -–—:;')
    clean = TITLE_NOISE_RE.sub('', clean).strip()
    return clean


def parse_author_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [normalize_title_text(item) for item in raw if normalize_title_text(item)]
    if isinstance(raw, str):
        loaded = safe_json_load(raw, None)
        if isinstance(loaded, list):
            return parse_author_list(loaded)
        if not raw.strip():
            return []
        parts = [part.strip() for part in re.split(r'\s*;\s*|\s+\&\s+|\s+and\s+', raw) if part.strip()]
        if len(parts) > 1:
            return [normalize_title_text(part) for part in parts if normalize_title_text(part)]
        return [normalize_title_text(raw)] if normalize_title_text(raw) else []
    text = normalize_title_text(str(raw))
    return [text] if text else []


def format_apa_author(name: str) -> str:
    clean = normalize_title_text(name)
    if not clean:
        return ''
    if ',' in clean:
        return clean
    parts = [part for part in clean.split(' ') if part]
    if len(parts) == 1:
        return parts[0]
    surname = parts[-1]
    initials = ' '.join(f'{part[0].upper()}.' for part in parts[:-1] if part)
    return f'{surname}, {initials}'


def format_apa_authors(authors: Any) -> str:
    parsed = [format_apa_author(a) for a in parse_author_list(authors)]
    parsed = [item for item in parsed if item]
    if not parsed:
        return ''
    if len(parsed) == 1:
        return parsed[0]
    if len(parsed) == 2:
        return f'{parsed[0]} & {parsed[1]}'
    return ', '.join(parsed[:-1]) + f', & {parsed[-1]}'


def is_polluted_title(title: str | None) -> bool:
    clean = normalize_title_text(title)
    if not clean:
        return True
    if TITLE_NOISE_RE.search(str(title or '')):
        return True
    if re.fullmatch(r'[A-Za-z]{1,12}\s+\d{4}(?:\s*\(\d+\))+', clean):
        return True
    if len(clean.split()) <= 3 and re.fullmatch(r'[A-Za-z]{1,20}\s+\d{4}', clean):
        return True
    return False


def looks_like_heading(line: str) -> bool:
    return bool(re.match(r'^(?:' + '|'.join(SECTION_HEADINGS) + r')\b', line, re.I))


def looks_like_author_or_affiliation(line: str) -> bool:
    clean = normalize_title_text(line)
    if not clean:
        return False
    low = clean.lower()
    if '@' in clean or 'orcid' in low:
        return True
    if any(token in low for token in ('university', 'institute', 'department', 'school', 'college', 'laboratory', 'centre', 'center', 'dept')):
        return True
    if ',' in clean and re.search(r'\d', clean):
        return True
    if re.match(r'^\d+[\*\†\+]?\s', clean):
        return True
    return False


def strip_trailing_author_fragment(title: str | None, authors: Any = None) -> str:
    clean = normalize_title_text(title)
    if not clean:
        return ''
    words = clean.split()
    if len(words) < 5:
        return clean
    surname_candidates = []
    for author in parse_author_list(authors):
        author_clean = normalize_title_text(author)
        if not author_clean:
            continue
        if ',' in author_clean:
            surname_candidates.append(author_clean.split(',', 1)[0].strip().lower())
        else:
            surname_candidates.append(author_clean.split()[-1].lower())
    if not surname_candidates:
        return clean
    last = words[-1].rstrip('.,;:').lower()
    if last in surname_candidates and len(words) >= 7:
        return ' '.join(words[:-2]).strip() or clean
    return clean


def extract_title_from_summary_doc(doc: str | None, authors: Any = None) -> str:
    text = str(doc or '')
    if 'Objective:' not in text:
        return ''
    after = text.split('Objective:', 1)[1]
    lines = [line.strip() for line in after.splitlines()]
    title_lines: list[str] = []
    for idx, line in enumerate(lines):
        clean = normalize_title_text(re.sub(r'^[\-\*\u2022]\s*', '', line))
        if not clean:
            if title_lines:
                break
            continue
        if looks_like_heading(clean):
            break
        if title_lines and looks_like_author_or_affiliation(clean):
            break
        if not title_lines and looks_like_author_or_affiliation(clean):
            break
        title_lines.append(clean)
        if len(title_lines) >= 4:
            break
    title = strip_trailing_author_fragment(' '.join(title_lines), authors)
    return normalize_title_text(title)


def build_display_title(title: str | None, doc: str | None = None, authors: Any = None) -> str:
    raw = normalize_title_text(title)
    extracted = extract_title_from_summary_doc(doc, authors)
    if is_polluted_title(raw) and extracted:
        return extracted
    candidate = extracted or raw
    return strip_trailing_author_fragment(candidate, authors)


def build_apa_citation(title: str | None, authors: Any = None, year: Any = None) -> str:
    display_title = build_display_title(title, None, authors)
    author_text = format_apa_authors(authors)
    year_text = str(year).strip() if year not in (None, '', []) else 'n.d.'
    if author_text:
        return f'{author_text} ({year_text}). {display_title or "Untitled paper"}.'
    return f'{display_title or "Untitled paper"} ({year_text}).'


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith('---\n'):
        return {}, text
    try:
        _, rest = text.split('---\n', 1)
        fm_text, body = rest.split('\n---\n', 1)
    except ValueError:
        return {}, text
    meta = {}
    for line in fm_text.splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body


def extract_note_record(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return None
    meta, body = parse_frontmatter(raw)
    title = normalize_title_text(meta.get('title') or path.stem)
    authors = meta.get('authors') or meta.get('author') or ''
    year = meta.get('year') or meta.get('date') or ''
    if isinstance(year, str):
        m = re.search(r'(19|20)\d{2}', year)
        year = m.group(0) if m else year
    doi_match = re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', raw, re.I)
    doi = meta.get('doi') or (doi_match.group(0) if doi_match else None)
    snippet = trim(re.sub(r'[#>`*_\-]+', ' ', body), 520)
    authors_list = parse_author_list(authors)
    citation = build_apa_citation(title, authors_list or authors, year)
    source = meta.get('source') or path.name
    key = normalize_key(source or title or path.stem)
    return {
        'id': key,
        'source': source,
        'title': title,
        'display_title': citation,
        'citation': citation,
        'authors': format_apa_authors(authors_list) or normalize_title_text(authors),
        'authors_list': authors_list,
        'year': year,
        'doi': doi,
        'snippet': snippet,
        'kind': 'summary',
        'generated_at': int(path.stat().st_mtime),
        'from_obsidian_note': True,
    }


class PaperRagStore:
    def __init__(self, db_path: Path = PAPERS_DB):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=str(db_path))
        self._cache: dict[str, Any] = {}

    def collection(self, name: str):
        try:
            return self.client.get_collection(name)
        except Exception:
            return None

    def count(self, name: str) -> int:
        collection = self.collection(name)
        if not collection:
            return 0
        try:
            return int(collection.count())
        except Exception:
            return 0

    def load_all(self, name: str, include_docs: bool = True) -> tuple[list[str], list[dict], list[str]]:
        collection = self.collection(name)
        if not collection:
            return [], [], []
        include = ['metadatas'] + (['documents'] if include_docs else [])
        results = collection.get(limit=collection.count(), include=include)
        docs = results.get('documents') or []
        metas = results.get('metadatas') or []
        ids = results.get('ids') or []
        return list(docs), list(metas), list(ids)

    def summary_index(self) -> dict[str, dict[str, Any]]:
        if 'summary_index' in self._cache:
            return self._cache['summary_index']

        docs, metas, _ = self.load_all(SUMMARIES_COLLECTION, include_docs=True)
        index: dict[str, dict[str, Any]] = {}
        for doc, meta in zip(docs, metas):
            meta = dict(meta or {})
            source_key = normalize_key(
                meta.get('filename')
                or meta.get('source_note')
                or meta.get('title')
                or meta.get('source')
            )
            if not source_key:
                continue
            raw_title = meta.get('title') or Path(str(meta.get('filename') or meta.get('source_note') or source_key)).stem
            authors_list = parse_author_list(meta.get('authors'))
            authors = format_apa_authors(authors_list) or normalize_title_text(meta.get('authors'))
            title = build_display_title(raw_title, doc, authors_list)
            citation = build_apa_citation(title, authors_list, meta.get('year'))
            index[source_key] = {
                'id': source_key,
                'source': meta.get('filename') or meta.get('source_note') or meta.get('source') or title,
                'title': title,
                'display_title': citation,
                'citation': citation,
                'authors': authors,
                'authors_list': authors_list,
                'year': meta.get('year'),
                'doi': meta.get('doi'),
                'snippet': trim(doc, 520),
                'kind': 'summary',
                'generated_at': meta.get('generated_at'),
            }
        if OBSIDIAN_PAPERS_DIR.exists():
            for note_path in OBSIDIAN_PAPERS_DIR.glob('*.md'):
                record = extract_note_record(note_path)
                if not record:
                    continue
                index[record['id']] = {**index.get(record['id'], {}), **record}
        self._cache['summary_index'] = index
        return index

    def _paper_chunks(self) -> list[dict[str, Any]]:
        if 'paper_chunks' in self._cache:
            return self._cache['paper_chunks']

        docs, metas, _ = self.load_all(PAPERS_COLLECTION, include_docs=True)
        summary_map = self.summary_index()
        chunks: list[dict[str, Any]] = []

        for doc, meta in zip(docs, metas):
            meta = dict(meta or {})
            source = meta.get('source') or meta.get('filename') or meta.get('source_note') or 'unknown'
            key = normalize_key(source)
            summary = summary_map.get(key, {})
            title = summary.get('title') or Path(str(source)).stem
            record = {
                'id': f"{source}#{meta.get('chunk', meta.get('chunk_index', 0))}",
                'source': source,
                'title': title,
                'display_title': summary.get('display_title') or build_apa_citation(title, summary.get('authors_list') or summary.get('authors'), summary.get('year')),
                'citation': summary.get('citation') or build_apa_citation(title, summary.get('authors_list') or summary.get('authors'), summary.get('year')),
                'authors': summary.get('authors'),
                'authors_list': summary.get('authors_list') or parse_author_list(summary.get('authors')),
                'year': summary.get('year'),
                'doi': summary.get('doi'),
                'snippet': trim(doc, 520),
                'kind': 'paper',
                'chunk': meta.get('chunk', meta.get('chunk_index', 0)),
                'full_text': doc,
            }
            chunks.append(record)

        self._cache['paper_chunks'] = chunks
        return chunks

    def papers(self) -> list[dict[str, Any]]:
        if 'papers' in self._cache:
            return self._cache['papers']

        summary_map = self.summary_index()
        chunks = self._paper_chunks()
        papers: dict[str, dict[str, Any]] = {}

        for chunk in chunks:
            key = normalize_key(chunk['source'])
            if key not in papers:
                papers[key] = {
                    'id': key,
                    'source': chunk['source'],
                    'title': chunk['title'],
                    'display_title': chunk.get('display_title') or build_apa_citation(chunk['title'], chunk.get('authors_list') or chunk.get('authors'), chunk.get('year')),
                    'citation': chunk.get('citation') or build_apa_citation(chunk['title'], chunk.get('authors_list') or chunk.get('authors'), chunk.get('year')),
                    'authors': chunk.get('authors'),
                    'authors_list': chunk.get('authors_list') or parse_author_list(chunk.get('authors')),
                    'year': chunk.get('year'),
                    'doi': chunk.get('doi'),
                    'snippet': chunk.get('snippet') or '',
                    'kind': 'paper',
                    'chunk_count': 1,
                    'has_summary': key in summary_map,
                }
            else:
                papers[key]['chunk_count'] += 1
                if not papers[key]['snippet'] and chunk.get('snippet'):
                    papers[key]['snippet'] = chunk['snippet']
                if not papers[key].get('display_title') and chunk.get('display_title'):
                    papers[key]['display_title'] = chunk['display_title']
                if not papers[key].get('citation') and chunk.get('citation'):
                    papers[key]['citation'] = chunk['citation']

        # Add summary-only records that do not have PDF chunks yet.
        for key, summary in summary_map.items():
            if key in papers:
                continue
            papers[key] = {
                'id': key,
                'source': summary.get('source'),
                'title': summary.get('title') or Path(str(summary.get('source') or key)).stem,
                'display_title': summary.get('display_title') or build_apa_citation(summary.get('title') or Path(str(summary.get('source') or key)).stem, summary.get('authors_list') or summary.get('authors'), summary.get('year')),
                'citation': summary.get('citation') or build_apa_citation(summary.get('title') or Path(str(summary.get('source') or key)).stem, summary.get('authors_list') or summary.get('authors'), summary.get('year')),
                'authors': summary.get('authors'),
                'authors_list': summary.get('authors_list') or parse_author_list(summary.get('authors')),
                'year': summary.get('year'),
                'doi': summary.get('doi'),
                'snippet': summary.get('snippet') or '',
                'kind': 'summary',
                'chunk_count': 0,
                'has_summary': True,
            }

        ordered = sorted(
            papers.values(),
            key=lambda row: (
                -(int(str(row.get('year') or 0)) if str(row.get('year') or '').isdigit() else 0),
                str(row.get('title') or '').lower(),
            ),
        )
        self._cache['papers'] = ordered
        return ordered

    def stats(self) -> dict[str, Any]:
        papers = self.papers()
        summary_count = len(self.summary_index())
        chunk_count = self.count(PAPERS_COLLECTION)
        fulltext_papers = len([p for p in papers if (p.get('chunk_count') or 0) > 0])
        metadata_only_records = len([p for p in papers if (p.get('chunk_count') or 0) == 0])
        return {
            'status': 'ok',
            'service': 'rag-api-wrapper',
            'version': '2.0.0',
            'mode': 'public_read_only',
            'indexed_papers': len(papers),
            'paper_count': fulltext_papers,
            'fulltext_papers': fulltext_papers,
            'metadata_only_records': metadata_only_records,
            'summary_count': summary_count,
            'collection_count': chunk_count,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'llm_ready': bool(os.getenv('OPENROUTER_API_KEY')),
        }

    def _tokens(self, text: str) -> list[str]:
        raw = [tok.lower() for tok in TOKEN_RE.findall(text or '') if len(tok) > 1]
        expanded = []
        for tok in raw:
            expanded.extend(TERM_EXPANSIONS.get(tok, [tok]))
        seen = []
        for tok in expanded:
            if tok not in seen:
                seen.append(tok)
        return seen

    def _score_text(self, query_tokens: list[str], fields: list[str]) -> float:
        if not query_tokens:
            return 0.0
        scored = 0.0
        joined = ' \n '.join(fields).lower()
        title = fields[0].lower() if fields else ''
        source = fields[-1].lower() if fields else ''
        for token in query_tokens:
            freq = joined.count(token)
            if not freq:
                continue
            weight = 1.0
            if token in title:
                weight += 2.5
            if token in source:
                weight += 3.0
            scored += min(6.0, freq) * weight
        phrase = ' '.join(query_tokens)
        if phrase and phrase in joined:
            scored += 4.0
        if phrase and phrase in source:
            scored += 6.0
        # Normalize into roughly 0..1
        denom = max(8.0, len(query_tokens) * 6.0)
        return min(0.99, scored / denom)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        q = (query or '').strip()
        if not q:
            return []
        tokens = self._tokens(q)
        if not tokens:
            return []

        candidates = []
        for paper in self.papers():
            fields = [
                str(paper.get('title') or ''),
                str(paper.get('authors') or ''),
                str(paper.get('year') or ''),
                str(paper.get('snippet') or ''),
                str(paper.get('source') or ''),
            ]
            score = self._score_text(tokens, fields)
            if score <= 0:
                continue
            candidates.append({
                'id': paper['id'],
                'source': paper.get('source'),
                'title': paper.get('title') or Path(str(paper.get('source') or paper['id'])).stem,
                'display_title': paper.get('display_title') or build_apa_citation(paper.get('title') or Path(str(paper.get('source') or paper['id'])).stem, paper.get('authors_list') or paper.get('authors'), paper.get('year')),
                'citation': paper.get('citation') or build_apa_citation(paper.get('title') or Path(str(paper.get('source') or paper['id'])).stem, paper.get('authors_list') or paper.get('authors'), paper.get('year')),
                'authors': paper.get('authors'),
                'authors_list': paper.get('authors_list') or parse_author_list(paper.get('authors')),
                'year': paper.get('year'),
                'doi': paper.get('doi'),
                'snippet': paper.get('snippet') or '',
                'score': round(score, 4),
                'kind': paper.get('kind'),
                'chunk_count': paper.get('chunk_count', 0),
                'has_summary': paper.get('has_summary', False),
            })

        candidates.sort(key=lambda item: (item['score'], str(item.get('year') or '')), reverse=True)
        deduped: dict[str, dict[str, Any]] = {}
        for item in candidates:
            key = normalize_key(item.get('source') or item.get('title') or item['id'])
            existing = deduped.get(key)
            if not existing or item['score'] > existing['score']:
                deduped[key] = item
        return list(deduped.values())[:top_k]

    def browse(self, page: int = 1, limit: int = 24) -> dict[str, Any]:
        page = max(1, int(page or 1))
        limit = max(1, min(100, int(limit or 24)))
        papers = self.papers()
        total = len(papers)
        start = (page - 1) * limit
        end = start + limit
        rows = papers[start:end]
        return {
            'page': page,
            'limit': limit,
            'total': total,
            'hasMore': end < total,
            'papers': rows,
        }

    def answer(self, query: str, top_k: int = 5) -> dict[str, Any]:
        results = self.search(query, top_k=max(3, min(8, top_k)))
        if not results:
            return {
                'query': query,
                'answer': 'Saya belum menemukan konteks yang cukup di indeks publik untuk pertanyaan itu.',
                'sources': [],
                'llm_used': False,
            }

        context_parts = []
        for item in results[:top_k]:
            source_line = item.get('citation') or item.get('display_title') or item.get('title') or item.get('source') or 'Unknown source'
            context_parts.append(
                f"[Source: {source_line}]\n"
                f"Authors: {item.get('authors') or '—'}\n"
                f"Score: {item.get('score')}\n"
                f"Snippet: {item.get('snippet') or '—'}"
            )
        context = '\n\n---\n\n'.join(context_parts)

        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            return {
                'query': query,
                'answer': 'LLM belum aktif di environment ini. Context retrieval sudah jalan, tapi API key OpenRouter belum tersedia.',
                'sources': results[:top_k],
                'llm_used': False,
            }

        prompt = (
            'You are the public Orebit RAG assistant.\n'
            'Answer using ONLY the provided paper context.\n'
            'If the context is insufficient, say so plainly.\n'
            'Reply in the same language as the user, concise but useful, and include the most relevant source titles at the end.\n\n'
            f'Context:\n{context}\n\n'
            f'Question: {query}\n'
        )

        payload = {
            'model': OPENROUTER_MODEL,
            'messages': [
                {'role': 'system', 'content': 'You answer from indexed academic paper context only.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0.2,
            'max_tokens': 700,
        }
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'X-Title': 'Orebit Public RAG',
        }

        answer = None
        llm_used = True
        fallback_reason = None
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            answer = data['choices'][0]['message']['content']
        except Exception as exc:
            llm_used = False
            fallback_reason = str(exc)
            top = results[0]
            second = results[1] if len(results) > 1 else None
            lead = trim(top.get('snippet') or '', 420)
            tail = trim(second.get('snippet') or '', 220) if second else ''
            pieces = [
                f"LLM sementara tidak tersedia ({fallback_reason}).",
                f"Dari konteks terindeks, {top.get('citation') or top.get('display_title') or top.get('title') or top.get('source') or 'sumber utama'} menunjukkan bahwa {lead}".rstrip(),
            ]
            if tail:
                pieces.append(f"Sumber pendukung menambahkan: {tail}")
            answer = ' '.join(piece for piece in pieces if piece)

        return {
            'query': query,
            'answer': answer,
            'sources': results[:top_k],
            'llm_used': llm_used,
            'fallback_reason': fallback_reason,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description='Public paper RAG data provider')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('stats')

    p_search = sub.add_parser('search')
    p_search.add_argument('--query', required=True)
    p_search.add_argument('--top-k', type=int, default=10)

    p_browse = sub.add_parser('browse')
    p_browse.add_argument('--page', type=int, default=1)
    p_browse.add_argument('--limit', type=int, default=24)

    p_answer = sub.add_parser('answer')
    p_answer.add_argument('--query', required=True)
    p_answer.add_argument('--top-k', type=int, default=5)

    args = parser.parse_args()
    store = PaperRagStore()

    if args.command == 'stats':
        print(json.dumps(store.stats(), ensure_ascii=False))
        return 0
    if args.command == 'search':
        results = store.search(args.query, args.top_k)
        print(json.dumps({'query': args.query, 'results': results, 'total': len(results)}, ensure_ascii=False))
        return 0
    if args.command == 'browse':
        print(json.dumps(store.browse(args.page, args.limit), ensure_ascii=False))
        return 0
    if args.command == 'answer':
        print(json.dumps(store.answer(args.query, args.top_k), ensure_ascii=False))
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
