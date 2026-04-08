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
import sqlite3
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import chromadb
warnings.filterwarnings(
    'ignore',
    message=r".*doesn't match a supported version!.*",
    category=Warning,
)
import requests

WORKSPACE = Path('/root/.openclaw/workspace')
PAPERS_DB = WORKSPACE / '.vector_db' / 'papers'
CHROMA_SQLITE = PAPERS_DB / 'chroma.sqlite3'
INDEX_STATE_PAPERS = PAPERS_DB / 'index_state_papers.json'
PAPERS_COLLECTION = 'papers'
SUMMARIES_COLLECTION = 'papers_summary'
OBSIDIAN_PAPERS_DIR = Path('/data/obsidian/3. Resources/Papers')
ACTIVE_PDF_ROOT = Path('/mnt/gdrive/AI_Knowledge')
TRACKER_PATH = WORKSPACE / 'research' / 'paper-tracker' / 'papers.json'
PAPER_PARITY_STATE = WORKSPACE / '.state' / 'paper_count_parity.json'
ZAI_MODEL = os.getenv('ZAI_MODEL', 'glm-4.7-flash')
ZAI_URL = os.getenv('ZAI_URL', 'https://api.z.ai/api/coding/paas/v4/chat/completions')

FOLLOWUP_PRONOUNS = {
    'it', 'its', 'they', 'them', 'their', 'theirs', 'this', 'that', 'these', 'those',
    'he', 'she', 'his', 'her', 'hers', 'former', 'latter'
}

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Stopwords — common English words that add noise to keyword scoring
STOPWORDS = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'must',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they',
    'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
    'how', 'when', 'where', 'why',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'into', 'about', 'between', 'through', 'during', 'before', 'after',
    'and', 'or', 'but', 'not', 'no', 'nor', 'so', 'if', 'then',
    'up', 'out', 'off', 'over', 'under', 'again', 'further',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'only', 'own', 'same', 'than', 'too', 'very',
    'just', 'also', 'now', 'here', 'there',
})

TERM_EXPANSIONS = {
    'bouger': ['bouger', 'bouguer'],
    'bouguer': ['bouguer', 'bouger'],
    'gravimetri': ['gravimetri', 'gravity', 'gravity anomaly'],
    'gravity': ['gravity', 'gravimetri', 'gravity anomaly'],
    'anomaly': ['anomaly', 'anomali'],
    'simamora': ['simamora'],
    'kriging': ['kriging', 'krige', 'geostatistic', 'geostatistics', 'variogram'],
    'variogram': ['variogram', 'semivariogram', 'kriging', 'geostatistic'],
    'geostatistic': ['geostatistic', 'geostatistics', 'kriging', 'variogram'],
    'geostatistics': ['geostatistics', 'geostatistic', 'kriging', 'variogram'],
    'interpolation': ['interpolation', 'interpolate', 'kriging', 'spatial'],
}


def load_paper_parity_state() -> dict[str, Any]:
    if not PAPER_PARITY_STATE.exists():
        return {}
    try:
        return json.loads(PAPER_PARITY_STATE.read_text(encoding='utf-8'))
    except Exception:
        return {}


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


def strip_xml_tags(text: str | None) -> str:
    if not text:
        return ''
    clean = str(text).replace('\r', '\n')
    clean = re.sub(r'</?jats:[^>]+>', ' ', clean, flags=re.I)
    clean = re.sub(r'</?[^>]+>', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()


def safe_json_load(text: str, fallback: Any = None) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return fallback


def clean_assistant_answer(text: str | None) -> str:
    if not text:
        return ''
    clean = strip_xml_tags(text).replace('\r', '\n')
    clean = clean.replace('**', '').replace('__', '').replace('`', '')
    clean = re.sub(r'(?m)^\s*[*-]\s+', '• ', clean)
    clean = re.sub(r'(?m)^\s*•\s*', '• ', clean)
    clean = re.sub(r'[ \t]+\n', '\n', clean)
    clean = re.sub(r'\n[ \t]+', '\n', clean)
    clean = re.sub(r'[ \t]{2,}', ' ', clean)
    clean = re.sub(r'\n{3,}', '\n\n', clean)
    return clean.strip()


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


def extract_markdown_section(body: str, heading: str) -> str:
    if not body:
        return ''
    pattern = re.compile(
        rf'^\s*##\s+{re.escape(heading)}\s*$([\s\S]*?)(?=^\s*##\s+|\Z)',
        re.I | re.M,
    )
    match = pattern.search(body)
    if not match:
        return ''
    section = match.group(1).strip()
    section = re.sub(r'(?m)^\s*[-*]\s+', '• ', section)
    return strip_xml_tags(section)


def extract_key_findings(body: str) -> list[str]:
    if not body:
        return []
    pattern = re.compile(
        r'^\s*##\s+Key Findings\s*$([\s\S]*?)(?=^\s*##\s+|\Z)',
        re.I | re.M,
    )
    match = pattern.search(body)
    if not match:
        return []
    section = match.group(1).strip()
    findings = []
    for line in section.splitlines():
        clean = line.strip()
        if not clean:
            continue
        clean = re.sub(r'^\s*[-*\u2022]\s*', '', clean)
        clean = re.sub(r'^\[[^\]]+\]\s*', '', clean)
        clean = strip_xml_tags(clean)
        clean = trim(clean, 220)
        if clean:
            findings.append(clean)
        if len(findings) >= 3:
            break
    return findings


def build_obsidian_uri(path: Path) -> str | None:
    try:
        relative = path.relative_to(Path('/data/obsidian'))
    except Exception:
        return None
    relative_text = str(relative).replace('\\', '/')
    return f"obsidian://open?vault=obsidian&file={quote(relative_text, safe='/')}"


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
    overview = extract_markdown_section(body, 'Overview')
    significance = extract_markdown_section(body, 'Significance')
    note_summary = trim(overview or significance or snippet, 900)
    relative_path = None
    try:
        relative_path = str(path.relative_to(Path('/data/obsidian'))).replace('\\', '/')
    except Exception:
        relative_path = path.name
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
        'obsidian_summary': note_summary,
        'obsidian_key_findings': extract_key_findings(body),
        'obsidian_note_path': relative_path,
        'obsidian_uri': build_obsidian_uri(path),
        'kind': 'summary',
        'generated_at': int(path.stat().st_mtime),
        'from_obsidian_note': True,
    }


def find_matching_summary(record: dict[str, Any], summary_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not summary_map:
        return None

    candidates = [
        normalize_key(record.get('id')),
        normalize_key(record.get('doi')),
        normalize_key(record.get('source')),
        normalize_key(record.get('title')),
        normalize_key(Path(str(record.get('source') or '')).stem),
    ]

    for candidate in candidates:
        if candidate and candidate in summary_map:
            return summary_map[candidate]

    record_doi = normalize_key(record.get('doi'))
    record_title = normalize_key(record.get('title'))
    record_source_stem = normalize_key(Path(str(record.get('source') or '')).stem)

    for summary in summary_map.values():
        summary_doi = normalize_key(summary.get('doi'))
        summary_title = normalize_key(summary.get('title'))
        summary_source_stem = normalize_key(Path(str(summary.get('source') or '')).stem)
        if record_doi and summary_doi and record_doi == summary_doi:
            return summary
        if record_title and summary_title and record_title == summary_title:
            return summary
        if record_source_stem and summary_source_stem and record_source_stem == summary_source_stem:
            return summary

    return None


def load_tracker_pending_records() -> list[dict[str, Any]]:
    if not TRACKER_PATH.exists():
        return []
    try:
        rows = json.loads(TRACKER_PATH.read_text())
    except Exception:
        return []

    records = []
    for row in rows:
        if row.get('status') != 'pending':
            continue
        title = normalize_title_text(row.get('title') or row.get('id') or 'Untitled')
        authors_list = parse_author_list(row.get('authors'))
        citation = build_apa_citation(title, authors_list or row.get('authors'), row.get('year'))
        blocker = str(row.get('pending_reason') or 'pending').replace('_', ' ')
        source_url = row.get('source_url') or row.get('pdf_url') or row.get('download_url')
        records.append({
            'id': normalize_key(row.get('doi') or title or row.get('id')),
            'source': source_url or row.get('id'),
            'title': title,
            'display_title': citation,
            'citation': citation,
            'authors': format_apa_authors(authors_list) or normalize_title_text(row.get('authors')),
            'authors_list': authors_list,
            'year': row.get('year'),
            'doi': row.get('doi'),
            'snippet': f"Approved paper pending full-text download. Current blocker: {blocker}.",
            'kind': 'pending_metadata',
            'generated_at': row.get('downloaded_date') or row.get('resolver_last_attempt_at'),
            'pending_reason': row.get('pending_reason'),
            'url': source_url,
            'source_ref': source_url,
            'has_summary': False,
            'chunk_count': 0,
        })
    return records


def load_indexed_paper_state() -> tuple[int, set[str]]:
    if not INDEX_STATE_PAPERS.exists():
        return 0, set()
    try:
        payload = json.loads(INDEX_STATE_PAPERS.read_text())
    except Exception:
        return 0, set()

    files = payload.get('files') if isinstance(payload, dict) else {}
    if not isinstance(files, dict):
        return 0, set()
    active_pdf_names = {p.name for p in ACTIVE_PDF_ROOT.glob('*.pdf')} if ACTIVE_PDF_ROOT.exists() else set()
    filtered_names = [name for name in files.keys() if str(name).strip() and (not active_pdf_names or str(name) in active_pdf_names)]
    normalized = {normalize_key(name) for name in filtered_names}
    return len(filtered_names), normalized


def chroma_collection_embedding_count(name: str) -> int:
    if not CHROMA_SQLITE.exists():
        return 0
    try:
        conn = sqlite3.connect(str(CHROMA_SQLITE))
        cur = conn.cursor()
        cur.execute(
            '''
            select count(e.id)
            from embeddings e
            join segments s on e.segment_id = s.id
            join collections c on s.collection = c.id
            where c.name = ?
            ''',
            (name,),
        )
        row = cur.fetchone()
        return int(row[0] or 0) if row else 0
    except Exception:
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


def build_tracker_active_records(indexed_keys: set[str]) -> list[dict[str, Any]]:
    if not TRACKER_PATH.exists():
        return []
    try:
        rows = json.loads(TRACKER_PATH.read_text())
    except Exception:
        return []

    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get('corpus_state') != 'active':
            continue
        meta = row.get('metadata') or {}
        source = (
            row.get('filename')
            or row.get('storage_location')
            or row.get('title')
            or row.get('id')
            or 'Untitled'
        )
        title = normalize_title_text(meta.get('title') or row.get('title') or Path(str(source)).stem)
        authors_list = parse_author_list(row.get('authors') or meta.get('authors'))
        citation = build_apa_citation(title, authors_list or row.get('authors') or meta.get('authors'), meta.get('year'))
        indexed = normalize_key(str(source)) in indexed_keys
        abstract = trim(meta.get('abstract') or '', 520)
        tags = ', '.join(str(tag) for tag in (row.get('tags') or [])[:4] if str(tag).strip())
        venue = normalize_title_text(row.get('journal') or meta.get('venue') or meta.get('publication'))
        snippet = abstract
        if not snippet:
            snippet_parts = []
            if venue:
                snippet_parts.append(f'Venue: {venue}.')
            if tags:
                snippet_parts.append(f'Tags: {tags}.')
            snippet = ' '.join(snippet_parts) or 'Metadata record available for this paper.'
        records.append({
            'id': normalize_key(row.get('doi') or source or row.get('id')),
            'source': source,
            'title': title,
            'display_title': citation,
            'citation': citation,
            'authors': format_apa_authors(authors_list) or normalize_title_text(', '.join(authors_list) if authors_list else row.get('authors')),
            'authors_list': authors_list,
            'year': meta.get('year'),
            'doi': row.get('doi') or meta.get('doi'),
            'snippet': snippet,
            'kind': 'paper' if indexed else 'metadata',
            'chunk_count': 1 if indexed else 0,
            'chunk_count_estimated': indexed,
            'indexed_fulltext': indexed,
            'has_summary': False,
            'url': row.get('url'),
            'source_ref': row.get('url'),
            'project': row.get('project'),
            'journal': venue,
        })
    return records


class SafePaperRagStore:
    def __init__(self):
        self._cache: dict[str, Any] = {}

    def _tokens(self, text: str) -> list[str]:
        raw = [tok.lower() for tok in TOKEN_RE.findall(text or '') if len(tok) > 1]
        filtered = [tok for tok in raw if tok not in STOPWORDS]
        if not filtered:
            filtered = raw
        expanded = []
        for tok in filtered:
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
        matched_tokens = 0
        joined = ' \n '.join(fields).lower()
        title = fields[0].lower() if fields else ''
        source = fields[-1].lower() if fields else ''
        for token in query_tokens:
            freq = joined.count(token)
            if not freq:
                continue
            matched_tokens += 1
            weight = 1.0
            if token in title:
                weight += 3.0
            if token in source:
                weight += 3.5
            scored += min(6.0, freq) * weight
        if not matched_tokens:
            return 0.0
        coverage = matched_tokens / len(query_tokens)
        phrase = ' '.join(query_tokens)
        if phrase and phrase in joined:
            scored += 5.0
        if phrase and phrase in title:
            scored += 8.0
        if phrase and phrase in source:
            scored += 6.0
        denom = max(10.0, len(query_tokens) * 7.0)
        raw_score = min(0.99, scored / denom)
        return round(raw_score * (0.4 + 0.6 * coverage), 4)

    def _rewrite_followup_query(self, query: str, history: list | None = None) -> str:
        q = (query or '').strip()
        if not q or not history:
            return q
        if not any(tok in FOLLOWUP_PRONOUNS for tok in [t.lower() for t in TOKEN_RE.findall(q)]):
            return q
        for msg in reversed(history[-6:]):
            if msg.get('role') != 'user':
                continue
            prev = str(msg.get('content') or '').strip()
            if not prev or prev == q:
                continue
            prev_tokens = self._tokens(prev)
            if len(prev_tokens) < 2:
                continue
            return f"{q} (context: {prev})"
        return q

    def _clean_snippet(self, text: str | None) -> str:
        raw = trim(strip_xml_tags(text or ''), 420)
        if not raw:
            return ''
        cleaned = raw.replace('LLM Summary:', '').replace('Objective:', '').replace('Overview', '').strip(' -:')
        cleaned = re.sub(r'\[\[[^\]]+\]\]', '', cleaned)
        cleaned = re.sub(r'(Related Concepts|Citation|Indexed|Source)\s*:?.*', '', cleaned, flags=re.I)
        cleaned = re.sub(r'\b(Related Concepts|Citation|Indexed|Source)\b.*', '', cleaned, flags=re.I)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip(' -:;,.')
        return cleaned

    def _extract_definition_line(self, text: str | None, query_tokens: list[str]) -> str:
        cleaned = self._clean_snippet(text)
        if not cleaned:
            return ''
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        priority = []
        for s in sentences:
            s2 = s.strip()
            low = s2.lower()
            if any(tok in low for tok in query_tokens) and any(key in low for key in [' is ', ' are ', ' refers to ', ' defined as ', ' method', ' interpolation', ' technique']):
                priority.append(s2)
        if priority:
            return trim(priority[0], 260)
        return trim(sentences[0] if sentences else cleaned, 260)

    def _rerank_candidates(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        q = (query or '').strip().lower()
        q_tokens = self._tokens(q)
        reranked = []
        for item in candidates:
            score = float(item.get('score') or 0.0)
            chunk_count = int(item.get('chunk_count') or 0)
            title = str(item.get('title') or '').lower()
            snippet = str(item.get('snippet') or '').lower()
            kind = str(item.get('kind') or '').lower()
            year_text = str(item.get('year') or '')
            year_val = int(year_text) if year_text.isdigit() else 0
            if kind == 'paper':
                score += 0.10
            if chunk_count > 0:
                score += 0.22
            else:
                score -= 0.10
            title_hits = sum(1 for tok in q_tokens if tok in title)
            score += min(0.24, title_hits * 0.08)
            if q.startswith('what is') or q.startswith('what are'):
                if any(key in snippet for key in [' is ', ' are ', 'refers to', 'defined as', 'interpolation', 'geostatistical method']):
                    score += 0.10
            if len(snippet) > 180:
                score += 0.03
            if year_val and year_val <= 2010:
                score += 0.02
            clean_snippet = self._clean_snippet(item.get('snippet') or '')
            definition_snippet = self._extract_definition_line(item.get('snippet') or '', q_tokens)
            reranked.append({
                **item,
                'score': round(min(0.99, score), 4),
                'snippet': clean_snippet,
                'definition_snippet': definition_snippet,
            })
        reranked.sort(key=lambda item: (item['score'], int(str(item.get('year') or 0)) if str(item.get('year') or '').isdigit() else 0), reverse=True)
        return reranked[:top_k]

    def summary_index(self) -> dict[str, dict[str, Any]]:
        if 'summary_index' in self._cache:
            return self._cache['summary_index']
        index: dict[str, dict[str, Any]] = {}
        if OBSIDIAN_PAPERS_DIR.exists():
            for note_path in OBSIDIAN_PAPERS_DIR.glob('*.md'):
                record = extract_note_record(note_path)
                if not record:
                    continue
                index[record['id']] = record
        self._cache['summary_index'] = index
        return index

    def papers(self) -> list[dict[str, Any]]:
        if 'papers' in self._cache:
            return self._cache['papers']

        indexed_count, indexed_keys = load_indexed_paper_state()
        summary_map = self.summary_index()
        papers: dict[str, dict[str, Any]] = {}

        for record in build_tracker_active_records(indexed_keys):
            papers[record['id']] = record

        for key, summary in summary_map.items():
            if key in papers:
                existing = papers[key]
                merged = {
                    **existing,
                    'title': summary.get('title') or existing.get('title'),
                    'display_title': summary.get('display_title') or existing.get('display_title'),
                    'citation': summary.get('citation') or existing.get('citation'),
                    'authors': summary.get('authors') or existing.get('authors'),
                    'authors_list': summary.get('authors_list') or existing.get('authors_list'),
                    'year': summary.get('year') or existing.get('year'),
                    'doi': summary.get('doi') or existing.get('doi'),
                    'snippet': summary.get('snippet') or existing.get('snippet'),
                    'obsidian_summary': summary.get('obsidian_summary') or existing.get('obsidian_summary'),
                    'obsidian_key_findings': summary.get('obsidian_key_findings') or existing.get('obsidian_key_findings'),
                    'obsidian_note_path': summary.get('obsidian_note_path') or existing.get('obsidian_note_path'),
                    'obsidian_uri': summary.get('obsidian_uri') or existing.get('obsidian_uri'),
                    'from_obsidian_note': bool(summary.get('from_obsidian_note') or existing.get('from_obsidian_note')),
                    'has_summary': True,
                }
                papers[key] = merged
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
                'obsidian_summary': summary.get('obsidian_summary'),
                'obsidian_key_findings': summary.get('obsidian_key_findings') or [],
                'obsidian_note_path': summary.get('obsidian_note_path'),
                'obsidian_uri': summary.get('obsidian_uri'),
                'from_obsidian_note': bool(summary.get('from_obsidian_note')),
                'kind': 'summary',
                'chunk_count': 0,
                'chunk_count_estimated': False,
                'indexed_fulltext': False,
                'has_summary': True,
                'url': summary.get('url') or summary.get('source_ref'),
                'source_ref': summary.get('source_ref') or summary.get('url'),
            }

        for key, paper in list(papers.items()):
            summary = find_matching_summary(paper, summary_map)
            if not summary:
                continue
            papers[key] = {
                **paper,
                'title': summary.get('title') or paper.get('title'),
                'display_title': summary.get('display_title') or paper.get('display_title'),
                'citation': summary.get('citation') or paper.get('citation'),
                'authors': summary.get('authors') or paper.get('authors'),
                'authors_list': summary.get('authors_list') or paper.get('authors_list'),
                'year': summary.get('year') or paper.get('year'),
                'doi': summary.get('doi') or paper.get('doi'),
                'snippet': summary.get('snippet') or paper.get('snippet'),
                'obsidian_summary': summary.get('obsidian_summary') or paper.get('obsidian_summary'),
                'obsidian_key_findings': summary.get('obsidian_key_findings') or paper.get('obsidian_key_findings') or [],
                'obsidian_note_path': summary.get('obsidian_note_path') or paper.get('obsidian_note_path'),
                'obsidian_uri': summary.get('obsidian_uri') or paper.get('obsidian_uri'),
                'from_obsidian_note': bool(summary.get('from_obsidian_note') or paper.get('from_obsidian_note')),
                'has_summary': True,
            }

        for pending in load_tracker_pending_records():
            pending_doi = str((pending.get('doi') or '')).lower()
            pending_key = pending.get('id')
            existing = papers.get(pending_key)
            if existing:
                continue
            if pending_doi and any(str((p.get('doi') or '')).lower() == pending_doi for p in papers.values()):
                continue
            papers[pending_key] = pending

        ordered = sorted(
            papers.values(),
            key=lambda row: (
                0 if row.get('indexed_fulltext') else 1,
                0 if row.get('has_summary') else 1,
                0 if row.get('kind') == 'paper' else 1,
                -(int(str(row.get('year') or 0)) if str(row.get('year') or '').isdigit() else 0),
                str(row.get('title') or '').lower(),
            ),
        )
        self._cache['papers'] = ordered
        self._cache['indexed_count'] = indexed_count
        return ordered

    def stats(self) -> dict[str, Any]:
        papers = self.papers()
        indexed_count = int(self._cache.get('indexed_count') or load_indexed_paper_state()[0])
        summary_count = len(self.summary_index())
        chunk_count = chroma_collection_embedding_count(PAPERS_COLLECTION)
        metadata_only_records = len([p for p in papers if not p.get('indexed_fulltext')])
        parity = load_paper_parity_state()
        peer = parity.get('peer_reviewed') or {}
        arxiv = parity.get('arxiv') or {}
        rag = parity.get('rag') or {}
        return {
            'status': 'ok',
            'service': 'rag-api-wrapper',
            'version': '2.0.1-safe',
            'mode': 'public_read_only',
            'llm_model': ZAI_MODEL,
            'indexed_papers': indexed_count,
            'paper_count': indexed_count,
            'fulltext_papers': indexed_count,
            'metadata_only_records': metadata_only_records,
            'summary_count': summary_count,
            'collection_count': chunk_count,
            'peer_reviewed_ok': parity.get('status') == 'PASS',
            'peer_reviewed_gdrive_pdfs': peer.get('active_gdrive_pdfs'),
            'peer_reviewed_obsidian_notes': peer.get('obsidian_summary_notes'),
            'parity_gap': peer.get('parity_gap'),
            'arxiv_violations': arxiv.get('tracker_downloaded_policy_violations'),
            'arxiv_workspace_rag': arxiv.get('workspace_rag_entries', rag.get('arxiv_workspace_rag')),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'llm_ready': bool(os.getenv('ZAI_API_KEY') or os.getenv('OPENCLAW_MODELS_PROVIDERS_ZAI_APIKEY')),
            'fallback_mode': 'tracker_sqlite_safe',
        }

    def search(self, query: str, top_k: int = 10, history: list | None = None) -> list[dict[str, Any]]:
        q = self._rewrite_followup_query((query or '').strip(), history)
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
                str(paper.get('project') or ''),
                str(paper.get('journal') or ''),
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
                'obsidian_summary': paper.get('obsidian_summary'),
                'obsidian_key_findings': paper.get('obsidian_key_findings') or [],
                'obsidian_note_path': paper.get('obsidian_note_path'),
                'obsidian_uri': paper.get('obsidian_uri'),
                'from_obsidian_note': bool(paper.get('from_obsidian_note')),
                'score': round(score, 4),
                'kind': paper.get('kind'),
                'chunk_count': paper.get('chunk_count', 0),
                'chunk_count_estimated': bool(paper.get('chunk_count_estimated')),
                'indexed_fulltext': bool(paper.get('indexed_fulltext')),
                'has_summary': paper.get('has_summary', False),
            })

        deduped: dict[str, dict[str, Any]] = {}
        for item in candidates:
            key = normalize_key(item.get('source') or item.get('title') or item['id'])
            existing = deduped.get(key)
            if not existing or item['score'] > existing['score']:
                deduped[key] = item

        reranked = self._rerank_candidates(q, list(deduped.values()), top_k=max(top_k * 2, 10))
        return reranked[:top_k]

    def browse(self, page: int = 1, limit: int = 24) -> dict[str, Any]:
        page = max(1, int(page or 1))
        limit = max(1, min(100, int(limit or 24)))
        papers = self.papers()
        total = len(papers)
        start = (page - 1) * limit
        end = start + limit
        return {
            'page': page,
            'limit': limit,
            'total': total,
            'hasMore': end < total,
            'papers': papers[start:end],
        }

    def answer(self, query: str, top_k: int = 5, history: list | None = None) -> dict[str, Any]:
        rewritten_query = self._rewrite_followup_query(query, history)
        results = self.search(rewritten_query, top_k=max(3, min(8, top_k)), history=history)
        if not results:
            return {
                'query': query,
                'answer': 'No relevant papers found in the public index for that query.',
                'sources': [],
                'llm_used': False,
            }

        context_parts = []
        for item in results[:top_k]:
            source_line = item.get('citation') or item.get('display_title') or item.get('title') or item.get('source') or 'Unknown source'
            best_snippet = item.get('definition_snippet') or item.get('snippet') or '—'
            context_parts.append(
                f"[Source: {source_line}]\n"
                f"Authors: {item.get('authors') or '—'}\n"
                f"Year: {item.get('year') or '—'}\n"
                f"DOI: {item.get('doi') or '—'}\n"
                f"Evidence: {best_snippet}"
            )
        context = '\n\n---\n\n'.join(context_parts)

        api_key = os.getenv('ZAI_API_KEY') or os.getenv('OPENCLAW_MODELS_PROVIDERS_ZAI_APIKEY')
        if not api_key:
            top = results[0]
            answer = top.get('definition_snippet') or top.get('snippet') or 'Relevant metadata found, but no summary text is available.'
            return {
                'query': query,
                'answer': f"{answer}\n\nKey sources: {top.get('title') or top.get('citation') or top.get('source')}",
                'sources': results[:top_k],
                'llm_used': False,
            }

        prompt = (
            'You are the public Orebit RAG assistant.\n'
            'Answer using ONLY the provided paper context.\n'
            'If the context is insufficient, say so plainly.\n'
            'Write a crisp, useful answer for a technical reader.\n'
            'For definitional questions, give: (1) one-sentence definition, (2) 1-2 key mechanics or assumptions, (3) one practical note if present in context.\n'
            'Return plain text only. Do not use markdown bold markers, bullet syntax, code fences, or HTML/XML tags.\n'
            'Do not invent claims beyond the sources. End naturally with a short line starting with "Key sources:" followed by 1-3 titles only.\n\n'
            f'Retrieval query used: {rewritten_query}\n\n'
            f'Context:\n{context}\n\n'
            f'Original user question: {query}\n'
        )

        payload = {
            'model': ZAI_MODEL,
            'messages': [
                {'role': 'system', 'content': 'You answer from indexed academic paper context only.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0.2,
            'max_tokens': 700,
            'thinking': {'type': 'disabled'},
        }
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

        try:
            response = requests.post(ZAI_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            msg = data['choices'][0]['message']
            answer = clean_assistant_answer(msg.get('content') or msg.get('reasoning_content') or '')
            return {
                'query': query,
                'answer': answer,
                'sources': results[:top_k],
                'llm_used': True,
            }
        except Exception as exc:
            top = results[0]
            answer = clean_assistant_answer(top.get('definition_snippet') or top.get('snippet') or 'Relevant metadata found, but no summary text is available.')
            return {
                'query': query,
                'answer': clean_assistant_answer(f"{answer}\n\nKey sources: {top.get('title') or top.get('citation') or top.get('source')}"),
                'sources': results[:top_k],
                'llm_used': False,
                'fallback_reason': str(exc),
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
            summary = find_matching_summary(chunk, summary_map) or {}
            if key not in papers:
                papers[key] = {
                    'id': key,
                    'source': chunk['source'],
                    'title': summary.get('title') or chunk['title'],
                    'display_title': summary.get('display_title') or chunk.get('display_title') or build_apa_citation(chunk['title'], chunk.get('authors_list') or chunk.get('authors'), chunk.get('year')),
                    'citation': summary.get('citation') or chunk.get('citation') or build_apa_citation(chunk['title'], chunk.get('authors_list') or chunk.get('authors'), chunk.get('year')),
                    'authors': summary.get('authors') or chunk.get('authors'),
                    'authors_list': summary.get('authors_list') or chunk.get('authors_list') or parse_author_list(chunk.get('authors')),
                    'year': summary.get('year') or chunk.get('year'),
                    'doi': summary.get('doi') or chunk.get('doi'),
                    'snippet': summary.get('snippet') or chunk.get('snippet') or '',
                    'obsidian_summary': summary.get('obsidian_summary'),
                    'obsidian_key_findings': summary.get('obsidian_key_findings') or [],
                    'obsidian_note_path': summary.get('obsidian_note_path'),
                    'obsidian_uri': summary.get('obsidian_uri'),
                    'from_obsidian_note': bool(summary.get('from_obsidian_note')),
                    'kind': 'paper',
                    'chunk_count': 1,
                    'has_summary': bool(summary),
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
                'obsidian_summary': summary.get('obsidian_summary'),
                'obsidian_key_findings': summary.get('obsidian_key_findings') or [],
                'obsidian_note_path': summary.get('obsidian_note_path'),
                'obsidian_uri': summary.get('obsidian_uri'),
                'from_obsidian_note': bool(summary.get('from_obsidian_note')),
                'kind': 'summary',
                'chunk_count': 0,
                'has_summary': True,
                'url': summary.get('url') or summary.get('source_ref'),
                'source_ref': summary.get('source_ref') or summary.get('url'),
            }

        existing_dois = {str((p.get('doi') or '')).lower() for p in papers.values() if p.get('doi')}
        for pending in load_tracker_pending_records():
            pending_doi = str((pending.get('doi') or '')).lower()
            pending_key = pending.get('id')
            if pending_key in papers:
                continue
            if pending_doi and pending_doi in existing_dois:
                continue
            papers[pending_key] = pending
            if pending_doi:
                existing_dois.add(pending_doi)

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
        parity = load_paper_parity_state()
        peer = parity.get('peer_reviewed') or {}
        arxiv = parity.get('arxiv') or {}
        rag = parity.get('rag') or {}
        return {
            'status': 'ok',
            'service': 'rag-api-wrapper',
            'version': '2.0.0',
            'mode': 'public_read_only',
            'llm_model': ZAI_MODEL,
            'indexed_papers': len(papers),
            'paper_count': fulltext_papers,
            'fulltext_papers': fulltext_papers,
            'metadata_only_records': metadata_only_records,
            'summary_count': summary_count,
            'collection_count': chunk_count,
            'peer_reviewed_ok': parity.get('status') == 'PASS',
            'peer_reviewed_gdrive_pdfs': peer.get('active_gdrive_pdfs'),
            'peer_reviewed_obsidian_notes': peer.get('obsidian_summary_notes'),
            'parity_gap': peer.get('parity_gap'),
            'arxiv_violations': arxiv.get('tracker_downloaded_policy_violations'),
            'arxiv_workspace_rag': arxiv.get('workspace_rag_entries', rag.get('arxiv_workspace_rag')),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'llm_ready': bool(os.getenv('ZAI_API_KEY') or os.getenv('OPENCLAW_MODELS_PROVIDERS_ZAI_APIKEY')),
        }

    def _tokens(self, text: str) -> list[str]:
        raw = [tok.lower() for tok in TOKEN_RE.findall(text or '') if len(tok) > 1]
        # Filter stopwords to avoid inflating scores on common words
        filtered = [tok for tok in raw if tok not in STOPWORDS]
        # If all tokens were stopwords, fall back to original (minus single chars)
        if not filtered:
            filtered = raw
        expanded = []
        for tok in filtered:
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
        matched_tokens = 0
        joined = ' \n '.join(fields).lower()
        title = fields[0].lower() if fields else ''
        source = fields[-1].lower() if fields else ''
        for token in query_tokens:
            freq = joined.count(token)
            if not freq:
                continue
            matched_tokens += 1
            weight = 1.0
            if token in title:
                weight += 3.0
            if token in source:
                weight += 3.5
            scored += min(6.0, freq) * weight
        # Require at least some token coverage to score well
        if not matched_tokens:
            return 0.0
        coverage = matched_tokens / len(query_tokens)
        phrase = ' '.join(query_tokens)
        if phrase and phrase in joined:
            scored += 5.0
        if phrase and phrase in title:
            scored += 8.0
        if phrase and phrase in source:
            scored += 6.0
        # Normalize into roughly 0..1, weighted by coverage
        denom = max(10.0, len(query_tokens) * 7.0)
        raw_score = min(0.99, scored / denom)
        # Apply coverage penalty: papers matching only 1 of 3 tokens score lower
        return round(raw_score * (0.4 + 0.6 * coverage), 4)

    def _rewrite_followup_query(self, query: str, history: list | None = None) -> str:
        q = (query or '').strip()
        if not q or not history:
            return q
        tokens = self._tokens(q)
        if not any(tok in FOLLOWUP_PRONOUNS for tok in [t.lower() for t in TOKEN_RE.findall(q)]):
            return q
        # Find the most recent user question with strong domain terms
        for msg in reversed(history[-6:]):
            if msg.get('role') != 'user':
                continue
            prev = str(msg.get('content') or '').strip()
            if not prev or prev == q:
                continue
            prev_tokens = self._tokens(prev)
            if len(prev_tokens) < 2:
                continue
            return f"{q} (context: {prev})"
        return q

    def _clean_snippet(self, text: str | None) -> str:
        raw = trim(text or '', 420)
        if not raw:
            return ''
        cleaned = raw.replace('LLM Summary:', '').replace('Objective:', '').replace('Overview', '').strip(' -:')
        cleaned = re.sub(r'\[\[[^\]]+\]\]', '', cleaned)
        cleaned = re.sub(r'(Related Concepts|Citation|Indexed|Source)\s*:?.*', '', cleaned, flags=re.I)
        cleaned = re.sub(r'\b(Related Concepts|Citation|Indexed|Source)\b.*', '', cleaned, flags=re.I)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip(' -:;,.')
        return cleaned

    def _extract_definition_line(self, text: str | None, query_tokens: list[str]) -> str:
        cleaned = self._clean_snippet(text)
        if not cleaned:
            return ''
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        priority = []
        for s in sentences:
            s2 = s.strip()
            low = s2.lower()
            if any(tok in low for tok in query_tokens) and any(key in low for key in [' is ', ' are ', ' refers to ', ' defined as ', ' method', ' interpolation', ' technique']):
                priority.append(s2)
        if priority:
            return trim(priority[0], 260)
        return trim(sentences[0] if sentences else cleaned, 260)

    def _rerank_candidates(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        q = (query or '').strip().lower()
        q_tokens = self._tokens(q)
        reranked = []
        for item in candidates:
            score = float(item.get('score') or 0.0)
            chunk_count = int(item.get('chunk_count') or 0)
            title = str(item.get('title') or '').lower()
            snippet = str(item.get('snippet') or '').lower()
            kind = str(item.get('kind') or '').lower()
            year_text = str(item.get('year') or '')
            year_val = int(year_text) if year_text.isdigit() else 0

            # Strongly prefer papers that have actual full-text chunks / paper records
            if kind == 'paper':
                score += 0.10
            if chunk_count > 0:
                score += 0.22
            else:
                score -= 0.10

            # Prefer title matches for domain / definition questions
            title_hits = sum(1 for tok in q_tokens if tok in title)
            score += min(0.24, title_hits * 0.08)

            # Prefer definitional snippets when user asks "what is / what are"
            if q.startswith('what is') or q.startswith('what are'):
                if any(key in snippet for key in [' is ', ' are ', 'refers to', 'defined as', 'interpolation', 'geostatistical method']):
                    score += 0.10

            # Richer snippet bonus
            if len(snippet) > 180:
                score += 0.03

            # Slight preference to classic foundational sources for definitions
            if year_val and year_val <= 2010:
                score += 0.02

            clean_snippet = self._clean_snippet(item.get('snippet') or '')
            definition_snippet = self._extract_definition_line(item.get('snippet') or '', q_tokens)
            item = {
                **item,
                'score': round(min(0.99, score), 4),
                'snippet': clean_snippet,
                'definition_snippet': definition_snippet,
            }
            reranked.append(item)

        reranked.sort(key=lambda item: (item['score'], int(str(item.get('year') or 0)) if str(item.get('year') or '').isdigit() else 0), reverse=True)
        return reranked[:top_k]

    def search(self, query: str, top_k: int = 10, history: list | None = None) -> list[dict[str, Any]]:
        q = self._rewrite_followup_query((query or '').strip(), history)
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

        deduped: dict[str, dict[str, Any]] = {}
        for item in candidates:
            key = normalize_key(item.get('source') or item.get('title') or item['id'])
            existing = deduped.get(key)
            if not existing or item['score'] > existing['score']:
                deduped[key] = item

        reranked = self._rerank_candidates(q, list(deduped.values()), top_k=max(top_k * 2, 10))
        return reranked[:top_k]

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

    def answer(self, query: str, top_k: int = 5, history: list | None = None) -> dict[str, Any]:
        rewritten_query = self._rewrite_followup_query(query, history)
        results = self.search(rewritten_query, top_k=max(3, min(8, top_k)), history=history)
        if not results:
            return {
                'query': query,
                'answer': 'No relevant papers found in the index for that query.',
                'sources': [],
                'llm_used': False,
            }

        context_parts = []
        for item in results[:top_k]:
            source_line = item.get('citation') or item.get('display_title') or item.get('title') or item.get('source') or 'Unknown source'
            best_snippet = item.get('definition_snippet') or item.get('snippet') or '—'
            context_parts.append(
                f"[Source: {source_line}]\n"
                f"Authors: {item.get('authors') or '—'}\n"
                f"Year: {item.get('year') or '—'}\n"
                f"DOI: {item.get('doi') or '—'}\n"
                f"Evidence: {best_snippet}"
            )
        context = '\n\n---\n\n'.join(context_parts)

        api_key = os.getenv('ZAI_API_KEY') or os.getenv('OPENCLAW_MODELS_PROVIDERS_ZAI_APIKEY')
        if not api_key:
            return {
                'query': query,
                'answer': 'LLM is not active in this environment. Context retrieval works, but the native z.ai API key is not available.',
                'sources': results[:top_k],
                'llm_used': False,
            }

        # Build conversation history for follow-up context
        history_text = ''
        if history:
            history_lines = []
            for msg in history[-6:]:  # Last 6 messages max
                role = msg.get('role', 'user')
                content = str(msg.get('content', ''))[:500]  # Truncate long messages
                history_lines.append(f"{role.upper()}: {content}")
            if history_lines:
                history_text = '\n\nConversation history (for follow-up context):\n' + '\n'.join(history_lines)

        prompt = (
            'You are the public Orebit RAG assistant.\n'
            'Answer using ONLY the provided paper context.\n'
            'If the context is insufficient, say so plainly.\n'
            'Write a crisp, useful answer for a technical reader. Avoid generic filler and avoid sounding like a textbook dump.\n'
            'For definitional questions, give: (1) one-sentence definition, (2) 1-2 key mechanics or assumptions, (3) one practical note if present in context.\n'
            'For follow-up questions, use the conversation history to resolve references.\n'
            'Return plain text only. Do not use markdown bold markers, bullet syntax, code fences, or HTML/XML tags.\n'
            'Do not print raw labels like "Source titles:". End naturally with a short line starting with "Key sources:" followed by 1-3 titles only.\n'
            'Keep the answer compact unless the user explicitly asks for detail.\n\n'
            f'Retrieval query used: {rewritten_query}\n\n'
            f'Context:\n{context}\n'
            f'{history_text}\n\n'
            f'Original user question: {query}\n'
        )

        payload = {
            'model': ZAI_MODEL,
            'messages': [
                {'role': 'system', 'content': 'You answer from indexed academic paper context only.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0.2,
            'max_tokens': 700,
            'thinking': {'type': 'disabled'},
        }
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

        answer = None
        llm_used = True
        fallback_reason = None
        try:
            response = requests.post(ZAI_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            msg = data['choices'][0]['message']
            answer = clean_assistant_answer(msg.get('content') or msg.get('reasoning_content') or '')
        except Exception as exc:
            llm_used = False
            fallback_reason = str(exc)
            top = results[0]
            second = results[1] if len(results) > 1 else None
            lead = trim(top.get('snippet') or '', 420)
            tail = trim(second.get('snippet') or '', 220) if second else ''
            pieces = [
                f"LLM temporarily unavailable ({fallback_reason}).",
                f"From indexed context, {top.get('citation') or top.get('display_title') or top.get('title') or top.get('source') or 'primary source'} indicates that {lead}".rstrip(),
            ]
            if tail:
                pieces.append(f"Supporting source adds: {tail}")
            answer = clean_assistant_answer(' '.join(piece for piece in pieces if piece))

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
    p_answer.add_argument('--history', type=str, default='[]')

    args = parser.parse_args()
    store = SafePaperRagStore()

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
        history = safe_json_load(args.history, [])
        if not isinstance(history, list):
            history = []
        print(json.dumps(store.answer(args.query, args.top_k, history=history), ensure_ascii=False))
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
