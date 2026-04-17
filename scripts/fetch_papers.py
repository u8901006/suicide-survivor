#!/usr/bin/env python3
"""
Fetch latest Suicide Bereavement / Survivors of Suicide Loss research papers
from PubMed E-utilities API.
Targets suicide-specific, bereavement/grief, psychiatry, and public health journals.
"""

import json
import sys
import re
import argparse
import glob
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote_plus

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

JOURNALS = [
    "Suicide Life Threat Behav",
    "Arch Suicide Res",
    "Crisis",
    "Omega (Westport)",
    "Death Stud",
    "J Loss Trauma",
    "J Affect Disord",
    "Psychiatry Res",
    "BMC Psychiatry",
    "Psychol Med",
    "J Psychiatr Res",
    "Int J Soc Psychiatry",
    "Soc Psychiatry Psychiatr Epidemiol",
    "Community Ment Health J",
    "Front Psychiatry",
    "Front Psychol",
    "Soc Sci Med",
    "SSM Popul Health",
    "BMC Public Health",
    "BMJ Open",
    "PLoS One",
    "Int J Environ Res Public Health",
]

HEADERS = {"User-Agent": "SuicideSurvivorBot/1.0 (research aggregator)"}


def load_seen_pmids(docs_dir: str = "docs", lookback_days: int = 7) -> set[str]:
    seen = set()
    if not os.path.isdir(docs_dir):
        return seen
    pattern = os.path.join(docs_dir, "survivor-*.html")
    tz_taipei = timezone(timedelta(hours=8))
    cutoff = (datetime.now(tz_taipei) - timedelta(days=lookback_days)).date()
    for filepath in sorted(glob.glob(pattern), reverse=True):
        basename = os.path.basename(filepath)
        date_str = basename.replace("survivor-", "").replace(".html", "")
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < cutoff:
            break
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            pmids = re.findall(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/", content)
            seen.update(pmids)
            print(
                f"[INFO] {basename}: found {len(pmids)} existing PMIDs",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[WARN] Could not read {filepath}: {e}", file=sys.stderr)
    print(
        f"[INFO] Total seen PMIDs in last {lookback_days} days: {len(seen)}",
        file=sys.stderr,
    )
    return seen


def build_core_query() -> str:
    return (
        '("suicide bereavement"[tiab] '
        'OR "bereavement by suicide"[tiab] '
        'OR "suicide-related bereavement"[tiab] '
        'OR "suicide loss"[tiab] '
        'OR "survivors of suicide loss"[tiab] '
        'OR "suicide-loss survivors"[tiab] '
        'OR "people bereaved by suicide"[tiab] '
        'OR "suicide exposure"[tiab] '
        'OR "suicide-exposed"[tiab] '
        "OR ((bereave*[tiab] OR grief[tiab] OR grieving[tiab]) AND suicid*[tiab]))"
    )


def build_journal_queries(days: int = 7) -> list[str]:
    lookback = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y/%m/%d")
    date_part = f'"{lookback}"[Date - Publication] : "3000"[Date - Publication]'
    core = build_core_query()
    queries = []
    journal_batch_size = 5
    for i in range(0, len(JOURNALS), journal_batch_size):
        batch = JOURNALS[i : i + journal_batch_size]
        journal_part = " OR ".join([f'"{j}"[jour]' for j in batch])
        queries.append(f"({core}) AND ({journal_part}) AND {date_part}")
    return queries


def search_papers(query: str, retmax: int = 50) -> list[str]:
    params = (
        f"?db=pubmed&term={quote_plus(query)}&retmax={retmax}&sort=date&retmode=json"
    )
    url = PUBMED_SEARCH + params
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"[ERROR] PubMed search failed: {e}", file=sys.stderr)
        return []


def fetch_details(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    ids = ",".join(pmids)
    params = f"?db=pubmed&id={ids}&retmode=xml"
    url = PUBMED_FETCH + params
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=60) as resp:
            xml_data = resp.read().decode()
    except Exception as e:
        print(f"[ERROR] PubMed fetch failed: {e}", file=sys.stderr)
        return []

    papers = []
    try:
        root = ET.fromstring(xml_data)
        for article in root.findall(".//PubmedArticle"):
            medline = article.find(".//MedlineCitation")
            art = medline.find(".//Article") if medline else None
            if art is None:
                continue

            title_el = art.find(".//ArticleTitle")
            title = (
                (title_el.text or "").strip()
                if title_el is not None and title_el.text
                else ""
            )

            abstract_parts = []
            for abs_el in art.findall(".//Abstract/AbstractText"):
                label = abs_el.get("Label", "")
                text = "".join(abs_el.itertext()).strip()
                if label and text:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)[:2000]

            journal_el = art.find(".//Journal/Title")
            journal = (
                (journal_el.text or "").strip()
                if journal_el is not None and journal_el.text
                else ""
            )

            pub_date = art.find(".//PubDate")
            date_str = ""
            if pub_date is not None:
                year = pub_date.findtext("Year", "")
                month = pub_date.findtext("Month", "")
                day = pub_date.findtext("Day", "")
                parts = [p for p in [year, month, day] if p]
                date_str = " ".join(parts)

            pmid_el = medline.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            keywords = []
            for kw in medline.findall(".//KeywordList/Keyword"):
                if kw.text:
                    keywords.append(kw.text.strip())

            authors = []
            for author in art.findall(".//AuthorList/Author")[:6]:
                last = author.findtext("LastName", "")
                fore = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{last} {fore}".strip())
            if len(art.findall(".//AuthorList/Author")) > 6:
                authors.append("et al.")

            papers.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "authors": "; ".join(authors),
                    "journal": journal,
                    "date": date_str,
                    "abstract": abstract,
                    "url": link,
                    "keywords": keywords,
                }
            )
    except ET.ParseError as e:
        print(f"[ERROR] XML parse failed: {e}", file=sys.stderr)

    return papers


def main():
    parser = argparse.ArgumentParser(
        description="Fetch suicide bereavement papers from PubMed"
    )
    parser.add_argument("--days", type=int, default=7, help="Lookback days")
    parser.add_argument(
        "--max-papers", type=int, default=100, help="Max papers to fetch"
    )
    parser.add_argument("--output", default="-", help="Output file (- for stdout)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--docs-dir", default="docs", help="Directory with existing reports for dedup"
    )
    args = parser.parse_args()

    seen_pmids = load_seen_pmids(args.docs_dir, lookback_days=7)

    queries = build_journal_queries(days=args.days)
    all_pmids = []
    per_query = max(args.max_papers // len(queries), 20)

    for i, query in enumerate(queries):
        print(
            f"[INFO] Searching batch {i + 1}/{len(queries)}...",
            file=sys.stderr,
        )
        pmids = search_papers(query, retmax=per_query)
        all_pmids.extend(pmids)
        print(f"[INFO] Batch {i + 1}: found {len(pmids)} PMIDs", file=sys.stderr)

    seen_set = set()
    unique_pmids = []
    for p in all_pmids:
        if p not in seen_set:
            seen_set.add(p)
            unique_pmids.append(p)
    all_pmids = unique_pmids[: args.max_papers]

    print(
        f"[INFO] Total unique PMIDs from PubMed: {len(all_pmids)}",
        file=sys.stderr,
    )

    if seen_pmids:
        new_pmids = [p for p in all_pmids if p not in seen_pmids]
        skipped = len(all_pmids) - len(new_pmids)
        print(
            f"[INFO] Dedup: skipped {skipped} already-seen papers, {len(new_pmids)} new",
            file=sys.stderr,
        )
        all_pmids = new_pmids

    if not all_pmids:
        print("NO_CONTENT", file=sys.stderr)
        if args.json:
            print(
                json.dumps(
                    {
                        "date": datetime.now(timezone(timedelta(hours=8))).strftime(
                            "%Y-%m-%d"
                        ),
                        "count": 0,
                        "papers": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return

    papers = fetch_details(all_pmids)
    print(f"[INFO] Fetched details for {len(papers)} papers", file=sys.stderr)

    output_data = {
        "date": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d"),
        "count": len(papers),
        "papers": papers,
    }

    out_str = json.dumps(output_data, ensure_ascii=False, indent=2)

    if args.output == "-":
        print(out_str)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_str)
        print(f"[INFO] Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
