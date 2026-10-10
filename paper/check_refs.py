"""Check every cited reference against its source record.

For each key cited in main.aux: the arXiv abstract page must open, and the arXiv API record must
match the BibTeX title, first-author surname and (preprint) year. Entries with a DOI are checked
against Crossref. Venue claims are cross-checked with Semantic Scholar. Writes ref_check.md.
"""
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "ref-check/1.0 (mailto:hamzanaeem588@gmail.com)"}


def get(url, accept=None):
    req = urllib.request.Request(url, headers={**UA, **({"Accept": accept} if accept else {})})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return e.code, ""
        except Exception:
            time.sleep(3)
    return 0, ""


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[{}\\$]", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def parse_bib(text):
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", text, re.S):
        kind, key, body = m.groups()
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body):
            fields[fm.group(1).lower()] = " ".join(fm.group(2).split())
        entries[key] = {"type": kind.lower(), **fields}
    return entries


def surname(author_field):
    first = author_field.split(" and ")[0].strip()
    return norm(first.split(",")[0] if "," in first else first.split()[-1])


def main():
    cited = sorted({k for m in re.finditer(r"\\citation\{([^}]*)\}", (HERE / "main.aux").read_text())
                    for k in m.group(1).split(",")})
    bib = parse_bib((HERE / "refs.bib").read_text(encoding="utf-8"))
    rows = []
    for key in cited:
        e = bib.get(key)
        if e is None:
            rows.append((key, "MISSING", "not in refs.bib", "", ""))
            continue
        aid = e.get("eprint") or (re.search(r"(\d{4}\.\d{4,5})", e.get("journal", "") + e.get("url", "")) or [None])[0]
        notes, status = [], "OK"
        if aid:
            code, _ = get(f"https://arxiv.org/abs/{aid}")
            notes.append(f"arXiv page {code}")
            if code != 200:
                status = "FAIL"
            _, xml = get(f"https://export.arxiv.org/api/query?id_list={aid}")
            titles = re.findall(r"<title>(.*?)</title>", xml, re.S)
            authors = re.findall(r"<name>(.*?)</name>", xml)
            pub = re.findall(r"<published>(\d{4})", xml)
            if len(titles) < 2:
                status = "FAIL"
                notes.append("no arXiv record")
            else:
                t_ok = norm(titles[1]) == norm(e.get("title", ""))
                a_ok = bool(authors) and surname(e.get("author", "")) in norm(authors[0])
                notes.append(f"title {'match' if t_ok else 'MISMATCH: ' + ' '.join(titles[1].split())}")
                notes.append(f"1st author {'match' if a_ok else 'MISMATCH: ' + (authors[0] if authors else '?')}")
                if pub:
                    notes.append(f"arXiv {pub[0]} / bib {e.get('year', '?')}")
                if not (t_ok and a_ok):
                    status = "CHECK"
            time.sleep(3)  # arXiv API etiquette
        if e.get("doi"):
            code, js = get(f"https://api.crossref.org/works/{e['doi']}")
            if code == 200:
                msg = json.loads(js)["message"]
                ct = msg.get("title", [""])[0]
                t_ok = norm(ct) == norm(e.get("title", ""))
                notes.append(f"Crossref DOI {'title match' if t_ok else 'title MISMATCH: ' + ct}")
                if not t_ok:
                    status = "CHECK"
            else:
                notes.append(f"Crossref DOI lookup {code}")
                status = "FAIL"
        if not aid and not e.get("doi"):
            notes.append("no arXiv ID or DOI to check")
            status = "CHECK"
        venue = e.get("booktitle") or e.get("journal", "")
        if venue and "arxiv" not in venue.lower():
            q = urllib.parse.quote(e.get("title", ""))
            code, js = get(f"https://api.semanticscholar.org/graph/v1/paper/search/match?query={q}&fields=venue,year,publicationVenue")
            if code == 200:
                d = (json.loads(js).get("data") or [{}])[0]
                notes.append(f"S2 venue: {d.get('venue') or (d.get('publicationVenue') or {}).get('name') or '?'} {d.get('year', '')}")
            else:
                notes.append(f"S2 lookup {code}")
            time.sleep(1.5)
        rows.append((key, status, "; ".join(notes), e.get("title", ""), venue))
        print(key, status, "|", "; ".join(notes), flush=True)
    lines = ["# Reference check", "", f"{len(cited)} cited keys.", "",
             "| Key | Status | Checks | BibTeX venue |", "|---|---|---|---|"]
    lines += [f"| {k} | {s} | {n} | {v} |" for k, s, n, _, v in rows]
    (HERE / "ref_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import urllib.parse
    main()
