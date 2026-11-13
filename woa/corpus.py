import hashlib
import json
import os
import tempfile
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from .sources import MAX_SOURCE_BYTES, fetch_public_text


WORLD_BANK_INDICATORS = {
    "trade": ("NE.TRD.GNFS.ZS", "Trade (% of GDP)"),
    "logistics": ("LP.LPI.OVRL.XQ", "Logistics performance index"),
}
NASA_QUERIES = {
    "aerospace": "aerospace technology",
    "forensics": "forensic science",
}
PUBLIC_TEXT_SOURCES = {
    "bible-kjv": {
        "category": "scripture",
        "url": "https://www.gutenberg.org/cache/epub/10/pg10.txt",
        "host": "www.gutenberg.org",
        "filename": "bible-kjv-gutenberg-10.txt",
        "title": "The King James Version of the Bible",
        "rights": (
            "Project Gutenberg eBook 10; public-domain source text in the US; "
            "the downloaded file retains the Project Gutenberg license notice."
        ),
    },
    "secret-teachings": {
        "category": "esoteric",
        "url": (
            "https://ia800702.us.archive.org/11/items/"
            "the-secret-teachings-of-all-ages-by-manly-p.-hall-1928/"
            "The%20Secret%20Teachings%20of%20All%20Ages%20by%20Manly%20P.%20"
            "Hall%201928_djvu.txt"
        ),
        "host": "ia800702.us.archive.org",
        "filename": "secret-teachings-of-all-ages-1928.txt",
        "title": "The Secret Teachings of All Ages (1928)",
        "rights": (
            "The Internet Archive OCR identifies the 1928 work as copyright "
            "not renewed; public-domain status can vary by jurisdiction."
        ),
    },
    "kabbalah-unveiled": {
        "category": "esoteric",
        "url": (
            "https://ia801505.us.archive.org/21/items/"
            "mathers-s.-l.-mac-gregor-the-kabbalah-unveiled-1887/"
            "Mathers%2C%20S.%20L.%20MacGregor%20-%20The%20Kabbalah%20"
            "Unveiled%20%281887%29_djvu.txt"
        ),
        "host": "ia801505.us.archive.org",
        "filename": "kabbalah-unveiled-mathers-1887.txt",
        "title": "The Kabbalah Unveiled (1887)",
        "rights": (
            "1887 English translation; public-domain-era text. The archive scan "
            "is an OCR transcription and can contain errors."
        ),
    },
    "sepher-yetzirah": {
        "category": "esoteric",
        "url": (
            "https://ia902807.us.archive.org/18/items/"
            "sepheryetzirahb00rittgoog/sepheryetzirahb00rittgoog_djvu.txt"
        ),
        "host": "ia902807.us.archive.org",
        "filename": "sepher-yetzirah-westcott-1893.txt",
        "title": "Sepher Yetzirah (1893)",
        "rights": (
            "Internet Archive metadata marks the 1893 Google Books item "
            "NOT_IN_COPYRIGHT; OCR may contain errors."
        ),
    },
    "republic": {
        "category": "philosophy",
        "url": "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
        "host": "www.gutenberg.org",
        "filename": "plato-republic-gutenberg-1497.txt",
        "title": "The Republic by Plato, translated by Benjamin Jowett",
        "rights": (
            "Project Gutenberg eBook 1497; public-domain source text in the US; "
            "the downloaded file retains the Project Gutenberg license notice."
        ),
        "scope_note": (
            "A historical classical philosophical dialogue; not a modern "
            "reference work."
        ),
    },
    "origin-of-species": {
        "category": "science",
        "url": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
        "host": "www.gutenberg.org",
        "filename": "darwin-origin-of-species-gutenberg-1228.txt",
        "title": "On the Origin of Species by Charles Darwin",
        "rights": (
            "Project Gutenberg eBook 1228; public-domain source text in the US; "
            "the downloaded file retains the Project Gutenberg license notice."
        ),
        "scope_note": (
            "A historical 1859 scientific work; consult current scientific "
            "sources for modern findings."
        ),
    },
    "history-of-art": {
        "category": "art",
        "url": "https://www.gutenberg.org/cache/epub/25632/pg25632.txt",
        "host": "www.gutenberg.org",
        "filename": "history-of-art-gutenberg-25632.txt",
        "title": (
            "A History of Art for Beginners and Students by Clara Erskine Clement"
        ),
        "rights": (
            "Project Gutenberg eBook 25632; public-domain source text in the US; "
            "the downloaded file retains the Project Gutenberg license notice."
        ),
        "scope_note": (
            "A historical survey of painting, sculpture, and architecture; "
            "its interpretations reflect its period."
        ),
    },
    "edison-inventions": {
        "category": "technology",
        "url": "https://www.gutenberg.org/cache/epub/820/pg820.txt",
        "host": "www.gutenberg.org",
        "filename": "edison-life-and-inventions-gutenberg-820.txt",
        "title": "Edison: His Life and Inventions by Dyer and Martin",
        "rights": (
            "Project Gutenberg eBook 820; public-domain source text in the US; "
            "the downloaded file retains the Project Gutenberg license notice."
        ),
        "scope_note": (
            "A historical account of invention and electrical technology, "
            "not a guide to current engineering."
        ),
    },
    "art-of-conversation": {
        "category": "communication",
        "url": "https://www.gutenberg.org/cache/epub/65638/pg65638.txt",
        "host": "www.gutenberg.org",
        "filename": "principles-of-conversation-gutenberg-65638.txt",
        "title": "The Principles of the Art of Conversation by J. P. Mahaffy",
        "rights": (
            "Project Gutenberg eBook 65638; public-domain source text in the US; "
            "the downloaded file retains the Project Gutenberg license notice."
        ),
        "scope_note": (
            "A historical work on listening, tact, and conversation; its social "
            "assumptions reflect its period."
        ),
    },
    "elements-of-style": {
        "category": "language",
        "url": "https://www.gutenberg.org/cache/epub/37134/pg37134.txt",
        "host": "www.gutenberg.org",
        "filename": "elements-of-style-gutenberg-37134.txt",
        "title": "The Elements of Style by William Strunk",
        "rights": (
            "Project Gutenberg eBook 37134; public-domain source text in the US; "
            "the downloaded file retains the Project Gutenberg license notice."
        ),
        "scope_note": (
            "An early 20th-century English prose style guide; its prescriptive "
            "rules are not universal language rules."
        ),
    },
}


def _atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _world_bank_text(indicator_id, display_name):
    url = (
        "https://api.worldbank.org/v2/country/all/indicator/"
        f"{indicator_id}?format=json&per_page=20000"
    )
    payload_text, source_info = fetch_public_text(
        url,
        ["api.worldbank.org"],
        max_bytes=MAX_SOURCE_BYTES,
        timeout=30,
    )
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"World Bank returned invalid JSON for {indicator_id}.") from error
    if (
        not isinstance(payload, list)
        or len(payload) < 2
        or not isinstance(payload[1], list)
    ):
        raise ValueError(f"World Bank returned an unexpected response for {indicator_id}.")
    header = payload[0] if isinstance(payload[0], dict) else {}
    records = payload[1]
    lines = [
        f"World Bank indicator: {display_name} ({indicator_id}).",
        "Source: World Bank Indicators API v2; subject to World Bank terms of use.",
        "Values are reported observations; blank values are omitted.",
        f"API last updated: {header.get('lastupdated', 'not supplied')}.",
        "",
    ]
    for record in records:
        if not isinstance(record, dict) or record.get("value") is None:
            continue
        country = record.get("country") or {}
        country_name = country.get("value", "Unknown")
        country_code = record.get("countryiso3code", "")
        year = record.get("date", "unknown")
        value = record["value"]
        lines.append(
            f"{country_name} ({country_code}), {year}: {display_name} = {value}."
        )
    if len(lines) == 5:
        raise ValueError(f"World Bank returned no observations for {indicator_id}.")
    return "\n".join(lines), source_info


def _nasa_abstract_text(query, max_records=50):
    if not 1 <= max_records <= 100:
        raise ValueError("NASA result limit must be between 1 and 100.")
    url = "https://ntrs.nasa.gov/api/citations/search?" + urllib.parse.urlencode(
        {"q": query, "size": max_records}
    )
    payload_text, source_info = fetch_public_text(
        url, ["ntrs.nasa.gov"], timeout=30
    )
    try:
        payload = json.loads(payload_text)
        results = payload["results"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("NASA NTRS returned an unexpected search response.") from error
    lines = [
        f"NASA Technical Reports Server search: {query}.",
        "Only public-use records without disclosed third-party material are included.",
        "This corpus contains record titles and abstracts, not full report text.",
        "",
    ]
    included = 0
    for result in results:
        if not isinstance(result, dict):
            continue
        copyright_info = result.get("copyright") or {}
        determination = copyright_info.get("determinationType")
        if determination not in {"GOV_PUBLIC_USE_PERMITTED", "PUBLIC_USE_PERMITTED"}:
            continue
        if copyright_info.get("containsThirdPartyMaterial"):
            continue
        if str(result.get("distribution", "")).upper() != "PUBLIC":
            continue
        title = result.get("title")
        abstract = result.get("abstract")
        record_id = result.get("id")
        if not isinstance(title, str) or not isinstance(abstract, str) or not abstract.strip():
            continue
        lines.extend(
            [
                f"NASA NTRS record {record_id}: {title}.",
                f"Source: https://ntrs.nasa.gov/citations/{record_id}",
                abstract.strip(),
                "",
            ]
        )
        included += 1
    if not included:
        raise ValueError(
            f"NASA search {query!r} returned no abstracts meeting the public-use filters."
        )
    source_info["included_records"] = included
    return "\n".join(lines), source_info


def download_starter_corpus(output_dir, categories=None, on_download=None):
    allowed_categories = {
        source["category"] for source in PUBLIC_TEXT_SOURCES.values()
    } | {"economics", "aerospace", "forensics"}
    categories = allowed_categories if categories is None else set(categories)
    unknown = categories - allowed_categories
    if unknown:
        raise ValueError(f"Unknown corpus categories: {', '.join(sorted(unknown))}.")
    output_dir = Path(output_dir)
    downloaded = []
    now = datetime.now(timezone.utc).isoformat()

    def save(
        name,
        category,
        filename,
        title,
        rights,
        text,
        source_info,
        source_url,
        scope_note=None,
    ):
        output_path = output_dir / filename
        _atomic_write(output_path, text)
        record = {
            "name": name,
            "category": category,
            "title": title,
            "url": source_url,
            "rights": rights,
            "path": str(output_path),
            "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source": source_info,
            "downloaded_at": now,
        }
        if scope_note:
            record["scope_note"] = scope_note
        downloaded.append(record)
        if on_download:
            on_download(record)

    for name in sorted(PUBLIC_TEXT_SOURCES):
        source = PUBLIC_TEXT_SOURCES[name]
        category = source["category"]
        if category not in categories:
            continue
        text, source_info = fetch_public_text(
            source["url"],
            [source["host"]],
            max_bytes=16 * 1024 * 1024,
            timeout=30,
        )
        save(
            name,
            category,
            source["filename"],
            source["title"],
            source["rights"],
            text,
            source_info,
            source["url"],
            source.get("scope_note"),
        )

    if "economics" in categories:
        for name, (indicator_id, display_name) in WORLD_BANK_INDICATORS.items():
            text, source_info = _world_bank_text(indicator_id, display_name)
            save(
                f"world-bank-{name}",
                "economics",
                f"world-bank-{name}-{indicator_id.lower()}.txt",
                display_name,
                "World Bank API data; retain source attribution and follow its terms.",
                text,
                source_info,
                (
                    "https://api.worldbank.org/v2/country/all/indicator/"
                    f"{indicator_id}?format=json&per_page=20000"
                ),
            )

    for category in ("aerospace", "forensics"):
        if category not in categories:
            continue
        text, source_info = _nasa_abstract_text(NASA_QUERIES[category])
        save(
            f"nasa-{category}-abstracts",
            category,
            f"nasa-{category}-public-abstracts.txt",
            f"NASA NTRS {category} abstracts",
            (
                "NASA NTRS records filtered to public distribution and public-use "
                "determination with no disclosed third-party material; abstracts only."
            ),
            text,
            source_info,
            "https://ntrs.nasa.gov/api/citations/search?q="
            + urllib.parse.quote(NASA_QUERIES[category]),
        )

    manifest_path = output_dir / "sources.json"
    existing = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing = manifest.get("sources", [])
        except (json.JSONDecodeError, OSError, AttributeError) as error:
            raise RuntimeError(f"Could not read corpus source manifest {manifest_path}.") from error
    records_by_name = {
        record["name"]: record
        for record in [*existing, *downloaded]
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    _atomic_write(
        manifest_path,
        json.dumps(
            {"updated_at": now, "sources": list(records_by_name.values())},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )
    return downloaded
