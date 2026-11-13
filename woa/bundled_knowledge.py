import hashlib
from importlib.resources import files

from .knowledge import KnowledgeIndex


BUNDLED_SOURCES = (
    {
        "filename": "plato-republic-gutenberg-1497.txt",
        "title": "The Republic by Plato, translated by Benjamin Jowett",
        "url": "https://www.gutenberg.org/ebooks/1497",
        "rights": (
            "Project Gutenberg eBook 1497; public-domain source text in the US; "
            "the text retains the Project Gutenberg license notice."
        ),
        "scope_note": "Historical classical philosophy, not a modern reference work.",
    },
    {
        "filename": "darwin-origin-of-species-gutenberg-1228.txt",
        "title": "On the Origin of Species by Charles Darwin",
        "url": "https://www.gutenberg.org/ebooks/1228",
        "rights": (
            "Project Gutenberg eBook 1228; public-domain source text in the US; "
            "the text retains the Project Gutenberg license notice."
        ),
        "scope_note": (
            "Historical scientific work; consult current sources for modern findings."
        ),
    },
    {
        "filename": "history-of-art-gutenberg-25632.txt",
        "title": "A History of Art for Beginners and Students by Clara Erskine Clement",
        "url": "https://www.gutenberg.org/ebooks/25632",
        "rights": (
            "Project Gutenberg eBook 25632; public-domain source text in the US; "
            "the text retains the Project Gutenberg license notice."
        ),
        "scope_note": "Historical art survey; interpretations reflect its period.",
    },
    {
        "filename": "edison-life-and-inventions-gutenberg-820.txt",
        "title": "Edison: His Life and Inventions by Dyer and Martin",
        "url": "https://www.gutenberg.org/ebooks/820",
        "rights": (
            "Project Gutenberg eBook 820; public-domain source text in the US; "
            "the text retains the Project Gutenberg license notice."
        ),
        "scope_note": "Historical account, not a guide to current engineering.",
    },
    {
        "filename": "principles-of-conversation-gutenberg-65638.txt",
        "title": "The Principles of the Art of Conversation by J. P. Mahaffy",
        "url": "https://www.gutenberg.org/ebooks/65638",
        "rights": (
            "Project Gutenberg eBook 65638; public-domain source text in the US; "
            "the text retains the Project Gutenberg license notice."
        ),
        "scope_note": "Historical social guidance; assumptions reflect its period.",
    },
    {
        "filename": "elements-of-style-gutenberg-37134.txt",
        "title": "The Elements of Style by William Strunk",
        "url": "https://www.gutenberg.org/ebooks/37134",
        "rights": (
            "Project Gutenberg eBook 37134; public-domain source text in the US; "
            "the text retains the Project Gutenberg license notice."
        ),
        "scope_note": "Historical prose guide; rules are not universal language rules.",
    },
    {
        "filename": "nasa-aerospace-public-abstracts.txt",
        "title": "NASA NTRS aerospace abstracts",
        "url": "https://ntrs.nasa.gov/api/citations/search?q=aerospace%20technology",
        "rights": (
            "NASA NTRS records filtered to public distribution and public-use "
            "determination with no disclosed third-party material; abstracts only."
        ),
        "scope_note": "Selected records, not full reports or comprehensive coverage.",
    },
    {
        "filename": "nasa-forensics-public-abstracts.txt",
        "title": "NASA NTRS forensic-science abstracts",
        "url": "https://ntrs.nasa.gov/api/citations/search?q=forensic%20science",
        "rights": (
            "NASA NTRS records filtered to public distribution and public-use "
            "determination with no disclosed third-party material; abstracts only."
        ),
        "scope_note": "Selected records, not full reports or comprehensive coverage.",
    },
)


def seed_bundled_knowledge(index):
    if not isinstance(index, KnowledgeIndex):
        raise TypeError("Bundled knowledge requires a KnowledgeIndex.")
    existing = {
        document["source"]: document["sha256"]
        for document in index.list_documents()
    }
    resource_directory = (
        files("woa").joinpath("data").joinpath("knowledge")
    )
    added = []
    for source in BUNDLED_SOURCES:
        filename = source["filename"]
        source_label = f"bundled/{filename}"
        try:
            text = resource_directory.joinpath(filename).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                f"Could not load bundled knowledge source {filename}."
            ) from error
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if existing.get(source_label) == content_hash:
            continue
        added.append(index.ingest_text(source_label, text))
        existing[source_label] = content_hash
    return added
