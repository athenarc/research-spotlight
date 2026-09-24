from __future__ import annotations

import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def read_jsonl(path: Path):
    import srsly

    return list(srsly.read_jsonl(str(path)))


def write_jsonl(path: Path, rows):
    import srsly

    path.parent.mkdir(parents=True, exist_ok=True)
    srsly.write_jsonl(str(path), rows)


def _is_missing_value(value: Any) -> bool:
    """Return True for empty/placeholder values used by external linkers."""
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}

    return False


def normalize_metadata(path: Path, pdf_dir: Path):
    rows = read_jsonl(path)
    normalized = []

    for row in rows:
        meta = row.get("meta", row)
        meta = dict(meta)

        if "author" not in meta:
            meta["author"] = meta.get("authors", []) or []

        if "authors" not in meta:
            meta["authors"] = meta.get("author", []) or []

        aliases = {
            "DOI": "doi",
            "page_start": "pageStart",
            "page_end": "pageEnd",
        }

        for old, new in aliases.items():
            if old in meta and new not in meta:
                meta[new] = meta[old]

        row = dict(row)
        row["meta"] = meta
        normalized.append(row)

    write_jsonl(path, normalized)
    return normalized


NER_MODEL_NAMES = {
    "method": "en_deberta_v3_base_ner_method",
    "activity": "en_deberta_v3_base_ner_activity",
    "goal": "en_deberta_v3_base_ner_goal",
}


def ensure_models_exist():
    """Accept the original notebook's local model directories or bundled spaCy packages."""
    import importlib.util

    root = Path(os.getenv("MODEL_ROOT", "/models"))
    missing = []
    resolved = {}

    for key, package_name in NER_MODEL_NAMES.items():
        local_dir = root / package_name

        if local_dir.exists():
            resolved[key] = str(local_dir)
        elif importlib.util.find_spec(package_name):
            resolved[key] = package_name
        else:
            missing.append(package_name)

    if missing:
        raise FileNotFoundError(
            "Missing NER models. Expected bundled spaCy packages or local model directories: "
            + ", ".join(missing)
        )

    return resolved


def text_extraction(run_dir: Path):
    from glmocr import GlmOcr
    from wtpsplit import SaT

    pdf_dir = run_dir / "PDF"
    out_dir = run_dir / "Text_Extraction"
    out_dir.mkdir(exist_ok=True)

    out_file = out_dir / "input_data.jsonl"

    if out_file.exists():
        out_file.unlink()

    metadata = normalize_metadata(
        pdf_dir / "metadata.jsonl",
        pdf_dir,
    )

    pdfs = sorted(
        [p for p in pdf_dir.glob("*.pdf")],
        key=lambda p: (
            int(p.stem)
            if p.stem.isdigit()
            else p.stem
        ),
    )

    cfg_path = os.getenv(
        "GLMOCR_CONFIG_PATH",
        "/app/dependencies/config.yaml",
    )

    sat = SaT("sat-12l-sm")

    all_rows = []

    for pdf in pdfs:
        matching = [
            m
            for m in metadata
            if str(
                m.get("meta", {}).get("id")
            ) == pdf.stem
        ]

        if not matching:
            continue

        meta = matching[0]

        with GlmOcr(
            config_path=cfg_path,
            mode="selfhosted",
            log_level="ERROR",
        ) as parser:
            result = parser.parse(str(pdf))

            final_text = [
                text.get("content")
                for element in result.json_result
                for text in element
                if text
            ]

        extracted_text = sat.split(final_text)

        # Preserve sentence order while removing duplicates.
        cleaned = []
        seen = set()

        for text in extracted_text:
            for element in text:
                sentence = element.strip()

                if (
                    sentence
                    and sentence not in seen
                ):
                    seen.add(sentence)
                    cleaned.append(sentence)

        for index, sentence in enumerate(cleaned):
            if len(sentence.split()) < 9:
                continue

            row = json.loads(json.dumps(meta))
            row["text"] = sentence
            row.setdefault("meta", {})[
                "sent_no"
            ] = f"{pdf.stem}/{index}"

            all_rows.append(row)

    write_jsonl(out_file, all_rows)

    return {
        "artifact": str(
            out_file.relative_to(run_dir)
        ),
        "rows": len(all_rows),
    }


def _load_ner(model_path: str | Path):
    import spacy

    return spacy.load(str(model_path))


def _ner_rows(
    rows,
    model_path: Path,
    entity: str,
):
    ner_model = _load_ner(model_path)
    annotated = []

    for row in rows:
        doc = ner_model(row["text"])

        ner_spans = [
            {
                "start": span.start_char,
                "end": span.end_char,
                "label": entity,
                "start_token": span.start,
                "end_token": span.end,
            }
            for span in doc.ents
        ]

        row = dict(row)
        row["spans"] = (
            row.get("spans", []) + ner_spans
        )
        row["_annotator_id"] = "NER"
        row["_session_id"] = "NER"
        row["tokens"] = [
            token.text
            for token in doc
        ]

        annotated.append(row)

    return annotated


def entity_extraction(run_dir: Path):
    models = ensure_models_exist()

    rows = read_jsonl(
        run_dir
        / "Text_Extraction"
        / "input_data.jsonl"
    )

    rows = _ner_rows(
        rows,
        models["method"],
        "METHOD",
    )

    rows = _ner_rows(
        rows,
        models["activity"],
        "ACTIVITY",
    )

    rows = _ner_rows(
        rows,
        models["goal"],
        "GOAL",
    )

    out = (
        run_dir
        / "Entity_Extraction"
        / "entity_extraction.jsonl"
    )

    write_jsonl(out, rows)

    return {
        "artifact": str(
            out.relative_to(run_dir)
        ),
        "rows": len(rows),
        "entities": sum(
            len(r.get("spans", []))
            for r in rows
        ),
    }


def entity_disambiguation(run_dir: Path):
    import requests
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )
    from dependencies.trie import Trie

    genre_model_path = os.getenv(
        "GENRE_MODEL_PATH",
        "/models/genre-linking-blink",
    )

    trie_path = (
        "/app/dependencies/"
        "kilt_titles_trie_dict.pkl"
    )

    with open(trie_path, "rb") as f:
        trie = Trie.load_from_dict(
            pickle.load(f)
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            genre_model_path,
            local_files_only=True,
        )
    )

    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            genre_model_path,
            local_files_only=True,
        )
        .eval()
    )

    def safe_prefix_fn(batch_id, sent):
        allowed = trie.get(sent.tolist())

        if not allowed:
            return [
                tokenizer.eos_token_id
            ]

        return allowed

    def disambiguate(text, start, end):
        """
        Resolve a METHOD mention to a Wikipedia page.

        The notebook originally returned only the Wikipedia URL.
        We additionally retain the generated title and a status so
        later Entity Linking can distinguish unresolved Wikipedia
        pages from failures in the metadata lookup.
        """
        mention = text[start:end]

        try:
            sentence = [
                text[:start]
                + "[START_ENT] "
                + mention
                + " [END_ENT]"
                + text[end:]
            ]

            outputs = model.generate(
                **tokenizer(
                    sentence,
                    return_tensors="pt",
                ),
                num_beams=5,
                num_return_sequences=1,
                prefix_allowed_tokens_fn=
                    safe_prefix_fn,
            )

            title = (
                tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                )[0]
                .strip()
            )

            if not title:
                return {
                    "wikipedia_url": None,
                    "wikipedia_title": None,
                    "disambiguation_status":
                        "unresolved",
                    "disambiguation_error":
                        "GENRE returned an empty title.",
                }

            api_url = (
                "https://en.wikipedia.org/"
                "w/api.php"
            )

            params = {
                "action": "query",
                "prop": "info",
                "inprop": "subjectid",
                "titles": title,
                "redirects": 1,
                "format": "json",
                "formatversion": 2,
            }

            headers = {"User-Agent": f"Random_User/1.0 (https://researchspotlight.org/random_user/; random_user@researchspotlight.org) information-linking-library/1.0"}

            response = requests.get(
                api_url,
                params=params,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()

            payload = response.json()

            pages = (
                payload
                .get("query", {})
                .get("pages", [])
            )

            if isinstance(pages, dict):
                pages = list(pages.values())

            valid_page = None

            for page in pages:
                if page.get("missing"):
                    continue

                if page.get("pageid") is not None:
                    valid_page = page
                    break

            if valid_page is None:
                return {
                    "wikipedia_url": None,
                    "wikipedia_title": title,
                    "disambiguation_status":
                        "unresolved",
                    "disambiguation_error":
                        f'Wikipedia page not found for GENRE title "{title}".',
                }

            page_id = valid_page.get(
                "pageid"
            )

            canonical_title = (
                valid_page.get("title")
                or title
            )

            return {
                "wikipedia_url":
                    f"https://en.wikipedia.org/wiki?curid={page_id}",
                "wikipedia_title":
                    canonical_title,
                "disambiguation_status":
                    "resolved",
                "disambiguation_error":
                    None,
            }

        except Exception as exc:
            logger.warning(
                "Wikipedia disambiguation failed for %r: %s",
                mention,
                exc,
            )

            return {
                "wikipedia_url": None,
                "wikipedia_title": None,
                "disambiguation_status":
                    "error",
                "disambiguation_error":
                    str(exc),
            }

    data = read_jsonl(
        run_dir
        / "Entity_Extraction"
        / "entity_extraction.jsonl"
    )

    for item in data:
        text = item.get("text", "")

        for span in item.get("spans", []):
            if span.get("label") != "METHOD":
                continue

            result = disambiguate(
                text,
                int(span.get("start", 0)),
                int(span.get("end", 0)),
            )

            span.update(result)

    out = (
        run_dir
        / "Entity_Disambiguation"
        / "entity_disambiguation.jsonl"
    )

    write_jsonl(out, data)

    return {
        "artifact": str(
            out.relative_to(run_dir)
        ),
        "rows": len(data),
    }


def entity_linking(run_dir: Path):
    from information_linking_queries.information_linking_orcid import (
        information_linking_orcid,
    )
    from information_linking_queries.information_linking_apis import (
        information_linking,
    )

    data = read_jsonl(
        run_dir
        / "Entity_Disambiguation"
        / "entity_disambiguation.jsonl"
    )

    headers = {"User-Agent": f"Random_User/1.0 (https://researchspotlight.org/random_user/; random_user@researchspotlight.org) information-linking-library/1.0"}

    for row in data:
        author_list = []

        for author in row.get(
            "meta", {}
        ).get("author", []):
            if not author:
                continue

            parts = author.split(
                maxsplit=1
            )

            f_name = parts[0].strip()
            l_name = parts[-1].strip()

            try:
                info = information_linking_orcid(
                    f_name,
                    l_name,
                )

                author_list.append(
                    {
                        "full_name": author,
                        "given_name": info[
                            "given-names"
                        ],
                        "family_name": info[
                            "family-names"
                        ],
                        "orcid": info[
                            "orcid-id"
                        ],
                        "affiliations": info.get(
                            "institution-name",
                            [],
                        ),
                    }
                )

            except Exception:
                author_list.append(
                    {
                        "full_name": author,
                        "given_name": f_name.capitalize(),
                        "family_name": l_name.capitalize(),
                        "orcid": "None",
                        "affiliations": "None",
                    }
                )

        row.setdefault(
            "meta", {}
        )["creator"] = author_list

        for label in row.get(
            "spans", []
        ):
            if label.get("label") != "METHOD":
                continue

            start = int(
                label.get("start", 0)
            )
            end = int(
                label.get("end", 0)
            )

            text = row.get(
                "text", ""
            )

            mention = text[start:end]

            wikipedia_url = label.get(
                "wikipedia_url"
            )

            wikipedia_title = label.get(
                "wikipedia_title"
            )

            # ----------------------------------------------------
            # Case 1: Entity Disambiguation could not resolve a
            # Wikipedia page.
            # ----------------------------------------------------
            if _is_missing_value(
                wikipedia_url
            ):
                fallback_name = (
                    wikipedia_title
                    if not _is_missing_value(
                        wikipedia_title
                    )
                    else mention
                )

                label["proper_name"] = (
                    fallback_name
                )
                label["description"] = None
                label["wikidata_url"] = None
                label["dbpedia_url"] = None
                label["linking_status"] = (
                    "unresolved"
                )
                label["linking_error"] = (
                    label.get(
                        "disambiguation_error"
                    )
                    or "Entity disambiguation did not resolve a Wikipedia page."
                )
                label["entity_name_source"] = (
                    "wikipedia_title"
                    if not _is_missing_value(
                        wikipedia_title
                    )
                    else "mention"
                )

                logger.warning(
                    "No Wikipedia page resolved for METHOD %r."
                    " title=%r error=%r",
                    mention,
                    wikipedia_title,
                    label.get(
                        "disambiguation_error"
                    ),
                )

                continue

            # ----------------------------------------------------
            # Case 2: Wikipedia resolved, but the later metadata
            # lookup fails or does not provide a canonical label.
            # ----------------------------------------------------
            try:
                info = information_linking(
                    wikipedia_url=wikipedia_url,
                    headers=headers,
                    dbpedia=False
                )

                resolved_name = info.get(
                    "label"
                )

                if _is_missing_value(
                    resolved_name
                ):
                    if not _is_missing_value(
                        wikipedia_title
                    ):
                        resolved_name = (
                            wikipedia_title
                        )
                        name_source = (
                            "wikipedia_title"
                        )
                    else:
                        resolved_name = mention
                        name_source = (
                            "mention"
                        )

                    linking_status = (
                        "fallback"
                    )

                    linking_error = (
                        "Wikipedia metadata lookup returned no canonical label."
                    )

                else:
                    name_source = (
                        "information_linking"
                    )
                    linking_status = (
                        "linked"
                    )
                    linking_error = None

                description = info.get(
                    "description"
                )

                wikidata_url = info.get(
                    "wikidata"
                )

                dbpedia_url = info.get(
                    "dbpedia"
                )

                label["description"] = (
                    description
                    if not _is_missing_value(
                        description
                    )
                    else None
                )

                label["proper_name"] = (
                    resolved_name
                )

                label["wikidata_url"] = (
                    wikidata_url
                    if not _is_missing_value(
                        wikidata_url
                    )
                    else None
                )

                label["dbpedia_url"] = (
                    dbpedia_url
                    if not _is_missing_value(
                        dbpedia_url
                    )
                    else None
                )

                label["linking_status"] = (
                    linking_status
                )

                label["linking_error"] = (
                    linking_error
                )

                label["entity_name_source"] = (
                    name_source
                )

            except Exception as exc:
                # Never write literal "None" for the method name.
                # Preserve the best available title or mention.
                if not _is_missing_value(
                    wikipedia_title
                ):
                    fallback_name = (
                        wikipedia_title
                    )
                    name_source = (
                        "wikipedia_title"
                    )
                else:
                    fallback_name = mention
                    name_source = "mention"

                label["proper_name"] = (
                    fallback_name
                )
                label["description"] = None
                label["wikidata_url"] = None
                label["dbpedia_url"] = None
                label["linking_status"] = (
                    "fallback"
                )
                label["linking_error"] = (
                    str(exc)
                )
                label["entity_name_source"] = (
                    name_source
                )

                logger.warning(
                    "Entity linking failed for METHOD %r"
                    " using Wikipedia URL %r: %s",
                    mention,
                    wikipedia_url,
                    exc,
                )

    out = (
        run_dir
        / "Entity_Linking"
        / "entity_linking.jsonl"
    )

    write_jsonl(out, data)

    return {
        "artifact": str(
            out.relative_to(run_dir)
        ),
        "rows": len(data),
    }


def _overlap(
    a_start,
    a_end,
    b_start,
    b_end,
):
    return max(
        a_start,
        b_start,
    ) < min(
        a_end,
        b_end,
    )


def _relation_span(
    span,
    text: str,
):
    """
    Return a validated relation endpoint.

    spaCy/Python character offsets are Unicode code-point offsets.
    Keep those offsets unchanged and include the exact text slice in
    the relation so the frontend can validate that it is drawing the
    correct characters.
    """
    try:
        start = int(
            span.get("start")
        )
        end = int(
            span.get("end")
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if (
        start < 0
        or end <= start
        or end > len(text)
    ):
        return None

    return {
        "start": start,
        "end": end,
        "label": span.get(
            "label"
        ),
        "text": text[
            start:end
        ],
    }


def _relation_employs(
    spans,
    text: str,
):
    acts = [
        span
        for span in spans
        if span.get("label") == "ACTIVITY"
    ]

    methods = [
        span
        for span in spans
        if span.get("label") == "METHOD"
    ]

    result = []

    for activity in acts:
        activity_ref = _relation_span(
            activity,
            text,
        )

        if activity_ref is None:
            continue

        for method in methods:
            method_ref = _relation_span(
                method,
                text,
            )

            if method_ref is None:
                continue

            if _overlap(
                activity_ref["start"],
                activity_ref["end"],
                method_ref["start"],
                method_ref["end"],
            ):
                result.append(
                    {
                        "domain": activity_ref,
                        "range": method_ref,
                        "label": "EMPLOYS",
                        "offset_unit":
                            "python_codepoint",
                    }
                )

    return result


def _relation_objective(
    spans,
    text: str,
):
    acts = [
        span
        for span in spans
        if span.get("label") == "ACTIVITY"
    ]

    goals = [
        span
        for span in spans
        if span.get("label") == "GOAL"
    ]

    result = []

    for activity in acts:
        activity_ref = _relation_span(
            activity,
            text,
        )

        if activity_ref is None:
            continue

        for goal in goals:
            goal_ref = _relation_span(
                goal,
                text,
            )

            if goal_ref is None:
                continue

            result.append(
                {
                    "domain": activity_ref,
                    "range": goal_ref,
                    "label": "HAS_OBJECTIVE",
                    "offset_unit":
                        "python_codepoint",
                }
            )

    return result


def relation_extraction(
    run_dir: Path,
):
    data = read_jsonl(
        run_dir
        / "Entity_Linking"
        / "entity_linking.jsonl"
    )

    for row in data:
        text = str(
            row.get("text", "")
        )

        spans = row.get(
            "spans",
            [],
        )

        labels = {
            s.get("label")
            for s in spans
        }

        relations = []

        if {
            "ACTIVITY",
            "METHOD",
        }.issubset(labels):
            relations += _relation_employs(
                spans,
                text,
            )

        if {
            "ACTIVITY",
            "GOAL",
        }.issubset(labels):
            relations += _relation_objective(
                spans,
                text,
            )

        row["relations"] = relations

    out = (
        run_dir
        / "Relation_Extraction"
        / "relation_extraction.jsonl"
    )

    write_jsonl(
        out,
        data,
    )

    return {
        "artifact": str(
            out.relative_to(
                run_dir
            )
        ),
        "rows": len(data),
        "relations": sum(
            len(
                r.get(
                    "relations",
                    [],
                )
            )
            for r in data
        ),
    }


def _uri_safe(value: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_:/.-]+",
        "_",
        value or "unknown",
    )


def rdf_generation(run_dir: Path):
    from rdflib import (
        Graph,
        Namespace,
        RDF,
        RDFS,
        OWL,
        URIRef,
        Literal,
        XSD,
    )

    data = read_jsonl(
        run_dir
        / "Relation_Extraction"
        / "relation_extraction.jsonl"
    )

    out = (
        run_dir
        / "RDF_Generation"
        / "rdf_generation.rdf"
    )

    g = Graph()

    schema_ns = Namespace(
        "https://scholarlyontology.aueb.gr/resources/so_schema/so_MLE#"
    )

    instances_ns = Namespace(
        "https://scholarlyontology.aueb.gr/resources/so_instances/so_MLE#"
    )

    g.bind(
        "so",
        schema_ns,
    )

    classes = [
        "Article",
        "Sentence",
        "Aggregation",
        "Topic",
        "Person",
        "Organization",
        "Activity",
        "Goal",
        "Method",
    ]

    for c in classes:
        g.add(
            (
                schema_ns[c],
                RDF.type,
                OWL.Class,
            )
        )

    object_properties = {
        "is_part_of": (
            "Sentence",
            "Article",
        ),
        "is_member_of": (
            "Article",
            "Aggregation",
        ),
        "is_topic_of": (
            "Topic",
            "Article",
        ),
        "is_author_of": (
            "Person",
            "Article",
        ),
        "is_affiliated_to": (
            "Person",
            "Organization",
        ),
        "participates_in": (
            "Person",
            "Activity",
        ),
        "has_goal": (
            "Person",
            "Goal",
        ),
        "has_sentence_context": (
            "Activity",
            "Sentence",
        ),
        "employs": (
            "Activity",
            "Method",
        ),
        "has_objective": (
            "Activity",
            "Goal",
        ),
        "uses_method": (
            "Person",
            "Method",
        ),
    }

    for prop, (
        domain,
        range_,
    ) in object_properties.items():
        p = schema_ns[prop]

        g.add(
            (
                p,
                RDF.type,
                OWL.ObjectProperty,
            )
        )

        g.add(
            (
                p,
                RDFS.domain,
                schema_ns[domain],
            )
        )

        g.add(
            (
                p,
                RDFS.range,
                schema_ns[range_],
            )
        )

    datatype_properties = {
        "abstract": (
            "Article",
            XSD.string,
        ),
        "article_DOI": (
            "Article",
            XSD.string,
        ),
        "article_ID": (
            "Article",
            XSD.string,
        ),
        "publication_date": (
            "Article",
            XSD.string,
        ),
        "publication_year": (
            "Article",
            XSD.gYear,
        ),
        "issue_number": (
            "Article",
            XSD.string,
        ),
        "volume_number": (
            "Article",
            XSD.string,
        ),
        "page_count": (
            "Article",
            XSD.integer,
        ),
        "page_start": (
            "Article",
            XSD.integer,
        ),
        "page_end": (
            "Article",
            XSD.integer,
        ),
        "publisher": (
            "Article",
            XSD.string,
        ),
        "doctype": (
            "Article",
            XSD.string,
        ),
        "language": (
            "Article",
            XSD.string,
        ),
        "citation": (
            "Article",
            XSD.string,
        ),
        "family_name": (
            "Person",
            XSD.string,
        ),
        "given_name": (
            "Person",
            XSD.string,
        ),
        "description": (
            "Method",
            XSD.string,
        ),
        "aliases": (
            "Method",
            XSD.string,
        ),
        "wikidata_url": (
            "Method",
            XSD.string,
        ),
        "wikipedia_url": (
            "Method",
            XSD.string,
        ),
        "dbpedia_url": (
            "Method",
            XSD.string,
        ),
        "qid": (
            "Method",
            XSD.string,
        ),
        "orcid": (
            "Person",
            XSD.string,
        ),
        "begin_index": (
            "Method",
            XSD.integer,
        ),
        "end_index": (
            "Method",
            XSD.integer,
        ),
    }

    for prop, (
        domain,
        range_,
    ) in datatype_properties.items():
        p = schema_ns[prop]

        g.add(
            (
                p,
                RDF.type,
                OWL.DatatypeProperty,
            )
        )

        g.add(
            (
                p,
                RDFS.domain,
                schema_ns[domain],
            )
        )

        g.add(
            (
                p,
                RDFS.range,
                range_,
            )
        )

    for row in data:
        meta = row.get(
            "meta", {}
        )

        sent_no = str(
            meta.get(
                "sent_no",
                "0/0",
            )
        )

        article_id = sent_no.split(
            "/"
        )[0]

        article_uri = URIRef(
            str(instances_ns)
            + "Article/"
            + _uri_safe(article_id)
        )

        sent_uri = URIRef(
            str(instances_ns)
            + "Sentence/"
            + _uri_safe(sent_no)
        )

        g.add(
            (
                article_uri,
                RDF.type,
                schema_ns.Article,
            )
        )

        g.add(
            (
                article_uri,
                RDFS.label,
                Literal(
                    meta.get(
                        "title",
                        article_id,
                    )
                ),
            )
        )

        for key, pred in [
            (
                "articleId",
                "article_ID",
            ),
            (
                "doi",
                "article_DOI",
            ),
            (
                "abstract",
                "abstract",
            ),
            (
                "datePublished",
                "publication_date",
            ),
            (
                "publicationYear",
                "publication_year",
            ),
            (
                "issueNumber",
                "issue_number",
            ),
            (
                "volumeNumber",
                "volume_number",
            ),
            (
                "publisher",
                "publisher",
            ),
            (
                "pageCount",
                "page_count",
            ),
            (
                "pageStart",
                "page_start",
            ),
            (
                "pageEnd",
                "page_end",
            ),
            (
                "language",
                "language",
            ),
            (
                "citation",
                "citation",
            ),
        ]:
            if meta.get(key) not in (
                None,
                "",
            ):
                g.add(
                    (
                        article_uri,
                        schema_ns[pred],
                        Literal(
                            meta[key]
                        ),
                    )
                )

        g.add(
            (
                sent_uri,
                RDF.type,
                schema_ns.Sentence,
            )
        )

        g.add(
            (
                sent_uri,
                RDFS.label,
                Literal(
                    row.get(
                        "text",
                        "",
                    )
                ),
            )
        )

        g.add(
            (
                sent_uri,
                schema_ns.is_part_of,
                article_uri,
            )
        )

        authors = (
            meta.get(
                "creator",
                [],
            )
            or []
        )

        author_uris = []

        for author in authors:
            name = author.get(
                "full_name",
                "Unknown",
            )

            key = (
                author.get("orcid")
                if author.get(
                    "orcid"
                ) not in (
                    None,
                    "None",
                    "",
                )
                else name
            )

            person_uri = URIRef(
                str(instances_ns)
                + "Person/"
                + _uri_safe(str(key))
            )

            author_uris.append(
                person_uri
            )

            g.add(
                (
                    person_uri,
                    RDF.type,
                    schema_ns.Person,
                )
            )

            g.add(
                (
                    person_uri,
                    RDFS.label,
                    Literal(name),
                )
            )

            if author.get("orcid") not in (
                None,
                "None",
                "",
            ):
                g.add(
                    (
                        person_uri,
                        schema_ns.orcid,
                        Literal(
                            author["orcid"]
                        ),
                    )
                )

            if author.get(
                "family_name"
            ):
                g.add(
                    (
                        person_uri,
                        schema_ns.family_name,
                        Literal(
                            author[
                                "family_name"
                            ]
                        ),
                    )
                )

            if author.get(
                "given_name"
            ):
                g.add(
                    (
                        person_uri,
                        schema_ns.given_name,
                        Literal(
                            author[
                                "given_name"
                            ]
                        ),
                    )
                )

            g.add(
                (
                    person_uri,
                    schema_ns.is_author_of,
                    article_uri,
                )
            )

        activity_uris = []
        goal_uris = []
        method_uris = []

        for span in row.get(
            "spans", []
        ):
            start = int(
                span.get("start", 0)
            )
            end = int(
                span.get("end", 0)
            )

            label = span.get(
                "label"
            )

            if label == "ACTIVITY":
                u = URIRef(
                    str(instances_ns)
                    + "Activity/"
                    + f"{_uri_safe(sent_no)}_{start}_{end}"
                )

                activity_uris.append(
                    (u, span)
                )

                g.add(
                    (
                        u,
                        RDF.type,
                        schema_ns.Activity,
                    )
                )

                g.add(
                    (
                        u,
                        RDFS.label,
                        Literal(
                            row.get(
                                "text",
                                "",
                            )[start:end]
                        ),
                    )
                )

                g.add(
                    (
                        u,
                        schema_ns.has_sentence_context,
                        sent_uri,
                    )
                )

                g.add(
                    (
                        u,
                        schema_ns.begin_index,
                        Literal(start),
                    )
                )

                g.add(
                    (
                        u,
                        schema_ns.end_index,
                        Literal(end),
                    )
                )

                for person in author_uris:
                    g.add(
                        (
                            person,
                            schema_ns.participates_in,
                            u,
                        )
                    )

            elif label == "GOAL":
                u = URIRef(
                    str(instances_ns)
                    + "Goal/"
                    + f"{_uri_safe(sent_no)}_{start}_{end}"
                )

                goal_uris.append(
                    (u, span)
                )

                g.add(
                    (
                        u,
                        RDF.type,
                        schema_ns.Goal,
                    )
                )

                g.add(
                    (
                        u,
                        RDFS.label,
                        Literal(
                            row.get(
                                "text",
                                "",
                            )[start:end]
                        ),
                    )
                )

                g.add(
                    (
                        u,
                        schema_ns.has_sentence_context,
                        sent_uri,
                    )
                )

                g.add(
                    (
                        u,
                        schema_ns.begin_index,
                        Literal(start),
                    )
                )

                g.add(
                    (
                        u,
                        schema_ns.end_index,
                        Literal(end),
                    )
                )

                for person in author_uris:
                    g.add(
                        (
                            person,
                            schema_ns.has_goal,
                            u,
                        )
                    )

            elif label == "METHOD":
                proper = span.get(
                    "proper_name"
                )

                if _is_missing_value(
                    proper
                ):
                    proper = row.get(
                        "text",
                        "",
                    )[start:end]

                u = URIRef(
                    str(instances_ns)
                    + "Method/"
                    + _uri_safe(str(proper))
                )

                method_uris.append(
                    (u, span)
                )

                g.add(
                    (
                        u,
                        RDF.type,
                        schema_ns.Method,
                    )
                )

                g.add(
                    (
                        u,
                        RDFS.label,
                        Literal(str(proper)),
                    )
                )

                g.add(
                    (
                        u,
                        schema_ns.has_sentence_context,
                        sent_uri,
                    )
                )

                g.add(
                    (
                        u,
                        schema_ns.begin_index,
                        Literal(start),
                    )
                )

                g.add(
                    (
                        u,
                        schema_ns.end_index,
                        Literal(end),
                    )
                )

                for key in [
                    "description",
                    "aliases",
                    "wikidata_url",
                    "wikipedia_url",
                    "dbpedia_url",
                    "qid",
                ]:
                    value = span.get(key)

                    if not _is_missing_value(value):
                        g.add(
                            (
                                u,
                                schema_ns[key],
                                Literal(str(value)),
                            )
                        )

        # Prefer relation extraction output, which preserves
        # notebook semantics.
        for rel in row.get(
            "relations", []
        ):
            domain = rel.get(
                "domain", {}
            )
            range_ = rel.get(
                "range", {}
            )

            if rel.get("label") == "EMPLOYS":
                du = next(
                    (
                        u
                        for u, s in activity_uris
                        if s.get("start") == domain.get("start")
                        and s.get("end") == domain.get("end")
                    ),
                    None,
                )

                ru = next(
                    (
                        u
                        for u, s in method_uris
                        if s.get("start") == range_.get("start")
                        and s.get("end") == range_.get("end")
                    ),
                    None,
                )

                if du and ru:
                    g.add(
                        (
                            du,
                            schema_ns.employs,
                            ru,
                        )
                    )

                    for person in author_uris:
                        g.add(
                            (
                                person,
                                schema_ns.uses_method,
                                ru,
                            )
                        )

            elif rel.get("label") == "HAS_OBJECTIVE":
                du = next(
                    (
                        u
                        for u, s in activity_uris
                        if s.get("start") == domain.get("start")
                        and s.get("end") == domain.get("end")
                    ),
                    None,
                )

                for gu, gs in goal_uris:
                    if du:
                        g.add(
                            (
                                du,
                                schema_ns.has_objective,
                                gu,
                            )
                        )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    g.serialize(
        destination=str(out),
        format="xml",
    )

    return {
        "artifact": str(
            out.relative_to(run_dir)
        ),
        "triples": len(g),
    }
