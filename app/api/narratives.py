"""
API routes for narrative generation.
Handles synchronous narrative generation requests.
For async/long-running jobs, use the /jobs endpoint instead.
"""

import logging
import unicodedata
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from data.cache.parquet_cache import load_snapshot, snapshot_exists, save_snapshot
from data.profiling.profiler import profile_dataset
from data.profiling.matcher import match_template, MatchResult, TemplateMatch
from data.profiling.template_definitions import TemplateType, TEMPLATE_MAP
from data.analytics.evidence_bundle import BundleBuilder
from data.storage.catalog_index import CatalogIndex
from data.storage.metadata import MetadataStore
from data.ingestion.ckan_client import CKANClient, CKANError
from data.metadata_normalize import from_ckan as normalize_ckan_metadata
from app.api.package import PackageBuilder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/narratives", tags=["narratives"])

# Shared instances
_catalog_index = CatalogIndex(db_path=settings.metadata_db_path)
_metadata_store = MetadataStore(db_path=settings.metadata_db_path)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    dataset_id: str
    user_message: Optional[str] = None
    title: Optional[str] = None


class AskRequest(BaseModel):
    user_message: str
    title: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS_EN = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "about", "what", "which", "who", "whom", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "and", "but", "or", "if", "while", "because", "until", "although",
    "tell", "show", "give", "get", "know", "think", "look", "like",
    "much", "many", "over", "years", "year", "changed", "change",
    "trends", "trend", "data", "city",
}

_STOP_WORDS_IS = {
    # Question words
    "hversu", "hvað", "hvad", "hvar", "hvenær", "hvenaer", "hvernig",
    "hver", "hvers", "hverju", "hvern", "hverja", "hverjir",
    # Prepositions / conjunctions
    "eftir", "um", "við", "vid", "með", "med", "frá", "fra", "til",
    "upp", "úr", "ur", "og", "eða", "eda", "en", "sem", "að", "ad",
    # Pronouns
    "ég", "eg", "þú", "thu", "hann", "hún", "hun", "það", "thad",
    "við", "þið", "thid", "þeir", "their", "þær", "thaer", "mig",
    "þig", "thig", "sig", "mér", "mer", "þér", "ther", "sér", "ser",
    # Demonstratives
    "þetta", "thetta", "þessi", "thessi", "þessu", "thessu", "þennan", "sú",
    # Auxiliaries / common verbs
    "er", "var", "voru", "eru", "vera", "verða", "verda", "sé", "sér",
    "geta", "get", "fá", "fær", "fá", "má", "mun", "munu", "skyldi",
    # Common verb forms
    "berðu", "berdu", "segðu", "segdu", "segja", "sýna", "syna",
    "skoða", "skoda",
    # Adverbs / particles
    "saman", "líka", "lika", "bara", "mjög", "mjog", "ekki", "hér",
    "þar", "thar", "nú", "núna", "þá", "tha", "kannski",
    # Misc function words
    "einn", "ein", "eitt", "nokkur", "margir", "mörg", "morg", "mikið",
    "mikid", "lítið", "litid", "fleiri", "flest", "allir", "allt",
}

_STOP_WORDS = _STOP_WORDS_EN | _STOP_WORDS_IS


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from a natural language question."""
    words = text.lower().split()
    keywords = []
    for w in words:
        cleaned = w.strip("?.,!:;\"'()[]{}").strip()
        if cleaned and cleaned not in _STOP_WORDS and len(cleaned) > 2:
            keywords.append(cleaned)
    return keywords


def _ascii_fold(text: str) -> str:
    """Convert text to a lowercase ASCII approximation for robust matching."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _finance_hint_terms(text: str) -> list[str]:
    """
    Add deterministic budget/finance aliases to improve catalog lookup.
    This avoids over-reliance on LLM translation quality for Icelandic tags.
    """
    folded = _ascii_fold(text)
    finance_triggers = (
        "budget", "finance", "financial", "spending", "expenditure",
        "expense", "revenue", "tax", "fiscal",
        "fjarmal", "uppgjor", "rekstur", "fjarhag",
    )
    if not any(t in folded for t in finance_triggers):
        return []
    return [
        "budget",
        "finance",
        "fjármál",
        "fjarmal",
        "uppgjör",
        "uppgjor",
        "ársuppgjör",
        "arsuppgjor",
        "fjárhagsáætlun",
        "fjarhagsaaetlun",
        "rekstur",
    ]


def _dedupe_terms(terms: list[str]) -> list[str]:
    """Deduplicate while preserving order, dropping empty terms."""
    seen = set()
    cleaned = []
    for t in terms:
        value = (t or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned


async def _parse_intent(user_message: str):
    """
    Use the LLM intent parser to extract structured intent including
    dataset_query and dataset_query_local (translated to portal language).
    Returns (dataset_query, dataset_query_local) or falls back to keywords.
    """
    portal_lang = settings.ckan_portal_language

    try:
        from llm.interface import get_providers
        from llm.intent import IntentParser

        intent_provider, _ = get_providers()
        parser = IntentParser(intent_provider)
        result = await parser.parse(user_message, portal_language=portal_lang)

        if result.success and result.intent:
            query = result.intent.dataset_query
            query_local = result.intent.dataset_query_local or query
            logger.info(
                f"Intent parsed — query: '{query}', "
                f"query_local ({portal_lang}): '{query_local}'"
            )
            return query, query_local
    except Exception as e:
        logger.warning(f"Intent parsing unavailable ({e}), using keyword extraction")

    # Fallback: extract keywords (no translation available without LLM)
    keywords = _extract_keywords(user_message)
    fallback = " ".join(keywords) if keywords else user_message
    return fallback, fallback


def _stem_prefix(word: str, min_len: int = 4) -> str:
    """Return a rough Icelandic stem by trimming the last 1-2 suffix chars.

    Icelandic inflections mostly change word endings (-i, -a, -um, -ar,
    -ir, -ur, etc.) while the stem is a prefix.  Trimming the shorter of
    2 characters or 30 % of the word (minimum ``min_len`` chars) is a
    pragmatic heuristic that covers most cases.
    """
    if len(word) <= min_len:
        return word
    trim = min(2, max(1, len(word) // 3))
    return word[: max(min_len, len(word) - trim)]


def _score_candidate(
    title: str,
    description: str,
    tags: list[str],
    scoring_keywords: list[str],
) -> float:
    """Score a dataset candidate by how many user keywords it matches.

    For each keyword, the scorer first tries an exact substring match
    (score +1).  If that fails it tries a stem-prefix match (score +0.7)
    to handle Icelandic inflection (e.g. "hverfi" stem "hverf" matches
    "hverfum").
    """
    haystack = _ascii_fold(f"{title} {description} {' '.join(tags)}")
    score = 0.0
    for kw in scoring_keywords:
        folded = _ascii_fold(kw)
        if folded in haystack:
            score += 1.0
        elif _stem_prefix(folded) in haystack:
            score += 0.7
    return score


def _build_scoring_keywords(
    dataset_query: str,
    dataset_query_local: str,
    user_message: str,
) -> list[str]:
    """Build a de-duplicated list of meaningful keywords for candidate scoring."""
    raw = (
        _extract_keywords(dataset_query)
        + _extract_keywords(dataset_query_local)
        + _extract_keywords(user_message)
    )
    seen: set[str] = set()
    out: list[str] = []
    for kw in raw:
        folded = _ascii_fold(kw)
        if folded not in seen:
            seen.add(folded)
            out.append(kw)
    return out


async def _search_ckan(
    client: CKANClient,
    search_terms: list[str],
    scoring_keywords: list[str],
):
    """Search CKAN portal, collect candidates, and return the best match.

    Instead of stopping at the first hit, queries CKAN with up to
    ``_CKAN_SEARCH_BREADTH`` terms, collects all unique datasets that have
    supported resources, scores them by keyword overlap, and returns the
    highest-scoring candidate.
    """
    _CKAN_SEARCH_BREADTH = 6

    candidates: dict[str, tuple[str, str, float]] = {}  # slug → (name, slug, score)

    for term in search_terms[:_CKAN_SEARCH_BREADTH]:
        try:
            results = await client.search_datasets(query=term, rows=5)
            for ds in results:
                if ds.name in candidates or not ds.supported_resources:
                    continue
                name = ds.title or ds.name
                tags = [t for t in (ds.tags or [])]
                score = _score_candidate(
                    name, ds.notes or "", tags, scoring_keywords,
                )
                candidates[ds.name] = (name, ds.name, score)
                logger.debug(
                    f"CKAN candidate '{name}' (via '{term}') score={score}"
                )
        except Exception as e:
            logger.warning(f"CKAN search failed for '{term}': {e}")

    if not candidates:
        return None, None

    best_name, best_slug, best_score = max(
        candidates.values(), key=lambda c: c[2],
    )
    logger.info(
        f"CKAN best match: '{best_name}' (score={best_score}, "
        f"{len(candidates)} candidates evaluated)"
    )
    return best_name, best_slug


def _reconstruct_match(config: dict, profile) -> MatchResult:
    """
    Reconstruct a MatchResult from a stored template configuration,
    skipping the full match_template() computation.
    """
    template_type = TemplateType(config["template_type"])
    template = TEMPLATE_MAP[template_type]
    return MatchResult(
        dataset_id=config["dataset_id"],
        best_match=TemplateMatch(
            template_id=template_type,
            template_name=template.name,
            score=1.0,
            matched_columns=config["column_mappings"],
            missing_required=[],
            is_viable=True,
        ),
        all_matches=[],
        profile_summary=profile.column_types_summary,
    )


async def _parse_full_intent(user_message: str):
    """
    Parse user intent once and return both the structured intent object
    (for narrative generation) and search queries (for dataset lookup).
    Avoids redundant LLM calls.
    """
    portal_lang = settings.ckan_portal_language
    intent = None

    try:
        from llm.interface import get_providers
        from llm.intent import IntentParser

        intent_provider, _ = get_providers()
        parser = IntentParser(intent_provider)
        result = await parser.parse(user_message, portal_language=portal_lang)

        if result.success and result.intent:
            intent = result.intent
            query = intent.dataset_query
            query_local = intent.dataset_query_local or query
            logger.info(
                f"Intent parsed — query: '{query}', "
                f"query_local ({portal_lang}): '{query_local}'"
            )
            return query, query_local, intent
    except Exception as e:
        logger.warning(f"Intent parsing unavailable ({e}), using keyword extraction")

    keywords = _extract_keywords(user_message)
    fallback = " ".join(keywords) if keywords else user_message
    return fallback, fallback, intent


async def _find_dataset(user_message: str, intent_queries=None) -> tuple[str, str]:
    """
    Search the catalog for a dataset matching the user's question.
    Returns (dataset_name, ckan_slug).

    Args:
        user_message: The user's question.
        intent_queries: Optional (dataset_query, dataset_query_local) tuple
            from a prior intent parse, to avoid redundant LLM calls.
    """
    if intent_queries:
        dataset_query, dataset_query_local = intent_queries
    else:
        dataset_query, dataset_query_local = await _parse_intent(user_message)

    # Build search term list: intent terms + user message + keywords + aliases.
    # The full user message is included because CKAN Solr tokenises
    # multi-word strings and performs its own relevance ranking.
    search_terms = []
    if dataset_query_local:
        search_terms.append(dataset_query_local)
    search_terms.append(user_message)
    local_keywords = dataset_query_local.split() if dataset_query_local else []
    search_terms.extend(local_keywords)
    search_terms.append(dataset_query)
    search_terms.extend(_extract_keywords(dataset_query))
    search_terms.extend(_extract_keywords(user_message))
    search_terms.extend(_finance_hint_terms(dataset_query))
    search_terms.extend(_finance_hint_terms(dataset_query_local))
    search_terms.extend(_finance_hint_terms(user_message))
    search_terms = _dedupe_terms(search_terms)

    logger.info(f"Dataset search terms: {search_terms}")

    scoring_keywords = _build_scoring_keywords(
        dataset_query, dataset_query_local, user_message,
    )
    logger.info(f"Scoring keywords: {scoring_keywords}")

    client = CKANClient(settings.ckan_portal_url)

    # Strategy 1: Search CKAN portal API (multi-candidate scoring)
    dataset_name, ckan_slug = await _search_ckan(
        client, search_terms, scoring_keywords,
    )

    # Strategy 2: Fallback to local catalog (also scored)
    if not ckan_slug:
        local_candidates: dict[str, tuple[str, str, float]] = {}
        for kw in search_terms:
            results = _catalog_index.search(query=kw, limit=5)
            for entry in results:
                slug = entry["name"]
                if slug in local_candidates:
                    continue
                title = entry["title"] or slug
                tags = entry["tags"] if isinstance(entry["tags"], list) else []
                score = _score_candidate(
                    title, entry.get("description", ""), tags,
                    scoring_keywords,
                )
                local_candidates[slug] = (title, slug, score)
        if local_candidates:
            best_name, best_slug, best_score = max(
                local_candidates.values(), key=lambda c: c[2],
            )
            dataset_name = best_name
            ckan_slug = best_slug
            logger.info(
                f"Local catalog best match: '{best_name}' "
                f"(score={best_score}, {len(local_candidates)} candidates)"
            )

    if not ckan_slug:
        raise HTTPException(
            status_code=404,
            detail="Oops! I couldn't find data matching your question. Try different keywords!",
        )

    return dataset_name, ckan_slug


async def _get_dataframe(dataset_id: str, config: Optional[dict] = None):
    """
    Get a DataFrame for a dataset, using TTL cache or re-downloading.
    Returns (df, resource_url, ckan_dataset_or_None).
    """
    # Check TTL cache first
    df = load_snapshot(dataset_id)
    if df is not None:
        resource_url = config["resource_url"] if config else ""
        # Still fetch CKAN metadata even from cache
        try:
            client = CKANClient(settings.ckan_portal_url)
            ckan_ds = await client.get_dataset(dataset_id)
            if not resource_url and ckan_ds and ckan_ds.resources:
                resource_url = ckan_ds.resources[0].url or ""
        except Exception:
            ckan_ds = None
        return df, resource_url, ckan_ds

    # Need to download — get resource URL from config or CKAN API
    client = CKANClient(settings.ckan_portal_url)

    if config and config.get("resource_url"):
        # Re-download using stored resource URL
        logger.info(f"Re-downloading {dataset_id} from stored resource URL")
        from data.ingestion.ckan_client import CKANResource
        resource = CKANResource(id=dataset_id, url=config["resource_url"], format="CSV", name=dataset_id)
        df = await client.download_resource_as_dataframe(resource)
        save_snapshot(df, dataset_id)
        # Fetch metadata
        try:
            ckan_ds = await client.get_dataset(dataset_id)
        except Exception:
            ckan_ds = None
        return df, config["resource_url"], ckan_ds

    # Fetch from CKAN API
    logger.info(f"Fetching {dataset_id} from CKAN API")
    dataset = await client.get_dataset(dataset_id)
    supported = dataset.supported_resources
    if not supported:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset '{dataset_id}' has no supported file formats (CSV/JSON/Excel).",
        )

    resource = supported[0]
    df = await client.download_resource_as_dataframe(resource)
    save_snapshot(df, dataset_id)
    return df, resource.url, dataset


def _profile_and_match(df, dataset_id: str, config: Optional[dict] = None, resource_url: str = ""):
    """
    Profile a dataset and match a template.
    If a config exists, skip match_template() and reconstruct from stored config.
    If not, run full profiling + matching and save the config.
    Returns (profile, match_result).
    """
    profile_source = resource_url or settings.ckan_portal_url
    profile = profile_dataset(df, dataset_id=dataset_id, source=profile_source)

    if config:
        # Try to reuse stored template configuration
        try:
            match = _reconstruct_match(config, profile)
            # Validate the reconstructed match columns actually exist in the DataFrame
            missing_cols = [
                col for col in match.best_match.matched_columns.values()
                if col and col not in df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Stored config references missing columns: {missing_cols}"
                )
            _metadata_store.touch_config(dataset_id)
            logger.info(f"Reusing stored config for {dataset_id}: {config['template_type']}")
        except Exception as e:
            logger.warning(
                f"Stored config for {dataset_id} is invalid ({e}), re-matching"
            )
            config = None  # Fall through to fresh matching

    if not config:
        # First time or stale config: full matching
        match = match_template(profile)
        if match.best_match and match.best_match.is_viable:
            _metadata_store.save_config(
                dataset_id=dataset_id,
                portal_url=settings.ckan_portal_url,
                resource_url=resource_url,
                template_type=match.best_match.template_id.value,
                column_mappings=match.best_match.matched_columns,
                profiling_summary={
                    **profile.column_types_summary,
                    "row_count": profile.row_count,
                    "column_count": profile.column_count,
                },
            )
            logger.info(f"Saved new config for {dataset_id}: {match.best_match.template_id.value}")

    return profile, match


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/preview")
async def preview_narrative(request: GenerateRequest):
    """
    Generate a narrative preview WITHOUT LLM.
    Returns the evidence bundle with visualizations and key findings.
    """
    try:
        config = _metadata_store.get_config(request.dataset_id)
        df, resource_url, ckan_ds = await _get_dataframe(request.dataset_id, config)

        profile, match = _profile_and_match(df, request.dataset_id, config, resource_url)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"No viable template match for dataset. Types found: {profile.column_types_summary}",
            )

        # Normalize CKAN metadata for the evidence bundle
        metadata = None
        if ckan_ds:
            try:
                metadata = normalize_ckan_metadata(ckan_ds, portal_language=settings.ckan_portal_language)
            except Exception as e:
                logger.warning(f"Metadata normalization failed: {e}")

        builder = BundleBuilder()
        bundle = builder.build(df, profile, match, title=request.title, metadata=metadata)

        pkg_builder = PackageBuilder()
        package = pkg_builder.build_without_narrative(bundle)

        return package.to_api_response()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_narrative(request: GenerateRequest):
    """
    Generate a full narrative WITH LLM.
    Requires an LLM provider (Ollama) to be running.
    """
    try:
        config = _metadata_store.get_config(request.dataset_id)
        df, resource_url, ckan_ds = await _get_dataframe(request.dataset_id, config)

        profile, match = _profile_and_match(df, request.dataset_id, config, resource_url)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"No viable template match. Types found: {profile.column_types_summary}",
            )

        metadata = None
        if ckan_ds:
            try:
                metadata = normalize_ckan_metadata(ckan_ds, portal_language=settings.ckan_portal_language)
            except Exception as e:
                logger.warning(f"Metadata normalization failed: {e}")

        builder = BundleBuilder()
        bundle = builder.build(df, profile, match, title=request.title, metadata=metadata)

        from llm.interface import get_providers
        from llm.intent import IntentParser
        from llm.narrative import NarrativeGenerator

        intent_provider, generation_provider = get_providers()
        parsed_intent = None
        if request.user_message:
            try:
                parser = IntentParser(intent_provider)
                intent_result = await parser.parse(
                    request.user_message,
                    portal_language=settings.ckan_portal_language,
                )
                parsed_intent = intent_result.intent
            except Exception as e:
                logger.warning(f"Intent parse for chart labels failed: {e}")

        generator = NarrativeGenerator(generation_provider, intent_llm_provider=intent_provider)
        result = await generator.generate(
            bundle,
            user_message=request.user_message,
            intent=parsed_intent,
        )

        if not result.success:
            raise HTTPException(
                status_code=502,
                detail=f"Narrative generation failed: {result.error}",
            )

        try:
            from llm.chart_labels import ChartLabeler
            labeler = ChartLabeler()
            label_language = parsed_intent.language if parsed_intent else "en"
            await labeler.relabel_bundle(
                bundle,
                language=label_language,
                narrative=result.narrative,
            )
        except Exception as e:
            logger.warning(f"Chart title relabeling skipped: {e}")

        pkg_builder = PackageBuilder()
        llm_model_name = getattr(generation_provider, "model", "")
        package = pkg_builder.build(
            result,
            bundle,
            llm_provider_name=settings.llm_provider,
            llm_model_name=llm_model_name,
        )

        return package.to_api_response()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Narrative generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Unified "ask" endpoint — single prompt → full story
# ---------------------------------------------------------------------------

@router.post("/ask")
async def ask_narrative(request: AskRequest):
    """
    Unified endpoint: user asks a question → system finds the right dataset,
    fetches it, profiles, analyzes, generates a narrative, and returns the
    complete editorial data story.
    """
    try:
        # Step 1: Parse intent ONCE (reused for both dataset search and narrative)
        dataset_query, dataset_query_local, intent = await _parse_full_intent(
            request.user_message
        )

        # Step 2: Find dataset in catalog (reuses parsed queries)
        dataset_name, ckan_slug = await _find_dataset(
            request.user_message,
            intent_queries=(dataset_query, dataset_query_local),
        )

        # Step 3: Check for existing template configuration
        config = _metadata_store.get_config(ckan_slug)

        # Step 4: Get data (TTL cache or re-download)
        df, resource_url, ckan_ds = await _get_dataframe(ckan_slug, config)

        # Step 5: Profile and match (skip matching if config exists)
        profile, match = _profile_and_match(df, ckan_slug, config, resource_url)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"Could not analyze dataset '{dataset_name}'. "
                       f"Column types found: {profile.column_types_summary}",
            )

        # Step 5b: Normalize CKAN metadata for evidence bundle
        metadata = None
        if ckan_ds:
            try:
                metadata = normalize_ckan_metadata(ckan_ds, portal_language=settings.ckan_portal_language)
            except Exception as e:
                logger.warning(f"Metadata normalization failed: {e}")

        # Step 6: Build evidence bundle
        builder = BundleBuilder()
        bundle = builder.build(df, profile, match, title=request.title, metadata=metadata)

        # Step 7: Generate narrative with LLM (fall back to preview if unavailable)
        pkg_builder = PackageBuilder()
        try:
            from llm.interface import get_providers
            from llm.narrative import NarrativeGenerator

            _, generation_provider = get_providers()
            generator = NarrativeGenerator(generation_provider)
            # Pass pre-parsed intent to skip redundant LLM call
            result = await generator.generate(bundle, intent=intent)

            if result.success:
                try:
                    from llm.chart_labels import ChartLabeler
                    labeler = ChartLabeler()
                    label_language = intent.language if intent else "en"
                    await labeler.relabel_bundle(
                        bundle,
                        language=label_language,
                        narrative=result.narrative,
                    )
                except Exception as e:
                    logger.warning(f"Chart title relabeling skipped: {e}")
                llm_model_name = getattr(generation_provider, "model", "")
                package = pkg_builder.build(
                    result,
                    bundle,
                    llm_provider_name=settings.llm_provider,
                    llm_model_name=llm_model_name,
                )
            else:
                logger.warning(f"LLM generation failed, falling back to preview: {result.error}")
                package = pkg_builder.build_without_narrative(bundle)

        except Exception as llm_error:
            logger.warning(f"LLM unavailable ({llm_error}), using preview mode")
            package = pkg_builder.build_without_narrative(bundle)

        # Step 7: Return response with dataset info
        response = package.to_api_response()
        response["dataset_name"] = dataset_name
        response["dataset_source"] = bundle.source
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask narrative failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
