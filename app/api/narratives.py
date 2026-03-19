"""
API routes for narrative generation.
Handles synchronous narrative generation requests.
For async/long-running jobs, use the /jobs endpoint instead.
"""

import logging
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
from app.api.package import PackageBuilder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/narratives", tags=["narratives"])

# Shared instances
_catalog_index = CatalogIndex()
_metadata_store = MetadataStore()


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

def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from a natural language question."""
    stop_words = {
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
    words = text.lower().split()
    keywords = []
    for w in words:
        cleaned = w.strip("?.,!:;\"'()[]{}").strip()
        if cleaned and cleaned not in stop_words and len(cleaned) > 2:
            keywords.append(cleaned)
    return keywords


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


async def _search_ckan(client: CKANClient, search_terms: list[str]):
    """
    Search CKAN portal with a list of search terms (tries each until a
    dataset with supported formats is found).
    Returns (dataset_name, ckan_slug) or (None, None).
    """
    for term in search_terms:
        try:
            results = await client.search_datasets(query=term, rows=5)
            for ds in results:
                if ds.supported_resources:
                    name = ds.title or ds.name
                    logger.info(f"CKAN search found: '{name}' via '{term}'")
                    return name, ds.name
        except Exception as e:
            logger.warning(f"CKAN search failed for '{term}': {e}")
    return None, None


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


async def _find_dataset(user_message: str) -> tuple[str, str]:
    """
    Search the catalog for a dataset matching the user's question.
    Returns (dataset_name, ckan_slug).

    Flow:
    1. Parse intent with LLM → get dataset_query + dataset_query_local
    2. Search CKAN portal API with the local-language query
    3. Fall back to local catalog search
    """
    dataset_query, dataset_query_local = await _parse_intent(user_message)

    # Build search term list: local-language query first, then original keywords
    search_terms = []
    local_keywords = dataset_query_local.split()
    if dataset_query_local:
        search_terms.append(dataset_query_local)
    search_terms.extend(local_keywords)
    original_keywords = _extract_keywords(dataset_query)
    for kw in original_keywords:
        if kw not in search_terms:
            search_terms.append(kw)

    logger.info(f"Dataset search terms: {search_terms}")

    client = CKANClient(settings.ckan_portal_url)

    # Strategy 1: Search CKAN portal API
    dataset_name, ckan_slug = await _search_ckan(client, search_terms)

    # Strategy 2: Fallback to local catalog
    if not ckan_slug:
        for kw in search_terms:
            results = _catalog_index.search(query=kw, limit=5)
            if results:
                catalog_entry = results[0]
                dataset_name = catalog_entry["title"] or catalog_entry["name"]
                ckan_slug = catalog_entry["name"]
                logger.info(f"Local catalog found '{dataset_name}' via keyword '{kw}'")
                break

    if not ckan_slug:
        raise HTTPException(
            status_code=404,
            detail="No dataset found matching your question. Try different keywords.",
        )

    return dataset_name, ckan_slug


async def _get_dataframe(dataset_id: str, config: Optional[dict] = None):
    """
    Get a DataFrame for a dataset, using TTL cache or re-downloading.
    Returns (df, resource_url).
    """
    # Check TTL cache first
    df = load_snapshot(dataset_id)
    if df is not None:
        resource_url = config["resource_url"] if config else ""
        return df, resource_url

    # Need to download — get resource URL from config or CKAN API
    client = CKANClient(settings.ckan_portal_url)

    if config and config.get("resource_url"):
        # Re-download using stored resource URL
        logger.info(f"Re-downloading {dataset_id} from stored resource URL")
        from data.ingestion.ckan_client import CKANResource
        resource = CKANResource(url=config["resource_url"], format="CSV", name=dataset_id)
        df = await client.download_resource_as_dataframe(resource)
        save_snapshot(df, dataset_id)
        return df, config["resource_url"]

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
    return df, resource.url


def _profile_and_match(df, dataset_id: str, config: Optional[dict] = None, resource_url: str = ""):
    """
    Profile a dataset and match a template.
    If a config exists, skip match_template() and reconstruct from stored config.
    If not, run full profiling + matching and save the config.
    Returns (profile, match_result).
    """
    profile = profile_dataset(df, dataset_id=dataset_id)

    if config:
        # Reuse stored template configuration
        match = _reconstruct_match(config, profile)
        _metadata_store.touch_config(dataset_id)
        logger.info(f"Reusing stored config for {dataset_id}: {config['template_type']}")
    else:
        # First time: full matching
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
        df, resource_url = await _get_dataframe(request.dataset_id, config)

        profile, match = _profile_and_match(df, request.dataset_id, config, resource_url)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"No viable template match for dataset. Types found: {profile.column_types_summary}",
            )

        builder = BundleBuilder()
        bundle = builder.build(df, profile, match, title=request.title)

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
        df, resource_url = await _get_dataframe(request.dataset_id, config)

        profile, match = _profile_and_match(df, request.dataset_id, config, resource_url)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"No viable template match. Types found: {profile.column_types_summary}",
            )

        builder = BundleBuilder()
        bundle = builder.build(df, profile, match, title=request.title)

        from llm.interface import get_providers
        from llm.narrative import NarrativeGenerator

        intent_provider, generation_provider = get_providers()
        generator = NarrativeGenerator(generation_provider, intent_llm_provider=intent_provider)
        result = await generator.generate(bundle, user_message=request.user_message)

        if not result.success:
            raise HTTPException(
                status_code=502,
                detail=f"Narrative generation failed: {result.error}",
            )

        pkg_builder = PackageBuilder()
        package = pkg_builder.build(result, bundle, llm_provider_name="ollama")

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
        # Step 1: Find dataset in catalog
        dataset_name, ckan_slug = await _find_dataset(request.user_message)

        # Step 2: Check for existing template configuration
        config = _metadata_store.get_config(ckan_slug)

        # Step 3: Get data (TTL cache or re-download)
        df, resource_url = await _get_dataframe(ckan_slug, config)

        # Step 4: Profile and match (skip matching if config exists)
        profile, match = _profile_and_match(df, ckan_slug, config, resource_url)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"Could not analyze dataset '{dataset_name}'. "
                       f"Column types found: {profile.column_types_summary}",
            )

        # Step 5: Build evidence bundle
        builder = BundleBuilder()
        bundle = builder.build(df, profile, match, title=request.title)

        # Step 6: Generate narrative with LLM (fall back to preview if unavailable)
        pkg_builder = PackageBuilder()
        try:
            from llm.interface import get_providers
            from llm.narrative import NarrativeGenerator

            intent_provider, generation_provider = get_providers()
            generator = NarrativeGenerator(
                generation_provider, intent_llm_provider=intent_provider
            )
            result = await generator.generate(bundle, user_message=request.user_message)

            if result.success:
                package = pkg_builder.build(result, bundle, llm_provider_name="ollama")
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
