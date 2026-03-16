"""
API routes for narrative generation.
Handles synchronous narrative generation requests.
For async/long-running jobs, use the /jobs endpoint instead.
"""

import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from data.cache.parquet_cache import load_snapshot, snapshot_exists, save_snapshot
from data.profiling.profiler import profile_dataset
from data.profiling.matcher import match_template
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
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/preview")
async def preview_narrative(request: GenerateRequest):
    """
    Generate a narrative preview WITHOUT LLM.
    Returns the evidence bundle with visualizations and key findings.
    Useful for testing the pipeline or when the LLM is unavailable.
    """
    try:
        # Load dataset
        if not snapshot_exists(request.dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found")

        df = load_snapshot(request.dataset_id)

        # Profile → match → bundle
        profile = profile_dataset(df, dataset_id=request.dataset_id)
        match = match_template(profile)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"No viable template match for dataset. Types found: {profile.column_types_summary}",
            )

        builder = BundleBuilder()
        bundle = builder.build(df, profile, match, title=request.title)

        # Build package without narrative
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
    This is a synchronous endpoint — for long-running generation,
    use POST /jobs/generate instead.

    Requires an LLM provider (Ollama) to be running.
    """
    try:
        # Load dataset
        if not snapshot_exists(request.dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found")

        df = load_snapshot(request.dataset_id)

        # Profile → match → bundle
        profile = profile_dataset(df, dataset_id=request.dataset_id)
        match = match_template(profile)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"No viable template match. Types found: {profile.column_types_summary}",
            )

        builder = BundleBuilder()
        bundle = builder.build(df, profile, match, title=request.title)

        # Generate narrative with LLM
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

        # Build full package
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


async def _find_and_ingest_dataset(user_message: str) -> tuple[str, str]:
    """
    Search the catalog for a dataset matching the user's question,
    auto-ingest it if needed, and return (dataset_id, dataset_name).

    Flow:
    1. Parse intent with LLM → get dataset_query + dataset_query_local
       (translated to the portal's language by the LLM)
    2. Search CKAN portal API with the local-language query
    3. Fall back to local catalog search
    """
    # Step 1: Parse intent — LLM translates the query to portal language
    dataset_query, dataset_query_local = await _parse_intent(user_message)

    # Build search term list: local-language query first, then original keywords
    search_terms = []
    # Split the local query into individual keywords for broader matching
    local_keywords = dataset_query_local.split()
    if dataset_query_local:
        search_terms.append(dataset_query_local)  # try full phrase first
    search_terms.extend(local_keywords)
    # Add original keywords as fallback (user might type in portal language)
    original_keywords = _extract_keywords(dataset_query)
    for kw in original_keywords:
        if kw not in search_terms:
            search_terms.append(kw)

    logger.info(f"Dataset search terms: {search_terms}")

    client = CKANClient(settings.ckan_portal_url)

    # Strategy 1: Search CKAN portal API
    dataset_name, ckan_dataset_name = await _search_ckan(client, search_terms)

    # Strategy 2: Fallback to local catalog
    if not ckan_dataset_name:
        for kw in search_terms:
            results = _catalog_index.search(query=kw, limit=5)
            if results:
                catalog_entry = results[0]
                dataset_name = catalog_entry["title"] or catalog_entry["name"]
                ckan_dataset_name = catalog_entry["name"]
                logger.info(f"Local catalog found '{dataset_name}' via keyword '{kw}'")
                break

    if not ckan_dataset_name:
        raise HTTPException(
            status_code=404,
            detail="No dataset found matching your question. Try different keywords.",
        )

    # Check if already ingested
    existing = _metadata_store.list_datasets()
    for ds in existing:
        if ds.get("name") == dataset_name:
            logger.info(f"Reusing existing dataset: {ds['id']} ({dataset_name})")
            return ds["id"], dataset_name

    # Auto-ingest from CKAN
    logger.info(f"Auto-ingesting dataset: {ckan_dataset_name} ({dataset_name})")
    try:
        dataset = await client.get_dataset(ckan_dataset_name)

        supported = dataset.supported_resources
        if not supported:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset '{dataset_name}' has no supported file formats (CSV/JSON/Excel).",
            )

        resource = supported[0]
        df = await client.download_resource_as_dataframe(resource)

        dataset_id = str(uuid.uuid4())[:8]
        save_snapshot(df, dataset_id)
        _metadata_store.register_dataset(
            dataset_id=dataset_id,
            name=dataset_name,
            source_url=resource.url,
            description=dataset.notes,
            row_count=len(df),
        )

        return dataset_id, dataset_name

    except CKANError as e:
        raise HTTPException(status_code=502, detail=f"Failed to download dataset: {e}")


@router.post("/ask")
async def ask_narrative(request: AskRequest):
    """
    Unified endpoint: user asks a question → system finds the right dataset,
    ingests it, profiles, analyzes, generates a narrative, and returns the
    complete editorial data story.

    This is the primary endpoint for the simplified single-prompt interface.
    """
    try:
        # Step 1: Find and ingest the right dataset
        dataset_id, dataset_name = await _find_and_ingest_dataset(request.user_message)

        # Step 2: Load and profile
        df = load_snapshot(dataset_id)
        profile = profile_dataset(df, dataset_id=dataset_id)
        match = match_template(profile)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"Could not analyze dataset '{dataset_name}'. "
                       f"Column types found: {profile.column_types_summary}",
            )

        # Step 3: Build evidence bundle
        builder = BundleBuilder()
        bundle = builder.build(df, profile, match, title=request.title)

        # Step 4: Generate narrative with LLM (fall back to preview if unavailable)
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

        # Step 5: Return response with dataset info
        response = package.to_api_response()
        response["dataset_name"] = dataset_name
        response["dataset_source"] = bundle.source
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask narrative failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))