"""
LLM-assisted chart title/caption generation.
Keeps chart specs deterministic while improving chart readability.
"""

import re
from typing import Any, Optional

from data.analytics.evidence_bundle import EvidenceBundle

MAX_TITLE_WORDS = 12


def _trim_to_words(text: str, max_words: int = MAX_TITLE_WORDS) -> str:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip(" ,;:-")


def _clean_subject(text: str) -> str:
    return re.sub(r"[_\-]+", " ", (text or "")).strip()


def _contains_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in (text or ""))


def _ascii_fold(text: str) -> str:
    """Lowercase text and strip non-ascii chars for matching heuristics."""
    return "".join(ch.lower() if ord(ch) < 128 else " " for ch in (text or ""))


class ChartLabeler:
    """Generate deterministic chart titles by language."""

    def __init__(self):
        pass

    async def relabel_bundle(
        self,
        bundle: EvidenceBundle,
        language: str = "en",
        narrative: Optional[Any] = None,
    ) -> None:
        """
        Update chart titles deterministically (no LLM calls).
        """
        if not bundle.visualizations:
            return
        lang = (language or "en").lower()
        self._apply_deterministic_titles(bundle, language=lang, narrative=narrative)
        return

    def _apply_deterministic_titles(
        self,
        bundle: EvidenceBundle,
        language: str,
        narrative: Optional[Any],
    ) -> None:
        """Force explanatory deterministic titles for supported languages."""
        story_hints = self._chart_story_hints(bundle, narrative)
        date_range = bundle.metrics.get("date_range", {})
        period = ""
        if date_range.get("min") and date_range.get("max"):
            period = f"{str(date_range['min'])[:4]}\u2013{str(date_range['max'])[:4]}"
        subject = self._localized_subject(bundle, language)

        seen: set[str] = set()
        for idx, viz in enumerate(bundle.visualizations):
            metric_phrase = self._infer_metric_phrase_for_chart(
                bundle,
                idx,
                story_hints.get(idx, ""),
                language=language,
            )
            title = self._compose_localized_title(
                metric_phrase=metric_phrase,
                subject=subject,
                period=period,
                template_type=bundle.template_type,
                language=language,
            )

            key = title.lower()
            if key in seen:
                qualifier_map = {
                    "en": {
                        "line": "trend",
                        "bar": "comparison",
                        "area": "composition",
                        "scatter": "relationship",
                        "map": "locations",
                        "bubble_map": "locations",
                        "heatmap": "intensity",
                    },
                    "is": {
                        "line": "þróun",
                        "bar": "samanburður",
                        "area": "samsetning",
                        "scatter": "samband",
                        "map": "staðsetningar",
                        "bubble_map": "staðsetningar",
                        "heatmap": "styrkleiki",
                    },
                    "fi": {
                        "line": "trendi",
                        "bar": "vertailu",
                        "area": "koostumus",
                        "scatter": "suhde",
                        "map": "sijainnit",
                        "bubble_map": "sijainnit",
                        "heatmap": "intensiteetti",
                    },
                }
                lang_key = "en"
                if language.startswith("is"):
                    lang_key = "is"
                elif language.startswith("fi"):
                    lang_key = "fi"
                qualifier = qualifier_map[lang_key].get(viz.chart_type, "view")
                title = f"{title} — {qualifier}"
            seen.add(title.lower())

            viz.title = title[:120]
            viz.vega_lite_spec["title"] = viz.title
            cfg = viz.vega_lite_spec.setdefault("config", {})
            cfg["title"] = {
                **cfg.get("title", {}),
                "anchor": "start",
                "orient": "top",
            }

    def _infer_metric_phrase_for_chart(
        self,
        bundle: EvidenceBundle,
        chart_index: int,
        story_hint: str = "",
        language: str = "en",
    ) -> str:
        """
        Infer a human-readable metric phrase for one chart.
        Uses story hint text first, then chart encoding, then bundle-level fallback.
        """
        hint = _ascii_fold(story_hint)
        if "waste per person" in hint or "per citizen" in hint or "per capita" in hint:
            return self._localized_metric("waste_per_citizen", language)
        if "waste container" in hint or ("waste" in hint and "container" in hint):
            return self._localized_metric("waste_containers_total", language)
        if "public transport" in hint or "trips" in hint or "ridership" in hint:
            return self._localized_metric("public_transport_use", language)
        if "winter service" in hint:
            return self._localized_metric("winter_services_cost", language)

        viz = bundle.visualizations[chart_index]
        enc = viz.vega_lite_spec.get("encoding", {})
        y_field = str(enc.get("y", {}).get("field", "") or "")
        if y_field and y_field != "value":
            return self._infer_metric_phrase(bundle, y_field, language=language)

        # Folded measures (multi-measure charts)
        transforms = viz.vega_lite_spec.get("transform", [])
        if transforms and isinstance(transforms, list):
            first = transforms[0]
            if isinstance(first, dict) and isinstance(first.get("fold"), list):
                candidates = [str(c) for c in first.get("fold", []) if c]
                if candidates:
                    if hint:
                        best = self._pick_best_measure_from_hint(candidates, hint)
                        if best:
                            return self._infer_metric_phrase(bundle, best, language=language)
                    return self._infer_metric_phrase(bundle, candidates[0], language=language)

        measure_cols = bundle.metrics.get("measure_columns", [])
        metric_raw = str(measure_cols[0]) if measure_cols else "value"
        return self._infer_metric_phrase(bundle, metric_raw, language=language)

    def _pick_best_measure_from_hint(self, measures: list[str], hint_ascii: str) -> Optional[str]:
        """Pick measure whose tokens best overlap with the narrative hint."""
        best = None
        best_score = 0
        for m in measures:
            tokens = [t for t in re.split(r"[_\-\s]+", _ascii_fold(m)) if t]
            score = sum(1 for t in tokens if len(t) > 2 and t in hint_ascii)
            if score > best_score:
                best_score = score
                best = m
        return best

    def _chart_story_hints(self, bundle: EvidenceBundle, narrative: Optional[Any]) -> dict[int, str]:
        """
        Map chart index -> narrative text that references that chart.
        This gives the label model chart-specific context from the generated story.
        """
        hints: dict[int, str] = {}
        if not narrative:
            return hints
        blocks = getattr(narrative, "story_blocks", []) or []
        for block in blocks:
            idx = getattr(block, "viz_index", None)
            if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(bundle.visualizations):
                continue
            heading = getattr(block, "heading", "") or ""
            body = getattr(block, "body", "") or ""
            text = f"{heading}. {body}".strip(" .")
            if text:
                hints[idx] = text[:240]
        return hints

    def _infer_metric_phrase(self, bundle: EvidenceBundle, metric_raw: str, language: str = "en") -> str:
        """
        Infer a more descriptive English metric phrase than generic 'Count/Value'.
        Uses lightweight heuristics + keyword dictionary canonicals.
        """
        candidates = [metric_raw, bundle.dataset_id, *bundle.matched_columns.values()]
        folded = " ".join(_ascii_fold(c) for c in candidates if c)

        # Domain heuristics for known Icelandic/English patterns
        if ("kilo" in folded or "kg" in folded) and (
            "ibua" in folded or "resident" in folded or "capita" in folded
        ):
            return self._localized_metric("waste_per_citizen", language)
        if ("sorp" in folded or "waste" in folded) and (
            "ilat" in folded or "container" in folded or "bin" in folded
        ):
            return self._localized_metric("waste_containers_total", language)
        if "sorp" in folded or "waste" in folded:
            return self._localized_metric("waste", language)
        if "budget" in folded or "fjarmal" in folded or "uppgjor" in folded:
            return self._localized_metric("city_budget", language)
        if "ridership" in folded or "farthega" in folded or "almenningssamg" in folded:
            return self._localized_metric("public_transport_use", language)
        if "vetrar" in folded or "winter" in folded:
            return self._localized_metric("winter_services_cost", language)

        # Canonical fallback from keyword dictionary, excluding generic labels
        try:
            from data.profiling.keyword_dictionary import resolve_column
            signal = resolve_column(metric_raw)
            for canonical in signal.matched_canonicals:
                pretty = canonical.replace("_", " ").title()
                if pretty.lower() not in {"count", "value", "total", "amount"}:
                    return pretty
            if signal.matched_canonicals:
                return self._localize_canonical(signal.matched_canonicals[0], language)
        except Exception:
            pass

        # Raw fallback if ASCII and not generic
        if metric_raw and not _contains_non_ascii(metric_raw):
            raw_pretty = _clean_subject(metric_raw).title()
            if raw_pretty.lower() not in {"count", "value", "total", "amount"}:
                return raw_pretty

        # Last resort by template archetype
        if bundle.template_type == "time_series":
            return self._localized_metric("city_metric", language)
        if bundle.template_type == "categorical":
            return self._localized_metric("category_comparison", language)
        return self._localized_metric("city_metric", language)

    def _localized_metric(self, key: str, language: str) -> str:
        lang = "en"
        if language.startswith("is"):
            lang = "is"
        elif language.startswith("fi"):
            lang = "fi"
        labels = {
            "waste_per_citizen": {
                "en": "Kilogram of waste per citizen",
                "is": "Kíló af sorpi á hvern íbúa",
                "fi": "Jätemäärä kiloina asukasta kohden",
            },
            "waste_containers_total": {
                "en": "Total amount of waste containers",
                "is": "Heildarfjöldi sorpíláta",
                "fi": "Jäteastioiden kokonaismäärä",
            },
            "public_transport_use": {
                "en": "Use of public transportation",
                "is": "Notkun almenningssamgangna",
                "fi": "Joukkoliikenteen käyttö",
            },
            "winter_services_cost": {
                "en": "Cost of winter services",
                "is": "Kostnaður vetrarþjónustu",
                "fi": "Talvipalvelujen kustannukset",
            },
            "city_budget": {
                "en": "City budget",
                "is": "Fjárhagsáætlun borgarinnar",
                "fi": "Kaupungin budjetti",
            },
            "waste": {
                "en": "Waste",
                "is": "Sorp",
                "fi": "Jäte",
            },
            "city_metric": {
                "en": "City metric",
                "is": "Mælikvarði borgarinnar",
                "fi": "Kaupungin mittari",
            },
            "category_comparison": {
                "en": "Category comparison",
                "is": "Samanburður flokka",
                "fi": "Luokkien vertailu",
            },
        }
        return labels.get(key, labels["city_metric"]).get(lang, labels["city_metric"]["en"])

    def _localize_canonical(self, canonical: str, language: str) -> str:
        """Localize common canonical terms for deterministic titles."""
        lang = "en"
        if language.startswith("is"):
            lang = "is"
        elif language.startswith("fi"):
            lang = "fi"

        canonical_map = {
            "count": {"en": "Count", "is": "Fjöldi", "fi": "Määrä"},
            "value": {"en": "Value", "is": "Gildi", "fi": "Arvo"},
            "cost": {"en": "Cost", "is": "Kostnaður", "fi": "Kustannus"},
            "amount": {"en": "Amount", "is": "Magn", "fi": "Määrä"},
            "population": {"en": "Population", "is": "Íbúafjöldi", "fi": "Väestö"},
            "traffic": {"en": "Traffic", "is": "Umferð", "fi": "Liikenne"},
            "ridership": {"en": "Ridership", "is": "Farþegafjöldi", "fi": "Matkustajamäärä"},
        }
        if canonical in canonical_map:
            return canonical_map[canonical][lang]
        return canonical.replace("_", " ").title()

    def _localized_subject(self, bundle: EvidenceBundle, language: str) -> str:
        """Best-effort location label by language for deterministic titles."""
        if bundle.normalized_metadata and bundle.normalized_metadata.organization.available:
            org = _clean_subject(bundle.normalized_metadata.organization.value).strip()
            if org:
                return org

        city = self._city_from_source(bundle.source, language)
        if city:
            return city

        generic = {"is": "borgin", "fi": "kaupunki"}
        lang_key = "is" if language.startswith("is") else ("fi" if language.startswith("fi") else "en")
        return generic.get(lang_key, "the city")

    @staticmethod
    def _city_from_source(source: str, language: str) -> Optional[str]:
        """Extract a city name from a portal URL or source string."""
        folded = _ascii_fold(source)
        known_cities = [
            ("reykjavik", {"is": "Reykjavík", "default": "Reykjavik"}),
            ("helsinki", {"fi": "Helsinki", "default": "Helsinki"}),
            ("hel.fi", {"fi": "Helsinki", "default": "Helsinki"}),
            ("oslo", {"no": "Oslo", "default": "Oslo"}),
            ("copenhagen", {"da": "København", "default": "Copenhagen"}),
            ("kobenhavn", {"da": "København", "default": "Copenhagen"}),
            ("stockholm", {"sv": "Stockholm", "default": "Stockholm"}),
            ("tampere", {"fi": "Tampere", "default": "Tampere"}),
            ("turku", {"fi": "Turku", "default": "Turku"}),
            ("barcelona", {"es": "Barcelona", "default": "Barcelona"}),
            ("madrid", {"es": "Madrid", "default": "Madrid"}),
        ]
        for keyword, names in known_cities:
            if keyword in folded:
                return names.get(language[:2], names["default"])
        return None

    def _compose_localized_title(
        self,
        metric_phrase: str,
        subject: str,
        period: str,
        template_type: str,
        language: str,
    ) -> str:
        """Compose a deterministic title with language-specific phrasing."""
        lang = "en"
        if language.startswith("is"):
            lang = "is"
        elif language.startswith("fi"):
            lang = "fi"

        if lang == "is":
            title = f"{metric_phrase} í {subject}"
            if period and template_type == "time_series":
                title = f"{title} eftir árum ({period})"
            return title
        if lang == "fi":
            title = f"{metric_phrase} kaupungissa {subject}"
            if period and template_type == "time_series":
                title = f"{title} vuosittain ({period})"
            return title

        title = f"{metric_phrase} in {subject}"
        if period and template_type == "time_series":
            title = f"{title} by year ({period})"
        return title
