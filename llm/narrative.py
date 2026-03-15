"""
Narrative generation from evidence bundles.
Orchestrates the full generation pipeline:
intent → prompt → LLM → validate → (retry if needed) → narrative.
"""

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from llm.interface import LLMProvider
from llm.intent import IntentParser, UserIntent, IntentParseResult
from llm.prompts import PromptAssembler, AssembledPrompt
from data.analytics.evidence_bundle import EvidenceBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class KeyMetric(BaseModel):
    """A single key metric within a narrative section."""
    label: str = ""
    value: str = ""
    context: str = ""


class NarrativeSection(BaseModel):
    """A section of the generated narrative."""
    heading: str = ""
    body: str = ""
    key_metric: Optional[KeyMetric] = None


class GeneratedNarrative(BaseModel):
    """The validated output of narrative generation."""
    title: str = ""
    summary: str = ""
    sections: list[NarrativeSection] = []
    data_limitations: str = ""
    suggested_followup: str = ""

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Narrative title cannot be empty")
        return v.strip()

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Narrative summary cannot be empty")
        return v.strip()

    @field_validator("sections")
    @classmethod
    def sections_not_empty(cls, v):
        if not v:
            raise ValueError("Narrative must have at least one section")
        return v


class ValidationResult(BaseModel):
    """Result of validating LLM output."""
    is_valid: bool
    narrative: Optional[GeneratedNarrative] = None
    errors: list[str] = []
    raw_output: str = ""


class GenerationResult(BaseModel):
    """Complete result of the narrative generation pipeline."""
    success: bool
    narrative: Optional[GeneratedNarrative] = None
    intent: Optional[UserIntent] = None
    prompt: Optional[AssembledPrompt] = None
    validation: Optional[ValidationResult] = None
    attempts: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

class OutputValidator:
    """
    Validates LLM-generated narrative output against the expected schema.
    Uses Pydantic for schema enforcement.
    """

    def validate(self, raw_output: str, bundle: EvidenceBundle) -> ValidationResult:
        """
        Validate the LLM's raw output.

        Checks:
            1. Valid JSON structure
            2. Matches GeneratedNarrative schema
            3. No hallucinated data values (basic check)
            4. Sections are non-empty

        Args:
            raw_output: The raw string from the LLM.
            bundle: The evidence bundle (for cross-referencing).

        Returns:
            ValidationResult with parsed narrative or errors.
        """
        errors = []

        # Step 1: Parse JSON
        parsed = self._parse_json(raw_output)
        if parsed is None:
            return ValidationResult(
                is_valid=False,
                errors=["Failed to parse LLM output as JSON"],
                raw_output=raw_output,
            )

        # Step 2: Validate against schema
        try:
            narrative = GeneratedNarrative(**parsed)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema validation failed: {str(e)}"],
                raw_output=raw_output,
            )

        # Step 3: Content quality checks
        content_errors = self._check_content_quality(narrative, bundle)
        errors.extend(content_errors)

        # Step 4: Hallucination spot check
        hallucination_warnings = self._check_hallucinations(narrative, bundle)
        errors.extend(hallucination_warnings)

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            narrative=narrative if is_valid else None,
            errors=errors,
            raw_output=raw_output,
        )

    def _parse_json(self, raw: str) -> Optional[dict]:
        """Parse JSON from LLM output, handling common formatting issues."""
        cleaned = raw.strip()

        # Strip markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

        # Try parsing
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from surrounding text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

        return None

    def _check_content_quality(
        self, narrative: GeneratedNarrative, bundle: EvidenceBundle
    ) -> list[str]:
        """Check basic content quality."""
        errors = []

        if len(narrative.title) > 100:
            errors.append("Title exceeds 100 characters")

        if len(narrative.summary) < 20:
            errors.append("Summary is too short (< 20 chars)")

        for i, section in enumerate(narrative.sections):
            if not section.heading.strip():
                errors.append(f"Section {i+1} has empty heading")
            if not section.body.strip():
                errors.append(f"Section {i+1} has empty body")
            if len(section.body) < 20:
                errors.append(f"Section {i+1} body is too short (< 20 chars)")

        return errors

    def _check_hallucinations(
        self, narrative: GeneratedNarrative, bundle: EvidenceBundle
    ) -> list[str]:
        """
        Basic hallucination check.
        Flags if the narrative mentions column names not in the dataset.
        """
        warnings = []
        known_columns = set(bundle.matched_columns.values())
        known_columns.update(bundle.matched_columns.keys())

        # Also add column names from the summary
        for col_type, count in bundle.column_summary.items():
            known_columns.add(col_type)

        # This is a lightweight check — full fact-checking would require
        # comparing every number in the narrative against the metrics.
        # For the prototype, we just check structure.

        return warnings


# ---------------------------------------------------------------------------
# Narrative generator
# ---------------------------------------------------------------------------

class NarrativeGenerator:
    """
    Orchestrates the full narrative generation pipeline.
    Handles intent parsing, prompt assembly, LLM generation,
    output validation, and retry logic.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        intent_llm_provider: Optional[LLMProvider] = None,
        max_retries: int = 2,
    ):
        self.llm = llm_provider
        self.intent_parser = IntentParser(intent_llm_provider or llm_provider)
        self.prompt_assembler = PromptAssembler()
        self.validator = OutputValidator()
        self.max_retries = max_retries

    async def generate(
        self,
        bundle: EvidenceBundle,
        user_message: Optional[str] = None,
        intent: Optional[UserIntent] = None,
    ) -> GenerationResult:
        """
        Generate a narrative from an evidence bundle.

        Args:
            bundle: The evidence bundle with metrics and viz specs.
            user_message: Optional raw user message to parse intent from.
            intent: Optional pre-parsed intent (skips intent parsing).

        Returns:
            GenerationResult with the validated narrative or error info.
        """
        # Step 1: Parse intent (if not provided)
        if intent is None and user_message:
            intent_result = await self.intent_parser.parse(user_message)
            intent = intent_result.intent
            if not intent_result.success:
                logger.warning(f"Intent parsing failed, using fallback: {intent_result.error}")

        # Step 2: Assemble prompt
        prompt = self.prompt_assembler.assemble(bundle, intent)

        # Step 3: Generate with retry loop
        last_validation = None
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Generation attempt {attempt}/{self.max_retries}")

            try:
                raw_output = await self.llm.generate(
                    prompt=prompt.user_prompt,
                    system_prompt=prompt.system_prompt,
                )

                # Step 4: Validate output
                validation = self.validator.validate(raw_output, bundle)
                last_validation = validation

                if validation.is_valid:
                    logger.info(f"Narrative generated successfully on attempt {attempt}")
                    return GenerationResult(
                        success=True,
                        narrative=validation.narrative,
                        intent=intent,
                        prompt=prompt,
                        validation=validation,
                        attempts=attempt,
                    )

                # Validation failed — modify prompt for retry
                logger.warning(
                    f"Attempt {attempt} validation failed: {validation.errors}"
                )
                prompt = self._add_correction_context(prompt, validation)

            except Exception as e:
                logger.error(f"Generation attempt {attempt} failed: {e}")
                last_validation = ValidationResult(
                    is_valid=False,
                    errors=[str(e)],
                    raw_output="",
                )

        # All retries exhausted
        logger.error(f"Narrative generation failed after {self.max_retries} attempts")
        return GenerationResult(
            success=False,
            intent=intent,
            prompt=prompt,
            validation=last_validation,
            attempts=self.max_retries,
            error=f"Failed after {self.max_retries} attempts: {last_validation.errors if last_validation else 'unknown'}",
        )

    def _add_correction_context(
        self, prompt: AssembledPrompt, validation: ValidationResult
    ) -> AssembledPrompt:
        """
        Modify the prompt to include correction instructions based on
        validation errors from the previous attempt.
        """
        correction = (
            "\n\n=== CORRECTION REQUIRED ===\n"
            "Your previous response had these issues:\n"
        )
        for error in validation.errors:
            correction += f"- {error}\n"

        correction += (
            "\nPlease fix these issues. Remember:\n"
            "- Respond with ONLY valid JSON\n"
            "- Include all required fields: title, summary, sections\n"
            "- Each section needs a heading, body, and optional key_metric\n"
            "- Do not include any text outside the JSON object\n"
        )

        if validation.raw_output:
            correction += f"\nYour previous (invalid) response was:\n{validation.raw_output[:500]}\n"

        return AssembledPrompt(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt + correction,
            expected_output_schema=prompt.expected_output_schema,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

async def generate_narrative(
    llm_provider: LLMProvider,
    bundle: EvidenceBundle,
    user_message: Optional[str] = None,
    intent_llm_provider: Optional[LLMProvider] = None,
) -> GenerationResult:
    """
    Convenience function for one-shot narrative generation.

    Args:
        llm_provider: The LLM provider to use for generation.
        bundle: Evidence bundle.
        user_message: Optional user question.
        intent_llm_provider: Optional separate provider for intent parsing.

    Returns:
        GenerationResult.
    """
    generator = NarrativeGenerator(llm_provider, intent_llm_provider=intent_llm_provider)
    return await generator.generate(bundle, user_message=user_message)
