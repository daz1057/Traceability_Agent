"""User story parsing utilities."""

from __future__ import annotations

import re
from typing import List

from .data_models import ParsedStory, RawStory, iter_domain_terms
from .text_utils import keyphrase_candidates

STORY_PATTERN = re.compile(
    r"as an? (?P<persona>[^,]+?),\s*i want (?P<capability>[^.]+?)(?:,?\s*so that (?P<outcome>[^.]+))?",
    re.IGNORECASE,
)


GOVERNANCE_KEYWORDS = {
    "audit",
    "auditable",
    "policy",
    "control",
    "governance",
    "approval",
    "workflow",
    "compliance",
    "lineage",
    "role",
    "security",
    "permission",
    "access",
}


def governance_signal(text: str) -> int:
    """Return governance signal strength derived from the story text."""

    lowered = text.lower()
    if any(keyword in lowered for keyword in ("policy", "audit", "governance", "lineage")):
        return 2
    if any(keyword in lowered for keyword in GOVERNANCE_KEYWORDS):
        return 1
    return 0


def parse_story(story: RawStory) -> ParsedStory:
    """Parse a user story into comparable facets."""

    match = STORY_PATTERN.search(story.text)

    persona = "Stakeholder"
    capability = story.text.strip()
    functional_outcome = story.text.strip()
    business_value = story.text.strip()

    if match:
        persona = match.group("persona").strip()
        capability = match.group("capability").strip()
        functional_outcome = (
            match.group("outcome").strip() if match.group("outcome") else capability
        )
        business_value = functional_outcome
    elif "i need" in story.text.lower():
        need_match = re.search(r"i need to ([^.,]+)", story.text, flags=re.IGNORECASE)
        if need_match:
            capability = need_match.group(1).strip()
            functional_outcome = capability
            business_value = capability

    domain_terms = iter_domain_terms(keyphrase_candidates(f"{capability} {functional_outcome}"))
    governance = governance_signal(story.text)

    return ParsedStory(
        story_id=story.story_id,
        raw_text=story.text,
        persona=persona,
        capability=capability,
        functional_outcome=functional_outcome,
        business_value=business_value,
        domain_terms=domain_terms,
        governance_signal=governance,
        metadata=story.metadata,
    )


def parse_stories(stories: List[RawStory]) -> List[ParsedStory]:
    """Parse each story in a list."""

    return [parse_story(story) for story in stories]
