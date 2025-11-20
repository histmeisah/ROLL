"""
Prompt Template Loader for Mathematical Reasoning.

Loads prompt templates from YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Prompt template with metadata."""
    name: str
    template: str
    description: str
    category: str
    paper_reference: Optional[str] = None
    expected_performance: Optional[str] = None

    def format(self, problem: str) -> str:
        """Format the prompt template with the given problem."""
        return self.template.format(problem=problem)

    def __repr__(self):
        return f"PromptTemplate(name='{self.name}', category='{self.category}')"


class PromptLoader:
    """Loader for prompt templates from YAML configuration."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize prompt loader.

        Args:
            config_path: Path to YAML config file. If None, uses default path.
        """
        if config_path is None:
            # Default path relative to this file
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "prompts" / "math_reasoning_prompts.yaml"

        self.config_path = Path(config_path)
        self.prompts: Dict[str, PromptTemplate] = {}
        self.presets: Dict[str, Dict] = {}
        self.categories: Dict[str, Dict] = {}

        self._load_config()

    def _load_config(self):
        """Load prompts from YAML configuration file."""
        if not self.config_path.exists():
            logger.warning(f"Prompt config file not found: {self.config_path}")
            logger.warning("Using empty prompt library")
            return

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Load prompts
        for prompt_dict in config.get('prompts', []):
            prompt = PromptTemplate(
                name=prompt_dict['name'],
                template=prompt_dict['template'],
                description=prompt_dict['description'],
                category=prompt_dict.get('category', 'unknown'),
                paper_reference=prompt_dict.get('paper_reference'),
                expected_performance=prompt_dict.get('expected_performance'),
            )
            self.prompts[prompt.name] = prompt

        # Load presets
        self.presets = config.get('presets', {})

        # Load categories
        self.categories = config.get('categories', {})

        logger.info(f"Loaded {len(self.prompts)} prompts from {self.config_path}")
        logger.info(f"Available presets: {list(self.presets.keys())}")

    def get_prompt(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a prompt by name.

        Args:
            name: Prompt name

        Returns:
            PromptTemplate or None if not found
        """
        return self.prompts.get(name)

    def get_all_prompts(self) -> List[PromptTemplate]:
        """Get all available prompts."""
        return list(self.prompts.values())

    def get_prompts_by_category(self, category: str) -> List[PromptTemplate]:
        """
        Get all prompts of a specific category.

        Args:
            category: Category name

        Returns:
            List of prompts in that category
        """
        return [p for p in self.prompts.values() if p.category == category]

    def get_preset_prompts(self, preset_name: str) -> List[PromptTemplate]:
        """
        Get prompts from a preset configuration.

        Args:
            preset_name: Name of the preset

        Returns:
            List of prompt templates

        Raises:
            ValueError: If preset not found
        """
        if preset_name not in self.presets:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {list(self.presets.keys())}"
            )

        preset = self.presets[preset_name]
        prompt_names = preset.get('prompts', [])

        prompts = []
        for name in prompt_names:
            prompt = self.get_prompt(name)
            if prompt:
                prompts.append(prompt)
            else:
                logger.warning(f"Prompt '{name}' not found in preset '{preset_name}'")

        return prompts

    def get_random_prompts(self, n: int, exclude_category: Optional[str] = None, seed: int = 42) -> List[PromptTemplate]:
        """
        Get n random prompts.

        Args:
            n: Number of prompts to select
            exclude_category: Optional category to exclude (e.g., 'direct' for baselines)
            seed: Random seed

        Returns:
            List of randomly selected prompts
        """
        import random
        rng = random.Random(seed)

        available_prompts = self.get_all_prompts()

        if exclude_category:
            available_prompts = [p for p in available_prompts if p.category != exclude_category]

        if len(available_prompts) < n:
            logger.warning(
                f"Requested {n} prompts but only {len(available_prompts)} available"
            )
            return available_prompts

        return rng.sample(available_prompts, n)

    def list_categories(self) -> List[str]:
        """Get list of all prompt categories."""
        return list(set(p.category for p in self.prompts.values()))

    def list_presets(self) -> List[str]:
        """Get list of all available presets."""
        return list(self.presets.keys())

    def get_preset_info(self, preset_name: str) -> Dict:
        """
        Get information about a preset.

        Args:
            preset_name: Name of the preset

        Returns:
            Dictionary with preset information
        """
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}")

        preset = self.presets[preset_name]
        prompts = self.get_preset_prompts(preset_name)

        return {
            "name": preset_name,
            "description": preset.get('description', ''),
            "num_prompts": len(prompts),
            "prompt_names": [p.name for p in prompts],
            "categories": list(set(p.category for p in prompts)),
        }

    def get_summary(self) -> Dict:
        """Get summary of loaded prompts."""
        summary = {
            "total_prompts": len(self.prompts),
            "categories": {},
            "presets": list(self.presets.keys()),
            "config_path": str(self.config_path),
        }

        for category in self.list_categories():
            prompts = self.get_prompts_by_category(category)
            summary["categories"][category] = {
                "count": len(prompts),
                "names": [p.name for p in prompts],
            }

        return summary

    def print_summary(self):
        """Print a formatted summary of loaded prompts."""
        summary = self.get_summary()

        print(f"\n{'='*70}")
        print(f"Prompt Library Summary")
        print(f"{'='*70}")
        print(f"Config: {summary['config_path']}")
        print(f"Total Prompts: {summary['total_prompts']}")
        print(f"\nCategories:")
        for cat, info in summary['categories'].items():
            print(f"  - {cat}: {info['count']} prompts")

        print(f"\nAvailable Presets:")
        for preset in summary['presets']:
            preset_info = self.get_preset_info(preset)
            print(f"  - {preset}: {preset_info['num_prompts']} prompts")
            print(f"    {preset_info['description']}")

        print(f"{'='*70}\n")


# ============================================================================
# Convenience Functions
# ============================================================================

_default_loader: Optional[PromptLoader] = None


def get_default_loader() -> PromptLoader:
    """Get the default prompt loader (singleton)."""
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader


def load_prompts(config_path: Optional[Union[str, Path]] = None) -> List[PromptTemplate]:
    """
    Load all prompts from config file.

    Args:
        config_path: Optional path to config file

    Returns:
        List of all prompt templates
    """
    loader = PromptLoader(config_path) if config_path else get_default_loader()
    return loader.get_all_prompts()


def load_preset(preset_name: str, config_path: Optional[Union[str, Path]] = None) -> List[PromptTemplate]:
    """
    Load prompts from a preset configuration.

    Args:
        preset_name: Name of the preset
        config_path: Optional path to config file

    Returns:
        List of prompt templates
    """
    loader = PromptLoader(config_path) if config_path else get_default_loader()
    return loader.get_preset_prompts(preset_name)


def load_prompt_by_name(name: str, config_path: Optional[Union[str, Path]] = None) -> Optional[PromptTemplate]:
    """
    Load a specific prompt by name.

    Args:
        name: Prompt name
        config_path: Optional path to config file

    Returns:
        PromptTemplate or None if not found
    """
    loader = PromptLoader(config_path) if config_path else get_default_loader()
    return loader.get_prompt(name)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Load prompts
    loader = PromptLoader()

    # Print summary
    loader.print_summary()

    # Example: Get a specific prompt
    print("\nExample: Zero-shot CoT APE prompt")
    print("-" * 70)
    prompt = loader.get_prompt("zero_shot_cot_ape")
    if prompt:
        print(f"Name: {prompt.name}")
        print(f"Description: {prompt.description}")
        print(f"Template: {prompt.template}")

        # Format with a problem
        problem = "John has 5 apples. He gives 2 to Mary. How many does John have?"
        formatted = prompt.format(problem)
        print(f"\nFormatted:\n{formatted}")

    # Example: Load a preset
    print("\n\nExample: Load 'research_backed_8' preset")
    print("-" * 70)
    prompts = loader.get_preset_prompts("research_backed_8")
    print(f"Loaded {len(prompts)} prompts:")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. {p.name} ({p.category})")

    # Example: Get prompts by category
    print("\n\nExample: Zero-shot CoT prompts")
    print("-" * 70)
    cot_prompts = loader.get_prompts_by_category("zero_shot_cot")
    for p in cot_prompts:
        print(f"  - {p.name}: {p.description}")