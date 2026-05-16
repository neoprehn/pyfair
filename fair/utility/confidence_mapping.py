"""Confidence mapping utilities for FAIR distribution defaults.

This module centralizes the mapping between qualitative confidence levels
and the default shape/range parameters used for supported distributions.
"""

from .fair_exception import FairException


# Default shape/range parameters used when neither confidence nor
# an explicit shape parameter is provided.
DISTRIBUTION_DEFAULTS = {
    'beta': {'k': 15},
    'lognormal': {'sigma': 0.6},
    'poisson': {'range': 0.4},
    'pert': {'gamma': 4},
}


# Confidence-to-parameter mapping.
# These values reflect the current agreed default table.
CONFIDENCE_DEFAULTS = {
    'very_low': {
        'beta': {'k': 4},
        'lognormal': {'sigma': 0.25},
        'poisson': {'range': 2.5},
        'pert': {'gamma': 10},
    },
    'low': {
        'beta': {'k': 7},
        'lognormal': {'sigma': 0.4},
        'poisson': {'range': 0.75},
        'pert': {'gamma': 8},
    },
    'moderate': {
        'beta': {'k': 15},
        'lognormal': {'sigma': 0.6},
        'poisson': {'range': 0.4},
        'pert': {'gamma': 4},
    },
    'high': {
        'beta': {'k': 40},
        'lognormal': {'sigma': 0.9},
        'poisson': {'range': 0.25},
        'pert': {'gamma': 3},
    },
    'very_high': {
        'beta': {'k': 100},
        'lognormal': {'sigma': 1.2},
        'poisson': {'range': 0.15},
        'pert': {'gamma': 2},
    },
}


def get_distribution_defaults():
    """Return the default shape/range parameters by distribution."""
    return DISTRIBUTION_DEFAULTS


def get_confidence_defaults():
    """Return the confidence-to-parameter mapping."""
    return CONFIDENCE_DEFAULTS


def get_default_for_distribution(distribution):
    """Return default shape/range parameters for a distribution."""
    try:
        return dict(DISTRIBUTION_DEFAULTS[distribution])
    except KeyError as exc:
        raise FairException(
            f'"{distribution}" is not a supported distribution for defaults.'
        ) from exc


def get_default_for_confidence(confidence, distribution):
    """Return confidence-based shape/range parameters for a distribution."""
    try:
        return dict(CONFIDENCE_DEFAULTS[confidence][distribution])
    except KeyError as exc:
        raise FairException(
            f'No confidence mapping found for confidence="{confidence}" '
            f'and distribution="{distribution}".'
        ) from exc