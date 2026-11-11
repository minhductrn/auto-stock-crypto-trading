"""Shared error hierarchy (spec section 9, interface rule 4).

Every package in this repository raises errors derived from ``SdfTradeError``.
Raising a bare ``Exception`` is forbidden.
"""

from __future__ import annotations


class SdfTradeError(Exception):
    """Base class for all project errors."""


class DataContractError(SdfTradeError):
    """Input data violates a data-quality or schema contract."""


class DataProviderError(SdfTradeError):
    """A vendor adapter failed (network, empty payload, unexpected shape)."""


class SchemaContractError(SdfTradeError):
    """An output frame does not match its declared schema."""


class LeakageError(SdfTradeError):
    """A feature or label used information outside its allowed window."""


class ConfigError(SdfTradeError):
    """Invalid or missing configuration."""


class RiskLimitError(SdfTradeError):
    """A portfolio or order violates a risk-management limit."""
