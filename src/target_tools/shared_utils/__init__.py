"""
Shared utilities for TypeEvalPy runners.

This module provides common functionality used across multiple type inference tool runners,
including source code parsing and type normalization.
"""

from .codeindex import ModuleIndex, Position, FuncIndex
from .type_normalizer import normalize_types, strip_builtins_prefix

__all__ = [
    'ModuleIndex',
    'Position',
    'FuncIndex',
    'normalize_types',
    'strip_builtins_prefix',
]
