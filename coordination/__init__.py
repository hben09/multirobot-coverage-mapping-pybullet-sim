"""
Coordination package - Multi-robot coordination components.

Contains:
- UtilityCalculator: Computes utility values for frontier assignments
- TaskAllocator: Market-based task allocation for multi-robot systems
"""

from .utility_calculator import FrontierUtilityCalculator
from .task_allocator import TaskAllocator

__all__ = ['FrontierUtilityCalculator', 'TaskAllocator']
