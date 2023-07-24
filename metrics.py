from dataclass import dataclass

from graph import Address


def analyze_memory_access(addresses: list[Address]) -> MemStat:
    mem_accesses = [(x, i) for i, x in enumerate(addresses)].sort()

