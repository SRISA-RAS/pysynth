from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union
from queue import SimpleQueue

from instruction_types import SUBTYPE_TO_INSTRUCTIONS


class Address(int):
    _instruction_size = 4

    def __str__(self):
        return f'{self:#x}'

    def __add__(self, x):
        return Address(self + x)

    def __iadd__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        return Address(self - x)

    def __isub__(self, x):
        return self.__sub__(x)

    def get_next(self, count: int = 1):
        return Address(self + count * Address._instruction_size)


@dataclass
class Instruction:
    __slots__ = ('program_counter', 'instruction')
    program_counter: Address
    instruction: str

    def __repr__(self):
        return f'{self.__class__.__name__}(PC="0x{self.program_counter:x}", "{self.instruction}")'


@dataclass
class FlowChangeEdge:
    __slots__ = ('from_addr', 'to_addr', 'count')
    from_addr: Address
    to_addr: Address
    count: int

    def __hash__(self):
        return (self.from_addr, self.to_addr).__hash__()

    def __repr__(self):
        return f'{self.__class__.__name__}(from {self.from_addr:#x} to {self.to_addr:#x}, count = {self.count})'


class FlowChange(ABC):
    @abstractmethod
    def add(self, edge: FlowChangeEdge):
        pass

    def __iter__(self):
        yield from ()

    def __reversed__(self):
        yield from ()

    def __repr__(self):
        return self.__class__.__name__


@dataclass
class Unconditional(FlowChange):
    __slots__ = ('edge',)
    edge: FlowChangeEdge

    def add(self, edge: FlowChangeEdge):
        raise Exception("Unexpected flowchange addition")

    def __iter__(self):
        yield self.edge

    def __reversed__(self):
        yield self.edge

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.edge}'


@dataclass
class Branch(FlowChange):
    __slots__ = ('taken', 'nottaken')
    taken: FlowChangeEdge
    nottaken: FlowChangeEdge

    def add(self, edge: FlowChangeEdge):
        if self.taken is None:
            assert edge.from_addr == self.nottaken.from_addr
            self.taken = edge
        elif self.nottaken is None:
            assert edge.from_addr == self.taken.from_addr
            self.nottaken = edge
        else:
            raise Exception("Unexpected flowchange addition")

    def __iter__(self):
        if self.nottaken is not None:
            yield self.nottaken
        if self.taken is not None:
            yield self.taken

    def __reversed__(self):
        if self.taken is not None:
            yield self.taken
        if self.nottaken is not None:
            yield self.nottaken

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.nottaken}, {self.taken}'


@dataclass
class Multiple(FlowChange):
    __slots__ = ('edges',)
    edges: list[FlowChangeEdge]

    def add(self, edge: FlowChangeEdge):
        assert edge.from_addr == self.edges[-1].from_addr
        self.edges.append(edge)

    def __iter__(self):
        yield from self.edges

    def __reversed__(self):
        yield from reversed(self.edges)

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.edges}'


@dataclass
class Exit(FlowChange):
    __slots__ = ()

    def add(self, edge: FlowChangeEdge):
        raise Exception("Unexpected flowchange addition")


@dataclass
class Fallthrough(FlowChange):
    __slots__ = ('to',)
    to: FlowChangeEdge

    def add(self, edge: FlowChangeEdge):
        raise Exception("Unexpected flowchange addition")

    def __iter__(self):
        yield self.to

    def __reversed__(self):
        yield self.to

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.to}'


@dataclass
class FuncCall(FlowChange):
    __slots__ = ('target_edge', 'return_edge')
    target_edge: FlowChangeEdge
    return_edge: FlowChangeEdge

    def add(self, edge: FlowChangeEdge):
        raise Exception("Unexpected flowchange addition")

    def __iter__(self):
        yield self.target_edge
        yield self.return_edge

    def __reversed__(self):
        yield self.target_edge
        yield self.return_edge

    def __repr__(self):
        return f'{self.__class__.__name__}: call edge is {self.target_edge}, return edge is {self.return_edge}'


@dataclass
class CondFuncCall(FlowChange):
    __slots__ = ('target_edge', 'return_edge')
    target_edge: FlowChangeEdge
    return_edge: FlowChangeEdge

    def add(self, edge: FlowChangeEdge):
        if self.target_edge is None:
            assert edge.from_addr == self.return_edge.from_addr
            self.target_edge = edge
        elif self.return_edge is None:
            assert edge.from_addr == self.target_edge.from_addr
            self.return_edge = edge
        else:
            raise Exception("Unexpected flowchange addition")

    def __iter__(self):
        if self.return_edge is not None:
            yield self.return_edge
        if self.target_edge is not None:
            yield self.target_edge

    def __reversed__(self):
        if self.target_edge is not None:
            yield self.target_edge
        if self.return_edge is not None:
            yield self.return_edge

    def __repr__(self):
        return f'{self.__class__.__name__}: call edge is {self.target_edge}, return edge is {self.return_edge}'


@dataclass
class RegFuncCall(FlowChange):
    __slots__ = ('target_edges', 'return_edge')
    target_edges: list[FlowChangeEdge]
    return_edge: FlowChangeEdge

    def add(self, edge: FlowChangeEdge):
        assert edge.from_addr == self.target_edges[-1].from_addr
        self.target_edges.append(edge)

    def __iter__(self):
        yield from self.target_edges
        yield self.return_edge

    def __reversed__(self):
        yield from reversed(self.target_edges)
        yield self.return_edge

    def __repr__(self):
        return f'{self.__class__.__name__}: call edges are {self.target_edges}, return edge is {self.return_edge}'


@dataclass
class FuncExit(FlowChange):
    __slots__ = ()

    def add(self, edge: FlowChangeEdge):
        raise Exception("Unexpected flowchange addition")


FlowChangeType = Union[Unconditional, Branch, Multiple, Exit, Fallthrough, FuncCall, CondFuncCall, RegFuncCall, FuncExit]


@dataclass
class FlowNode:
    # waiting for 3.10 to correctly implement slots in dataclasses so default values would work with them
    __slots__ = ('address', 'end_addr', 'instructions', 'flowchange', 'metrics',
                 'is_synthetic', 'restores_regs', 'missing_delay_slot')
    address: Address
    end_addr: Address
    instructions: list[Instruction]
    flowchange: FlowChangeType
    metrics: NodeMetrics
    is_synthetic: bool      # if true then this node was created by the generator for internal usage
                            # and doesn't represent a part of an original application's control flow
    restores_regs: bool
    missing_delay_slot: bool  # move to a dict

    def __repr__(self):
        return (f'Node(Address="0x{self.address:x}", Instructions={len(self.instructions)}'
                f'{", Synthetic" if self.is_synthetic else ""})')


@dataclass
class NodeMetrics:
    __slots__ = 'instr_by_type',
    instr_by_type: dict[str, int]
    data_addresses: set[Address]
    # TODO: include other metrics:
    # Cache Hits/Misses
    # Memory Access Behaviour (stride values)
    # Branch Prediction Statistics
    # Data Dependencies (Dependency distance distribution, instruction level parallelism)
    # etc...


@dataclass
class FlowGraph:
    __slots__ = ('start_node', 'nodes', 'functions')
    start_node: FlowNode    # should be Address
    nodes: dict[Address, FlowNode]
    functions: set[Address]

    # TODO: redo/expand. for now only traverses the main part of the graph without entering any of the function
    def dfs(self):
        seen_nodes: set[Address] = set()
        node_stack: list[Address] = [self.start_node.address]

        while len(node_stack):
            cur_node_addr = node_stack.pop()
            if cur_node_addr in seen_nodes: # and isinstance(graph.nodes[cur_node_addr], FnCall)
                continue
            seen_nodes.add(cur_node_addr)
            node = self.nodes.get(cur_node_addr)
            if node is None:
                print(f'Warning: iteration over nonexistent nodes (no node with address {cur_node_addr:#x} in graph)')
                continue
            yield node
            if isinstance(node.flowchange, Branch):
                for next_edge in reversed(node.flowchange):
                    node_stack.append(next_edge.to_addr)
            elif isinstance(node.flowchange, (FuncCall, CondFuncCall, RegFuncCall)):
                node_stack.append(node.flowchange.return_edge.to_addr)
            elif isinstance(node.flowchange, FuncExit):
                raise Exception("Somehow iterated over FuncExit node {node.address:#x}")
            else:
                for next_edge in node.flowchange:
                    node_stack.append(next_edge.to_addr)

    #REDO
    # def bfs(self):
    #     seen_nodes: set[Address] = set()
    #     node_queue: SimpleQueue[tuple[Address, list[Address]]] = SimpleQueue()
    #     node_queue.put((self.start_node.address, []))
    #     # pending_returns: list[Address] to not put excessive amount of nodes into queue

    #     # print('BFS:')
    #     while not node_queue.empty():
    #         cur_node_addr, return_stack = node_queue.get()
    #         if cur_node_addr in seen_nodes:
    #             continue
    #         seen_nodes.add(cur_node_addr)
    #         # print(f'node 0x{cur_node_addr.hex()}')
    #         node = self.nodes[cur_node_addr]
    #         yield node
    #         if isinstance(node.flowchange, (FuncCall, CondFuncCall, RegFuncCall)):
    #             for next_block in node.flowchange:
    #                 next_addr = next_block.to_addr
    #                 if next_addr not in seen_nodes:
    #                     node_queue.put((next_addr, [*return_stack, node.flowchange.ret_addr]))
    #                 else:
    #                     for i, ret_address in enumerate(reversed(return_stack), 1):
    #                         if ret_address not in seen_nodes:
    #                             node_queue.put((ret_address, return_stack[:-i]))
    #                             break
    #         elif isinstance(node.flowchange, FuncExit):
    #             for i, ret_address in enumerate(reversed(return_stack), 1):
    #                 if ret_address not in seen_nodes:
    #                     node_queue.put((ret_address, return_stack[:-i]))
    #                     break
    #         else:
    #             for next_block in node.flowchange:
    #                 next_addr = next_block.to_addr
    #                 if next_addr not in seen_nodes:
    #                     node_queue.put((next_addr, return_stack))
    #                 else:
    #                     for i, ret_address in enumerate(reversed(return_stack), 1):
    #                         if ret_address not in seen_nodes:
    #                             node_queue.put((ret_address, return_stack[:-i]))
    #                             break
        # print('BFS finished')

    def max_address(self) -> int:
        return self.nodes[max(self.nodes.keys())].end_addr

    def min_address(self) -> int:
        return min(self.nodes.keys())


def get_instruction_type(instruction: str) -> str:
    operation = instruction.split()[0].lower()
    for instr_subtype, instructions in SUBTYPE_TO_INSTRUCTIONS.items():
        if operation in instructions:
            return instr_subtype
    print(f'Unknown instruction: {instruction=}, {operation=}')
    return 'unknown'

