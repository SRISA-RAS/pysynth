from __future__ import annotations

from collections import defaultdict
from dataclasses import astuple
from enum import Enum, unique
from typing import TextIO

from graph import Address, FlowChangeEdge, FlowGraph, FlowNode, Instruction, NodeMetrics,\
    Unconditional, Branch, Multiple, FuncCall, CondFuncCall, RegFuncCall, FuncExit, Fallthrough, Exit,\
    get_instruction_type
from instruction_types import FLOW_CHANGE_TYPES, SUBTYPE_TO_INSTRUCTIONS
from trace_parser import ParseSignal, SignalType, MetricData, MetricType, parse_trace


@unique
class LastInstrType(Enum):
    ORDINARY = 1
    FLOWCHANGE = 2
    DELAY_SLOT = 3


class GraphBuilder:
    __slots__ = ('trace_parser', 'current_node', 'graph', 'node_already_exists',
                 'callers', 'last_pc', 'last_instr', 'pending_reset')

    def __init__(self, trace: TextIO):
        self.trace_parser = parse_trace(trace)
        start_signal = next(self.trace_parser)
        assert start_signal.sig == SignalType.START
        self._reset(start_signal.data)

    def _reset(self, new_pc: int):
        self.last_pc = new_pc

        self.current_node = FlowNode.create(Address(new_pc))
        self.graph = FlowGraph(self.current_node, {self.current_node.address: self.current_node}, set())

        self.node_already_exists = False
        self.callers = []
        self.last_instr = LastInstrType.ORDINARY

        self.pending_reset = False

    def _handle_signal(self, signal: ParseSignal):
        if signal.sig == SignalType.RESTART:
            self._reset(signal.data)
        else:
            raise Exception(f'Unknown ParseSignal of type {signal.sig}')

    def _handle_metric(self, metric: MetricData):
        if metric.type == MetricType.LOADSTORE_ADDRESS:
            current_node.metrics.data_addresses.add(metric.data)
        else:
            raise Exception(f'Unknown MetricData of type {metric.type}')

    def _apply_fixes(self):
        _ensure_explicit_end(self.graph, self.last_pc, self.current_node)
        _graph_fix_missing_branches(self.graph)
        _graph_fix_missing_delay_slot(self.graph)
        _graph_fix_func_call_reg_restoration(self.graph)
        _graph_fix_func_calls(self.graph)
        _graph_calculate_instruction_mix(self.graph)

    def build(self) -> FlowGraph:
        for line in self.trace_parser:
            if isinstance(line, ParseSignal):
                self._handle_signal(line)
                continue

            elif isinstance(line, MetricData):
                self._handle_metric(line)
                continue

            num, pc, instruction = astuple(line)

            if self.pending_reset:
                self.pending_reset = False
                self._reset(pc)

            if self.last_instr == LastInstrType.FLOWCHANGE and pc != self.last_pc + 4:
                self.last_instr = LastInstrType.DELAY_SLOT   # likely branch with delay slot missing

            if self.last_instr == LastInstrType.DELAY_SLOT:
                _add_or_increment_flowchange(pc, self.current_node, self.node_already_exists, self.callers, self.graph.functions)
                self.current_node, self.node_already_exists = _get_next_node(self.graph, Address(pc), self.current_node)
            elif self.node_already_exists and pc > self.current_node.address and pc in self.graph.nodes.keys():
                assert isinstance(self.current_node.flowchange, Fallthrough)
                self.current_node.flowchange.to.count += 1
                self.current_node = self.graph.nodes[pc]

            if not self.node_already_exists:
                self.current_node, self.node_already_exists = _append_instruction_line(
                    self.graph, self.current_node, pc, self.node_already_exists, instruction, self.last_instr)

            if self.last_instr == LastInstrType.FLOWCHANGE and self.current_node.missing_delay_slot:
                self.current_node.instructions.append(Instruction(pc, instruction))
                self.current_node.missing_delay_slot = False

            if self.last_instr == LastInstrType.DELAY_SLOT:
                self.last_instr = LastInstrType.ORDINARY
            elif self.last_instr == LastInstrType.FLOWCHANGE:
                if _check_for_unmatched_return(self.graph, self.current_node, self.node_already_exists, len(self.callers)):
                    self.pending_reset = True   # restart if we get return from a function when not in function
                    continue
                self.last_instr = LastInstrType.DELAY_SLOT

            if instruction.split()[0] in SUBTYPE_TO_INSTRUCTIONS['branch']:
                self.last_instr = LastInstrType.FLOWCHANGE
                if not self.node_already_exists:
                    self.current_node.end_addr = pc + 4

            self.last_pc = pc

        self._apply_fixes()
        return self.graph


def _append_instruction_line(graph: FlowGraph, node: FlowNode, pc: Address, node_exists: bool,
                             instruction: str, last_instr: LastInstrType) -> tuple[FlowNode, bool]:
    """
    Append an instruction to an existing node or finish up a Fallthrough node if needed.
    Returns current node and whether the node already exists in graph after taken actions.
    """
    if len(node.instructions) > 0 and pc in graph.nodes.keys():
        entrance_count = sum(fc.count for fc in graph.nodes[pc].flowchange)
        node.flowchange = Fallthrough(FlowChangeEdge(node.address, pc, entrance_count))
        node.end_addr = node.instructions[-1].program_counter
        node.missing_delay_slot = False

        node = graph.nodes[pc]
        node_exists = True
    else:
        node.instructions.append(Instruction(pc, instruction))
        if last_instr == LastInstrType.FLOWCHANGE:
            node.missing_delay_slot = False
    return node, node_exists


def _check_for_unmatched_return(graph: FlowGraph, node: FlowNode, node_exists: bool, fn_level: int) -> bool:
    """Returns whether current flowchange is an unmatched return from a function"""
    if node_exists:
        node = graph.nodes[node.address]
    branch_line = node.instructions[-1 if node.missing_delay_slot else -2]
    branch_instruction = branch_line.instruction.split()[0]
    if (branch_instruction in FLOW_CHANGE_TYPES['multiple'] and
            branch_line.instruction.split()[1] == '$ra' and fn_level <= 0):
        return True
    return False


def splice_node(graph: FlowGraph, node_addr: Address, instr_pc: Address) -> tuple[FlowNode, FlowNode]:
    """
    Splices an existing node in a graph into two with one ending at
    an instruction instr_pc - 1, and the other starting from instr_pc.
    """
    node = graph.nodes[node_addr]
    assert node.end_addr is not None
    assert node.end_addr != 0x0
    assert len(node.instructions) > 1
    if not isinstance(node.flowchange, (Exit, Fallthrough)) and instr_pc == node.end_addr:
        raise Exception(f'Trying to splice a node\'s delay slot, delay slot address is {instr_pc:#x}')

    instr_index = (instr_pc - node_addr) // 4
    assert 0 < instr_index < len(node.instructions)

    fallthrough_count = sum(fc.count for fc in node.flowchange)

    new_node = FlowNode(instr_pc, node.end_addr, node.instructions[instr_index:], node.flowchange, node.metrics,
                        node.is_synthetic, node.restores_regs, node.missing_delay_slot)
    for edge in new_node.flowchange:
        edge.from_addr = new_node.address
    node.instructions[instr_index:] = []
    node.flowchange = Fallthrough(FlowChangeEdge(node.address, new_node.address, fallthrough_count))
    node.end_addr = node.instructions[-1].program_counter
    node.missing_delay_slot = False

    graph.nodes[new_node.address] = new_node
    return node, new_node


# TODO: consider what to do with function not returning where they are expected to return (some smart stack manipulation)
#       as this breaks everything later - we may end up iterating over function nodes in main graph
# TODO: graph and line params are for debugging - remove them
def _add_or_increment_flowchange(pc: Address, current_node: FlowNode, node_exists: bool, callers: list[FlowChangeEdge],
                                 functions: set[Address]) -> int:
    """
    Adds a flowchange for a node if it was not seen before or this is the first occurence of this node.
    Increments the count of this flowchange otherwise.

    Returns the nested function level.
    """
    if not node_exists:
        end_addr = current_node.end_addr
        branch_line = current_node.instructions[-1 if current_node.missing_delay_slot else -2]
        branch_instruction = branch_line.instruction.split()[0]
        edge = FlowChangeEdge(current_node.address, pc, 1)

        if branch_instruction in FLOW_CHANGE_TYPES['unconditional']:
            current_node.flowchange = Unconditional(edge)

        elif branch_instruction in FLOW_CHANGE_TYPES['branch']:
            operands = branch_line.instruction.split()[1]
            branch_address = operands[operands.rfind(',') + 1:]
            if pc == Address(branch_address[2:], base=16):
                current_node.flowchange = Branch(edge, None)
            else:
                current_node.flowchange = Branch(None, edge)

        elif branch_instruction in FLOW_CHANGE_TYPES['func_cond_call']:
            operands = branch_line.instruction.split()[1]
            branch_address = operands[operands.rfind(',') + 1:]
            if pc == Address(branch_address[2:], base=16):
                return_edge = FlowChangeEdge(current_node.address, Address(branch_line.program_counter + 8), 0)
                current_node.flowchange = CondFuncCall(edge, return_edge)
            else:
                current_node.flowchange = CondFuncCall(None, edge)
            callers.append(current_node.flowchange)
            functions.add(pc)

        elif branch_instruction in FLOW_CHANGE_TYPES['multiple']:
            if branch_line.instruction.split()[1] == '$ra' and len(callers) > 0:
                return_edge = callers.pop().return_edge
                assert return_edge.to_addr == pc, 'Unexpected address after return from a function call'
                current_node.flowchange = FuncExit()
            else:
                current_node.flowchange = Multiple([edge])

        elif branch_instruction in FLOW_CHANGE_TYPES['func_call']:
            return_edge = FlowChangeEdge(current_node.address, Address(branch_line.program_counter + 8), 0)
            current_node.flowchange = FuncCall(edge, return_edge)
            callers.append(current_node.flowchange)
            functions.add(pc)

        elif branch_instruction in FLOW_CHANGE_TYPES['func_call_reg']:
            return_edge = FlowChangeEdge(current_node.address, Address(branch_line.program_counter + 8), 0)
            current_node.flowchange = RegFuncCall([edge], return_edge)
            callers.append(current_node.flowchange)
            functions.add(pc)

        else:
            raise Exception(f'Unknown branch instruction at PC=0x{branch_line.program_counter:x}')

    else:
        if isinstance(current_node.flowchange, FuncExit):
            return_edge = callers.pop().return_edge
            assert return_edge.to_addr == pc, 'Unexpected address after return from a function call'
            return_edge.count += 1
            return

        if isinstance(current_node.flowchange, (FuncCall, CondFuncCall, RegFuncCall)):
            callers.append(current_node.flowchange)

        for edge in current_node.flowchange:
            if edge.to_addr == pc:
                edge.count += 1
                break
        else:
            edge = FlowChangeEdge(current_node.address, pc, 1)
            current_node.flowchange.add(edge)


def _get_next_node(graph: FlowGraph, pc: Address, current_node: FlowNode) -> tuple[FlowNode, bool]:
    """
    Returns an object of the currently being parsed node and whether it already exists.
    Splices an already existing node if needed.
    """
    if pc in graph.nodes.keys():
        return graph.nodes[pc], True
    elif current_node.address < pc <= current_node.end_addr:
        _, next_node = splice_node(graph, current_node.address, pc)
        return next_node, True
    else:
        for node in graph.nodes.values():
            if node.address < pc <= node.end_addr:
                _, next_node = splice_node(graph, node.address, pc)
                return next_node, True
        new_node = FlowNode.create(pc)
        graph.nodes[pc] = new_node
        return new_node, False


def _ensure_explicit_end(graph: FlowGraph, pc: Address, last_node: FlowNode):
    """
    Adds a branch to a wait instruction at the line the trace ended if it lacks an explicit one.
    """
    last_instruction = last_node.instructions[-1].instruction
    if last_instruction.startswith(('wait', 'break')):
        last_node.missing_delay_slot = False
        last_node.end_addr = last_node.instructions[-1].program_counter
    else:
        new_address = graph.max_address() + 4
        wait_instr = Instruction(new_address, 'wait 0x7ffff')
        endnode = FlowNode(new_address, new_address, [wait_instr], Exit(),
                           NodeMetrics({'privileged': 1}), True, False, False)
        assert endnode.address not in graph.nodes.keys()
        graph.nodes[endnode.address] = endnode
        split_index = -1 if last_node.missing_delay_slot or isinstance(last_node.flowchange, Fallthrough) else -2
        branch_or_end_instr_pc = last_node.instructions[split_index].program_counter
        splice_instr_pc = min(pc + 4, branch_or_end_instr_pc)
        first, _ = splice_node(graph, last_node.address, splice_instr_pc)
        endnode_edge = FlowChangeEdge(first.address, endnode.address, 1)
        first.flowchange = Branch(endnode_edge, first.flowchange.to)


def calculate_instruction_mix(node: FlowNode) -> dict[str, int]:
    """Calculates instruction mix of a given node."""
    result = {}
    for instr in node.instructions:
        instr_type = get_instruction_type(instr.instruction)
        result[instr_type] = result.get(instr_type, 0) + 1
    assert len(node.instructions) == sum(result.values())
    return result


def _graph_calculate_instruction_mix(graph: FlowGraph):
    for node in graph.nodes.values():
        node.metrics.instr_by_type = calculate_instruction_mix(node)


# def loadstore_extract_address(mnemonic: str) -> Address:
#     operation, *operands = mnemonic.split()
#     # will not work
#     if operation in {'lb', 'lbu', 'lh', 'lhu', 'lw', 'lwu', 'lwl', 'lwr', 'ld', 'ldl', 'ldr', 'ldc1'}:
#         pos_comma = operands[-1].find(',')
#         pos_open_paren = operands[-1].find('(')
#         base = Address(operands[-1][pos_comma:pos_open_paren])
#         offset = Address(operands[-1][pos_open_paren:-1])
#     ...


# def _graph_find_data_addresses(graph: FlowGraph):
#     for node in graph.nodes.items():
#         data_addresses = node.metrics.data_addresses
#         for instruction in node.instructions:
#             mnemonic = instruction.instruction
#             if get_instruction_type(mnemonic) == 'loadstore':
#                 data_addresses.add(loadstore_extract_address(mnemonic))


# potentially direct them to the bad_end node
def _graph_fix_missing_branches(graph: FlowGraph):
    """Replaces empty flow change edges of branch nodes with flow changes of another type."""
    for node in graph.nodes.values():
        if isinstance(node.flowchange, Branch):
            if node.flowchange.taken is None:
                node.flowchange = Unconditional(node.flowchange.nottaken)
            elif node.flowchange.nottaken is None:
                node.flowchange = Unconditional(node.flowchange.taken)
        elif isinstance(node.flowchange, CondFuncCall) and node.flowchange.target_edge is None:
            node.flowchange.target_edge = FlowChangeEdge(node.address, node.address, 0)


def _graph_fix_missing_delay_slot(graph: FlowGraph):
    """
    Nodes with 'likely' branches may be missing a delay slot instruction as it was never actually seen in the trace.
    Adds a nop instruction in its place in such cases.
    """
    for node in graph.nodes.values():
        if node.missing_delay_slot:
            branch_instr = node.instructions[-1]
            slot_pc = branch_instr.program_counter + 4
            node.instructions.append(Instruction(slot_pc, 'nop'))
            node.missing_delay_slot = False


def _graph_fix_func_call_reg_restoration(graph: FlowGraph):
    """
    Nodes that are being returned into must restore regs when generating code.
    """
    for node in graph.nodes.values():
        if isinstance(node.flowchange, (FuncCall, RegFuncCall, CondFuncCall)):
            if (target_addr := node.end_addr + 4) in graph.nodes:
                graph.nodes[target_addr].restores_regs = True
            else:
                print('Warning: node to make restoring registers does not exist '
                      f'(node calling function is {node.address:#x})')


### Possibly redo, possibly delete
### properly handle situations where we branch and link to the next instruction
### why do i use dfs there?
def _graph_fix_func_calls(graph: FlowGraph):
    """
    Nodes that are being returned into must restore regs when generating code.

    Also if there are basic blocks that appear after a function call (meaning
    they must restore registers) that gets jumped/branch into from somewhere
    else, split such node into two.
    """
    return_addresses = set()
    for node in graph.nodes.values():
        if (isinstance(node.flowchange, (FuncCall, RegFuncCall, CondFuncCall))
                and ((target_addr := node.end_addr + 4) in graph.nodes)):
            return_addresses.add(target_addr)

    to_fix = defaultdict(list)
    for node in graph.dfs():
        if not isinstance(node.flowchange, FuncExit):
            for fc in node.flowchange:
                if fc.to_addr in return_addresses:
                    to_fix[fc.to_addr].append(node.address)

    res = [f'{victim:#x}: {[f"{x:#x}" for x in reasons]}' for victim, reasons in to_fix.items()]
    if res:
        print(f'Needs fixing: {{{res}}}')

# _graph_check_no_function_block_outside_entrance()

