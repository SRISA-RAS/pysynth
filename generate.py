from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import itertools
import operator
from pathlib import Path
import random
from typing import BinaryIO, TextIO

from graph import FlowGraph, FlowNode, FlowChangeEdge, Unconditional, Branch, Multiple, Fallthrough, \
    FuncCall, CondFuncCall, RegFuncCall, FuncExit, Exit
from graph import Address
from instruction_types import SUBTYPE_TO_INSTRUCTIONS, OperandType, INSTRUCTION_OPERAND_TYPES, \
    remove_unused_instructions


@dataclass
class GraphLoops:
    __slots__ = ('loops', 'ranks')
    loops: dict[tuple[Address, Address], FlowChangeEdge]
    ranks: dict[Address, int]


def find_loops(begin: FlowNode, graph: FlowGraph, depth_limit: int = 100) -> list[list[Address]]:
    """Returns a list of cycles in a function graph starting from a node given by 'begin'."""
    cycles = []
    queue: deque[tuple[Address, list[Address]]] = deque()   # use BFS with a path of previously traversed nodes
    queue.append((begin.address, []))
    while len(queue) > 0:
        address, path = queue.popleft()
        assert len(path) < depth_limit, f'Got over {depth_limit=} in find_loops'
        node = graph.nodes[address]
        path = [*path, address]
        for edge in node.flowchange:
            next_address = edge.to_addr
            if next_address in path:
                pos = path.index(next_address)
                cycles.append(path[pos:])
            else:
                queue.append((next_address, path))
    return cycles


def reduce_loops(graph: FlowGraph, loops: list[list[Address]], r_fac: int, min_count: int):
    """
    Reduces each flow transition count in a loop from 'loops' by a factor of 'r_fac'
    but not reducing them to be less than 'min_count'.
    Each transition is reduced only once.
    """
    # At the moment the algorithm is not particularly helpful - it does the same as if
    # we just reduced every flowchange count in a given function without any regard to
    # inner composition of loops
    # TODO: improve the algorithm, add more knobs to control how it works
    reduced: set[int] = set()
    for loop in loops:
        for address_from, address_to in itertools.pairwise(loop):
            for fc in graph.nodes[address_from].flowchange:
                if fc.to_addr == address_to:
                    if (fc_id := id(fc)) not in reduced:
                        new_count = max(fc.count // r_fac, min(min_count, fc.count))
                        fc.count = new_count
                        reduced.add(fc_id)
                    break


def _get_next_branch(left: FlowChangeEdge, right: FlowChangeEdge) -> tuple[FlowChangeEdge, bool]:
    if left is None:
        return right, True
    elif right is None:
        return left, False
    taken, *_ = random.choices([False, True], weights=[left.count, right.count])
    return (right if taken else left), taken


def _get_next_jump(node: Multiple | RegFuncCall) -> FlowChangeEdge:
    edges = node.target_edges if isinstance(node, RegFuncCall) else node.edges
    counts = [e.count for e in edges]
    next_jump, *_ = random.choices(edges, weights=counts)
    return next_jump


# _fill_flowchange_data -> ...[..., list[bytes]]...
def fill_branch_data_(graph: FlowGraph) -> tuple[list[bool], list[FlowChangeEdge]]:
    """
    Generates a list of taken/not taken branches in a graph.
    """
    branches_result: list[bool] = []
    jumps_result: list[FlowChangeEdge] = []
    res_cnt = 0
    current_node = graph.start_node
    return_stack: list[Address] = []


    ## TO DO: if throws replace the throwing FuncExit instance with Multiple one

    while not isinstance(current_node.flowchange, Exit):
        ###DEBUG
        # print(f'Iterating over {current_node.address:#x}: {current_node.flowchange}')
        ###
        node_flowchange = current_node.flowchange
        if isinstance(node_flowchange, Unconditional):
            current_node = graph.nodes[node_flowchange.edge.to_addr]
        elif isinstance(node_flowchange, Fallthrough):
            current_node = graph.nodes[node_flowchange.to.to_addr]
        elif isinstance(node_flowchange, FuncExit):
            current_node = graph.nodes[return_stack.pop()]
        elif isinstance(node_flowchange, FuncCall):
            current_node = graph.nodes[node_flowchange.target_edge.to_addr]
            return_stack.append(node_flowchange.return_edge.to_addr)
        elif isinstance(node_flowchange, CondFuncCall):
            assert node_flowchange.target_edge is not None
            assert node_flowchange.return_edge is not None
            next_edge, branch_taken = _get_next_branch(node_flowchange.return_edge, node_flowchange.target_edge)
            return_stack.append(node_flowchange.return_edge.to_addr)   # be wary of unexpected behaviour there
            branches_result.append(branch_taken)
            current_node = graph.nodes[next_edge.to_addr]
        elif isinstance(node_flowchange, RegFuncCall):
            next_edge = _get_next_jump(node_flowchange)
            return_stack.append(node_flowchange.return_edge.to_addr)
            jumps_result.append(next_edge)
            current_node = graph.nodes[next_edge.to_addr]
        elif isinstance(node_flowchange, Multiple):
            next_edge = _get_next_jump(node_flowchange)
            jumps_result.append(next_edge)
            current_node = graph.nodes[next_edge.to_addr]
        else:
            assert isinstance(node_flowchange, Branch)
            assert node_flowchange.nottaken is not None
            assert node_flowchange.taken is not None
            next_edge, branch_taken = _get_next_branch(node_flowchange.nottaken, node_flowchange.taken)
            branches_result.append(branch_taken)
            current_node = graph.nodes[next_edge.to_addr]

    return branches_result, jumps_result


# def fill_jump_data(graph: FlowGraph) -> list[bytes]:


def _write_branch_data(branches: list[bool], out: BinaryIO) -> None:
    """
    Writes list of branches taken/not taken data into an open binary file.
    Every branch item takes a byte.
    """
    for b in branches:
        out.write(int(b).to_bytes(1, byteorder='little'))
    if (filled_cacheline := len(branches) % 32) != 0:
        out.write(b'\x00' * (32 - filled_cacheline))


def _write_jump_data(jumps: list[FlowChangeEdge], offsets: dict[Address, int], out: BinaryIO) -> None:
    base_addr = 0xffffffff80000000
    for jump in jumps:
        out.write((base_addr + offsets[jump.to_addr]).to_bytes(8, byteorder='big'))


def generate_operands_(instruction: str) -> str:
    """
    Generates operands for a given instruction in accordance with operands types allowed for it.
    """
    if instruction.startswith('c.'):
        instruction = 'c.cond.fmt'
    operands_type = INSTRUCTION_OPERAND_TYPES[instruction]
    if operands_type == OperandType.REG:
        return f't{random.randint(0, 7)}'
    elif operands_type == OperandType.REG_REG:
        if instruction in {'div', 'divu', 'ddiv', 'ddivu'}:
            return f'r0, t{random.randint(0, 7)}, t{random.randint(0, 7)}'
        return f't{random.randint(0, 7)}, t{random.randint(0, 7)}'
    elif operands_type == OperandType.REG_REG_REG:
        return f't{random.randint(0, 7)}, t{random.randint(0, 7)}, t{random.randint(0, 7)}'
    elif operands_type == OperandType.REG_REG_REG_REG:
        return f't{random.randint(0, 7)}, t{random.randint(0, 7)}, t{random.randint(0, 7)}, t{random.randint(0, 7)}'
    elif operands_type == OperandType.REG_REG_IMM:
        return (f't{random.randint(0, 7)}, t{random.randint(0, 7)}, '
                f'{int.from_bytes(random.randbytes(2), byteorder="little", signed=True)}')
    elif operands_type == OperandType.REG_REG_UIMM:
        return (f't{random.randint(0, 7)}, t{random.randint(0, 7)}, '
                f'{int.from_bytes(random.randbytes(2), byteorder="little", signed=False):#x}')
    elif operands_type == OperandType.REG_IMM:
        return f't{random.randint(0, 7)}, 0x{random.randbytes(2).hex()}'
    elif operands_type == OperandType.NO_OPERANDS:
        return ''
    elif operands_type == OperandType.LS_OPERANDS:
        if instruction in {'lb', 'lbu', 'sb', 'ldl', 'ldr', 'sdl', 'sdr', 'lwl', 'lwr', 'swl', 'swr'}:
            return f't{random.randint(0, 7)}, {random.randint(-32768, 32767)}(a{random.randint(0, 3)})'
        elif instruction in {'lh', 'lhu', 'sh'}:
            return f't{random.randint(0, 7)}, {random.randint(-16384, 16383) << 1}(a{random.randint(0, 3)})'
        elif instruction in {'lw', 'lwu', 'sw'}:
            return f't{random.randint(0, 7)}, {random.randint(-8192, 8191) << 2}(a{random.randint(0, 3)})'
        elif instruction in {'ld', 'sd'}:
            return f't{random.randint(0, 7)}, {random.randint(-4096, 4095) << 3}(a{random.randint(0, 3)})'
        raise Exception('unknown load/store instruction')
    elif operands_type == OperandType.PREF_OPERANDS:
        return f'{random.choice(["0x0", "0x1e"])}, {random.randint(-32768, 32767)}(a{random.randint(0, 3)})'
    elif operands_type == OperandType.PREFX_OPERANDS:
        raise Exception('prefx is supposed to be unused')
    elif operands_type == OperandType.SYNCI_OPERANDS:
        raise Exception('synci is supposed to be unused')
    elif operands_type == OperandType.EXTRACT_INSERT_OPERANDS:
        if instruction == 'dext':
            pos_size = random.randint(1, 63)
            if pos_size <= 32:
                size = random.randint(1, pos_size)
            else:
                size = random.randint(pos_size - 32, 32)
            pos = pos_size - size
        elif instruction == 'dextm':
            pos_size = random.randint(33, 64)
            size = random.randint(33, pos_size)
            pos = pos_size - size
        elif instruction == 'dextu' or instruction == 'dinsu':
            pos_size = random.randint(33, 64)
            pos = random.randint(32, pos_size - 1)
            size = pos_size - pos
        elif instruction == 'ins' or instruction == 'ext' or instruction == 'dins':
            pos_size = random.randint(1, 32)
            size = random.randint(1, pos_size)
            pos = pos_size - size
        elif instruction == 'dinsm':
            pos_size = random.randint(33, 64)
            pos = random.randint(0, 31)
            size = pos_size - pos
        else:
            raise Exception('unknown extract/insert instruction')
        return f't{random.randint(0, 7)}, t{random.randint(0, 7)}, {pos}, {size}'
    elif operands_type == OperandType.SHIFT_OPERANDS:
        return f't{random.randint(0, 7)}, t{random.randint(0, 7)}, {random.randint(0, 31)}'
    elif operands_type == OperandType.FLOAT_COND_MOVE_OPERANDS:
        return f't{random.randint(0, 7)}, t{random.randint(0, 7)}, $fcc{random.randint(0, 7)}'
    elif operands_type == OperandType.F_REG_REG:
        return f'fp{random.randint(0, 31)}, fp{random.randint(0, 31)}'
    elif operands_type == OperandType.F_REG_REG_REG:
        return f'fp{random.randint(0, 31)}, fp{random.randint(0, 31)}, fp{random.randint(0, 31)}'
    elif operands_type == OperandType.F_REG_REG_REG_REG:
        return f'fp{random.randint(0, 31)}, fp{random.randint(0, 31)}, fp{random.randint(0, 31)}, fp{random.randint(0, 31)}'
    elif operands_type == OperandType.F_COMPARE:
        return f'$fcc{random.randint(0, 7)}, fp{random.randint(0, 31)}, fp{random.randint(0, 31)}'
    elif operands_type == OperandType.F_ALIGN:
        return f'fp{random.randint(0, 31)}, fp{random.randint(0, 31)}, fp{random.randint(0, 31)}, t{random.randint(0, 7)}'
    elif operands_type == OperandType.F_LOADSTORE:
        if instruction in {'lwc1', 'swc1'}:
            return f'fp{random.randint(0, 31)}, {random.randint(-8192, 8191) << 2}(a{random.randint(0, 3)})'
        elif instruction in {'ldc1', 'sdc1'}:
            return f'fp{random.randint(0, 31)}, {random.randint(-4096, 4095) << 3}(a{random.randint(0, 3)})'
        else:
            raise Exception('unknown floating point load/store instruction')
    elif operands_type == OperandType.F_LOADSTORE_INDEXED:
        return f'fp{random.randint(0, 31)}, t{random.randint(0, 7)}(a{random.randint(0, 3)})'
    elif operands_type == OperandType.F_MOVE:
        return f't{random.randint(0, 7)}, fp{random.randint(0, 31)}'
    elif operands_type == OperandType.F_CONTROL_MOVE:
        return f't{random.randint(0, 7)}, {random.choice(("C1_FIR", "C1_FCCR", "C1_FEXR", "C1_FENR", "C1_FCSR"))}'
    elif operands_type == OperandType.F_REG_REG_COND:
        return f'fp{random.randint(0, 31)}, fp{random.randint(0, 31)}, $fcc{random.randint(0, 7)}'
    elif operands_type == OperandType.F_REG_REG_CONDREG:
        return f'fp{random.randint(0, 31)}, fp{random.randint(0, 31)}, t{random.randint(0, 7)}'
    else:
        raise Exception('unexpected operand type of an instruction')


def generate_instruction_(node: FlowNode) -> str:
    """
    Randomly generates an instruction, following an instruction type distribution of the given node.
    """
    node_instr_dict = node.metrics.instr_by_type
    if sum(node_instr_dict.values()) != 0:
        generated_type, = random.choices(list(node_instr_dict.keys()), list(node_instr_dict.values()))
        # node_instr_dict[generated_type] -= 1
        generated_instruction, = random.sample(SUBTYPE_TO_INSTRUCTIONS[generated_type], 1)
    else:
        generated_instruction = 'nop'
    return f'\t\t{generated_instruction}\t{generate_operands_(generated_instruction)}\n'


def get_write_list(graph: FlowGraph) -> list[FlowNode]:
    return [node for _, node in sorted(graph.nodes.items(), key=operator.itemgetter(0))]


def _get_min_gen_count(node: FlowNode) -> int:
    """Returns a minimal amount of instructions that may be generated for a node with a given FlowChange value."""
    fc = node.flowchange
    result = 2 if node.restores_regs else 0
    if isinstance(fc, (Branch, Multiple)):
        # three instructions for calculating the target + one in delay slot
        result = 4
    elif isinstance(fc, FuncCall):
        # three instructions for saving registers and calling the function + one in delay slot
        result = 4
    elif isinstance(fc, RegFuncCall):
        # five instructions for saving registers and calling the function + one in delay slot
        result = 6
    elif isinstance(fc, CondFuncCall):
        # six instructions for saving registers and calling the function + one in delay slot
        result = 7
    elif isinstance(fc, (Unconditional, FuncExit)):
        # has at least 'j ...' / 'jr ra' + delay slot
        result = 2
    elif isinstance(fc, Exit):
        # has at least a 'wait' instruction
        result = 1
    else:
        result = 0
    return result + (2 if node.restores_regs else 0)


# move writelist from parameters into a body of the function
def _spread_basic_blocks(nodes: list[FlowNode]) -> dict[Address, int]:
    START_OFFSET = 0x2068
    MAX_DIFF = 0x0010_0000

    result = {}
    curr_offset = START_OFFSET
    exp_begin = nodes[0].address
    for node in nodes:
        act_begin = node.address
        diff = act_begin - exp_begin
        if diff >= MAX_DIFF:
            diff = 0
        result[node.address] = curr_offset + diff
        exp_begin = act_begin + len(node.instructions) * 4
        curr_offset += diff + max(len(node.instructions), _get_min_gen_count(node)) * 4
    return result


def _write_block(node: FlowNode, out: TextIO, labels: dict[Address, str], offsets: dict[Address, int]):
    block_instr_count = sum(node.metrics.instr_by_type.values())

    # remove unsupported instructions
    node.metrics.instr_by_type.pop('branch', None)
    node.metrics.instr_by_type.pop('unknown', None)
    node.metrics.instr_by_type.pop('privileged', None)
    node.metrics.instr_by_type.pop('trap', None)

    if offset := offsets.get(node.address):
        out.write(f'\t\t.org 0x{offset:x}\n')
    out.write(f'{labels[node.address]}:\n')

    DEBUG_OUTPUT: bool = True
    if DEBUG_OUTPUT:
        out.write(f'\\\\ source at {node.address:#x}\n')

    if node.restores_regs:
        out.write('\t\tld\tra, 8(sp)\n'
                  '\t\tdaddiu\tsp, sp, 8\n')
        block_instr_count = max(block_instr_count - 2, 0)
    for _ in range(block_instr_count - 7):
        out.write(generate_instruction_(node))
    if isinstance(node.flowchange, Unconditional):
        for _ in range(max(block_instr_count - 7, 0), block_instr_count - 2):
            out.write(generate_instruction_(node))
        out.write(f'\t\tj\t{labels[node.flowchange.edge.to_addr]}\n')
        out.write(generate_instruction_(node))
    elif isinstance(node.flowchange, Branch):
        assert node.flowchange.taken is not None and node.flowchange.nottaken is not None
        for _ in range(max(block_instr_count - 7, 0), block_instr_count - 4):
            out.write(generate_instruction_(node))
        out.write('\t\tlb\tt1, 0x0(s7)\n'
                  '\t\tdaddiu\ts7, s7, 1\n'
                  f'\t\tbne\tt1, r0, {labels[node.flowchange.taken.to_addr]}\n')
        out.write(generate_instruction_(node))
    elif isinstance(node.flowchange, Multiple):
        for _ in range(max(block_instr_count - 7, 0), block_instr_count - 4):
            out.write(generate_instruction_(node))
        out.write('\t\tld\tt1, 0x0(s6)\n'
                  '\t\tdaddiu\ts6, s6, 8\n'
                  '\t\tjr\tt1\n')
        out.write(generate_instruction_(node))
    elif isinstance(node.flowchange, Fallthrough):
        for _ in range(4):
            out.write(generate_instruction_(node))
    elif isinstance(node.flowchange, FuncCall):
        for _ in range(max(block_instr_count - 7, 0), block_instr_count - 4):
            out.write(generate_instruction_(node))
        out.write('\t\tdaddiu\tsp, sp, -8\n'
                  '\t\tsd\tra, 8(sp)\n'
                  f'\t\tjal\t{labels[node.flowchange.target_edge.to_addr]}\n')
        out.write(generate_instruction_(node))
    elif isinstance(node.flowchange, CondFuncCall):
        assert node.flowchange.target_edge.to_addr is not None
        out.write('\t\tdaddiu\tsp, sp, -8\n'
                  '\t\tsd\tra, 8(sp)\n'
                  '\t\tlb\tt1, 0x0(s7)\n'
                  '\t\tdaddiu\ts7, s7, 1\n'
                  '\t\tdsubu\tt1, t1, 1\n'
                  f'\t\tbgezal\tt1, {labels[node.flowchange.target_edge.to_addr]}\n')
        out.write(generate_instruction_(node))
    elif isinstance(node.flowchange, RegFuncCall):
        for _ in range(max(block_instr_count - 7, 0), block_instr_count - 6):
            out.write(generate_instruction_(node))
        out.write('\t\tdaddiu\tsp, sp, -8\n'
                  '\t\tsd\tra, 8(sp)\n'
                  '\t\tld\tt1, 0x0(s6)\n'
                  '\t\tdaddiu\ts6, s6, 8\n'
                  '\t\tjalr\tt1\n')
        out.write(generate_instruction_(node))
    elif isinstance(node.flowchange, FuncExit):
        for _ in range(max(block_instr_count - 7, 0), block_instr_count - 2):
            out.write(generate_instruction_(node))
        out.write('\t\tjr\tra\n')
        out.write(generate_instruction_(node))
    else:
        assert isinstance(node.flowchange, Exit)
        for _ in range(max(block_instr_count - 7, 0), block_instr_count - 1):
            out.write(generate_instruction_(node))
        out.write('\t\twait\t0x7ffff\n')


def generate_code_from_graph(graph: FlowGraph, out: TextIO, branch_data: BinaryIO, jump_data: BinaryIO, ram_table: TextIO,
                             reduction_type: str, reduction_factor: int, name: str = "", seed: int = 0,
                             trace_path: str = "") -> None:
    """
    TODO: write a docstring, refactor
    TODO: split up into meaningful subprocedures
    """
    remove_unused_instructions()  # should get rid of global state, encapsulating this stuff
    assert 'alnv.ps' not in SUBTYPE_TO_INSTRUCTIONS['fpu.convert']

    random.seed(seed)

    if reduction_type != 'none':
        for entry_address in graph.functions | {graph.start_node.address}:
            loops = find_loops(graph.nodes[entry_address], graph)
            reduce_loops(graph, loops, r_fac=100, min_count=1)

    branches_list, jumps_list = fill_branch_data_(graph)
    _write_branch_data(branches_list, branch_data)

    ram_table.write('00000000 test.bin\n'
                    '10000000 btable.bin\n'
                    '20000000 jtable.bin\n')

    label_gen = (f'label_{i}' for i in itertools.count(1))
    address_to_label = {addr: label for addr, label in zip(graph.nodes.keys(), label_gen)}

    # write metadata
    if name:
        out.write(f'\\\\ {name}\n')
    out.write(f'\\\\ generated from trace at {Path(trace_path).resolve()}\n')
    out.write(f'\\\\ using seed = {seed}\n\n')

    # write header (init) data
    btable_addr = 0x8000000050000000
    jtable_addr = 0x8000000060000000
    mem_area_0 = 0x8000000041000000
    mem_area_1 = 0x8000000042000000
    mem_area_2 = 0x8000000043000000
    mem_area_3 = 0x8000000044000000

    out.write('.nolist\n'
              '.set noreorder\n'
              '#include "regdef_k64.h"\n'
              '#include "kernel_k64.h"\n'
              '#include "handlers.h"\n\n'
              f'#define BRANCH_TABLE_ADDR\t{btable_addr:#x}\n'
              f'#define JUMP_TABLE_ADDR\t\t{jtable_addr:#x}\n'
              f'#define MEM_AREA_0\t\t{mem_area_0:#x}\n'
              f'#define MEM_AREA_1\t\t{mem_area_1:#x}\n'
              f'#define MEM_AREA_2\t\t{mem_area_2:#x}\n'
              f'#define MEM_AREA_3\t\t{mem_area_3:#x}\n\n'
              '.list\n'
              '.text\n'
              '.globl\t__start\n'
              '\t\t.org 0x2000\n'
              '\t\t.set mips64r2\n'
              '__start:\n'
              '\t\tdli\ts7, BRANCH_TABLE_ADDR\n'
              '\t\tdli\ts6, JUMP_TABLE_ADDR\n'
              '\t\tdli\ta0, MEM_AREA_0\n'
              '\t\tdli\ta1, MEM_AREA_1\n'
              '\t\tdli\ta2, MEM_AREA_2\n'
              '\t\tdli\ta3, MEM_AREA_3\n'
              f'\t\tj\t{address_to_label[graph.start_node.address]}\n'
              '\t\tnop\n\n'
              '.align\t0x3\n\n\n')

    write_list = get_write_list(graph)
    offsets = _spread_basic_blocks(write_list)

    _write_jump_data(jumps_list, offsets, jump_data)

    for node in get_write_list(graph):
        _write_block(node, out, address_to_label, offsets)

