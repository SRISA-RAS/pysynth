from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Iterator, TextIO, Union

from graph import Address, get_instruction_type
from instruction_types import FLOW_CHANGE_TYPES


@unique
class SignalType(Enum):
    START = 1
    RESTART = 2


@dataclass
class ParseSignal:
    __slots__ = ('sig', 'data')
    sig: SignalType
    data: Any


@dataclass
class LineData:
    __slots__ = ('num', 'va', 'instruction')
    num: int
    va: Address
    instruction: str


@unique
class MetricType(Enum):
    LOADSTORE_ADDRESS = 1


@dataclass
class MetricData:
    __slots__ = ('metric', 'data')
    metric: MetricType
    data: Any


START_LINE = '************* START *************\n'
END_LINE = '************* HALT *************\n'


def _skip_to_trace_start(file: TextIO) -> Address:
    """Skips header and returns PC address of the first instruction."""
    while file.readline() != START_LINE:
        pass
    file.readline()  # skip empty line
    return _get_next_line_program_counter(file)


def _trace_skip_unused(file: TextIO) -> Iterator[str]:
    while (line := file.readline()) != END_LINE and line != '':
        if line == '\n':
            continue
        first_sep = line.find(' ')
        if line.startswith(('icache', 'dcache', 'scache', 'dmemacc', 'Reg', 'tlb'), first_sep + 1):
            continue
        if not line[:first_sep].isnumeric():
            continue
        yield line


def _get_next_line_program_counter(file: TextIO) -> Address:
    prev_seek_pos = file.tell()
    ln = file.readline()
    while not ln.split()[1].startswith('PC='):
        ln = file.readline()
    pc = int(ln.split()[1][5:], base=16)
    file.seek(prev_seek_pos)
    return pc


def _jumped_into_branch_target(pc: Address, *instructions: str) -> bool:
    for instr in instructions:
        if get_instruction_type(instr) in FLOW_CHANGE_TYPES:
            return True
    return False


# replace TextIO with pathlib.Path parameter, use 'with open(trace_file) as f:' and change trace_file to f
def parse_trace(trace_file: TextIO) -> Iterator[Union[LineData, ParseSignal]]:
    last_pc = _skip_to_trace_start(trace_file)
    yield ParseSignal(SignalType.START, last_pc)

    exception_lvl: int = 0
    exception_return_pending: bool = False
    restart_pending: bool = False
    last_instructions: deque[str] = deque(maxlen=2)

    for ln in _trace_skip_unused(trace_file):
        if (exception_lvl == 0) and (ln.startswith('[')):
            pos_sep = ln.find(' ')
            yield MetricData(MetricType.LOADSTORE_ADDRESS, Address(ln[4:pos_sep]))
            continue

        if ln.startswith('Exception', ln.find(' ') + 1):    # TODO: flow changes
            exception_lvl += 1
            continue

        split_line = ln.split()

        if split_line[4] == 'eret':
            if exception_lvl > 0:
                exception_lvl -= 1
                if exception_lvl == 0:
                    exception_return_pending = True
            else:
                restart_pending = True
                exception_return_pending = False
            continue

        elif exception_lvl > 0:
            continue

        elif restart_pending:
            restart_pending = False
            exception_return_pending = False
            last_instructions.clear()
            pc = int(split_line[1][5:], base=16)
            yield ParseSignal(SignalType.RESTART, pc)

        elif exception_return_pending:
            ret_pc = int(split_line[1][5:], base=16)
            if ret_pc == last_pc - 4:
                continue

            exception_return_pending = False
            if ret_pc == last_pc:
                continue
            if not ret_pc == last_pc + 4 and not _jumped_into_branch_target(ret_pc, *last_instructions):
                raise Exception(f'Changing return address in exception is unsupported for now '
                                f'(line {split_line[0]})')

        line_number = int(split_line[0])
        pc = int(split_line[1][5:], base=16)
        instruction = ' '.join(split_line[4:])
        yield LineData(line_number, pc, instruction)
        last_instructions.append(instruction)
        last_pc = pc

