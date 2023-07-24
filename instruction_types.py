from enum import Enum, unique


INSTRUCTION_TYPES = {
    'branch': ('b', 'bal', 'beq', 'bgez', 'bgezal', 'bgtz', 'blez', 'bltz', 'bltzal', 'bne',
               'j', 'jr', 'jr.hb', 'jal', 'jalr', 'jalr.hb',
               'beql', 'bgezall', 'bgezl', 'bgtzl', 'blezl', 'bltzall', 'bltzl', 'bnel',

               'bc1f', 'bc1fl', 'bc1t', 'bc1tl'),

    'arith': ('add', 'addi', 'addiu', 'addu', 'clo', 'clz', 'dadd', 'daddi',
              'daddiu', 'daddu', 'dclo', 'dclz', 'ddiv', 'ddivu', 'div', 'divu', 'dmult',
              'dmultu', 'dsub', 'dsubu', 'madd', 'maddu', 'msub', 'msubu', 'mul', 'mult',
              'multu', 'seb', 'seh', 'slt', 'slti', 'sltiu', 'sltu', 'sub', 'subu'),

    'loadstore': {'load': ('lb', 'lbu', 'ld', 'ldl', 'ldr', 'lh', 'lhu', 'll', 'lld', 'lw', 'lwl', 'lwr', 'lwu',
                           'ldc1', 'ldxc1', 'luxc1', 'lwc1', 'lwxc1', 'pref'),

                  'store': ('sb', 'sc', 'scd', 'sd', 'sdl', 'sdr', 'sh', 'sw', 'swl', 'swr',
                            'sdc1', 'sdxc1', 'suxc1', 'swc1', 'swxc1'),

                  'ls.other': ('sync', 'synci', 'prefx')},

    'logic': ('and', 'andi', 'lui', 'nor', 'or', 'ori', 'xor', 'xori',

              'drotr', 'drotr32', 'drotrv', 'dsll', 'dsll32', 'dsllv', 'dsra', 'dsra32', 'dsrav', 'dsrl', 'dsrl32',
              'dsrlv', 'rotr', 'rotrv', 'sll', 'sllv', 'sra', 'srav', 'srl', 'srlv',

              'dext', 'dextm', 'dextu', 'dins', 'dinsm', 'dinsu', 'dsbh', 'dshd', 'ext', 'ins', 'wsbh'),

    'move': ('mfhi', 'mflo', 'movf', 'movn', 'movt', 'movz', 'mthi', 'mtlo', 'rdhwr'),

    'fpu': {'fpu.arith': ('abs.s', 'abs.d', 'abs.ps', 'add.s', 'add.d', 'add.ps', 'div.s', 'div.d', 'madd.s', 'madd.d',
                          'madd.ps', 'msub.s', 'msub.d', 'msub.ps', 'mul.s', 'mul.d', 'mul.ps', 'neg.s', 'neg.d', 'neg.ps',
                          'nmadd.s', 'nmadd.d', 'nmadd.ps', 'nmsub.s', 'nmsub.d', 'nmsub.ps', 'recip.s', 'recip.d',
                          'rsqrt.s', 'rsqrt.d', 'sqrt.s', 'sqrt.d', 'sub.s', 'sub.d', 'sub.ps'),

            'fpu.compare': ('c.f.s', 'c.un.s', 'c.eq.s', 'c.ueq.s', 'c.olt.s', 'c.ult.s', 'c.ole.s', 'c.ule.s', 'c.sf.s',
                            'c.ngle.s', 'c.seq.s', 'c.ngl.s', 'c.lt.s', 'c.nge.s', 'c.le.s', 'c.ngt.s',
                            'c.f.d', 'c.un.d', 'c.eq.d', 'c.ueq.d', 'c.olt.d', 'c.ult.d', 'c.ole.d', 'c.ule.d', 'c.sf.d',
                            'c.ngle.d', 'c.seq.d', 'c.ngl.d', 'c.lt.d', 'c.nge.d', 'c.le.d', 'c.ngt.d',
                            'c.f.ps', 'c.un.ps', 'c.eq.ps', 'c.ueq.ps', 'c.olt.ps', 'c.ult.ps', 'c.ole.ps', 'c.ule.ps', 'c.sf.ps',
                            'c.ngle.ps', 'c.seq.ps', 'c.ngl.ps', 'c.lt.ps', 'c.nge.ps', 'c.le.ps', 'c.ngt.ps'),

            'fpu.convert': ('alnv.ps', 'ceil.l.s', 'ceil.l.d', 'ceil.w.s', 'ceil.w.d', 'cvt.d.s', 'cvt.d.w', 'cvt.d.l', 'cvt.l.s',
                            'cvt.l.d', 'cvt.ps.s', 'cvt.s.pl', 'cvt.s.pu', 'cvt.s.d', 'cvt.s.w', 'cvt.s.l', 'cvt.w.s', 'cvt.w.d',
                            'floor.l.s', 'floor.l.d', 'floor.w.s', 'floor.w.d', 'pll.ps', 'plu.ps', 'pul.ps', 'puu.ps',
                            'round.l.s', 'round.l.d', 'round.w.s', 'round.w.d', 'trunc.l.s', 'trunc.l.d', 'trunc.w.s',
                            'trunc.w.d'),

            'fpu.move': ('cfc1', 'ctc1', 'dmfc1', 'dmtc1', 'mfc1', 'mfhc1', 'mov.s', 'mov.d', 'mov.ps', 'movf.s', 'movf.d',
                         'movf.ps', 'movn.s', 'movn.d', 'movn.ps', 'movt.s', 'movt.d', 'movt.ps', 'movz.s', 'movz.d', 'movz.ps',
                         'mtc1', 'mthc1')},

    'nop': ('nop', 'ssnop', 'ehb'),

    'privileged': ('cache', 'di', 'cfc0', 'ctc0', 'dmfc0', 'dmtc0', 'ei', 'eret', 'mfc0', 'mtc0',
                   'rdpgpr', 'tlbp', 'tlbr', 'tlbwi', 'tlbwr', 'wait', 'wrpgpr'),

    'trap': ('break', 'syscall', 'teq', 'teqi', 'tge', 'tgei', 'tgeiu', 'tgeu',
             'tlt', 'tlti', 'tltiu', 'tltu', 'tne', 'tnei')

    # 'coproc2'
}

FLOW_CHANGE_TYPES = {
    'unconditional': ('b', 'j'),
    'func_call': ('bal', 'jal'),
    'func_cond_call': ('bgezal', 'bgezall', 'bltzal', 'bltzall'),
    'func_call_reg': ('jalr', 'jalr.hb'),
    'branch': ('beq', 'beql', 'bgez', 'bgezl', 'bgtz', 'bgtzl', 'blez', 'blezl',
               'bltz', 'bltzl', 'bne', 'bnel', 'bc1t', 'bc1tl', 'bc1f', 'bc1fl'),
    'multiple': ('jr', 'jr.hb')
}

# remove this constant (unused)
LIKELY_BRANCHES = ('beql', 'bgezall', 'bgezl', 'bgtzl', 'blezl', 'bltzall', 'bltzl', 'bnel')


SUBTYPE_TO_INSTRUCTIONS = {}
for instr_type, value in INSTRUCTION_TYPES.items():
    if isinstance(value, dict):
        for subtype, instrs in value.items():
            SUBTYPE_TO_INSTRUCTIONS[subtype] = instrs
    else:
        SUBTYPE_TO_INSTRUCTIONS[instr_type] = value


def remove_unused_instructions() -> None:
    to_remove = {'ls.other': ('prefx', 'synci'),
                 'load': ('ll', 'lld', 'ldxc1', 'lwxc1', 'luxc1'),
                 'store': ('sc', 'scd', 'sdxc1', 'swxc1', 'suxc1'),
                 'move': ('rdhwr',),
                 'fpu.convert': ('alnv.ps',)}
    for subtype, remove_instructions in to_remove.items():
        updated_instructions = [*SUBTYPE_TO_INSTRUCTIONS[subtype]]
        for instr in remove_instructions:
            updated_instructions.remove(instr)
        SUBTYPE_TO_INSTRUCTIONS[subtype] = (*updated_instructions,)


@unique
class OperandType(Enum):
    REG = ('register',)
    REG_REG = ('register', 'register')
    REG_REG_REG = ('register', 'register', 'register')
    REG_REG_REG_REG = ('register', 'register', 'register', 'register')
    REG_REG_IMM = ('register', 'register', 'immediate')
    REG_REG_UIMM = ('register', 'register', 'unsigned_immediate')
    REG_IMM = ('register', 'immediate')
    NO_OPERANDS = ()
    LS_OPERANDS = ('register', 'offset', 'base')
    PREF_OPERANDS = ('hint', 'offset', 'base')
    PREFX_OPERANDS = ('hint', 'register', 'register')
    SYNCI_OPERANDS = ('offset', 'base')
    EXTRACT_INSERT_OPERANDS = ('register', 'register', 'position', 'size')
    SHIFT_OPERANDS = ('register', 'register', 'shift_amount')
    FLOAT_COND_MOVE_OPERANDS = ('register', 'register', 'condition_code')
    F_REG_REG = ('fpu_register', 'fpu_register')
    F_REG_REG_REG = ('fpu_register', 'fpu_register', 'fpu_register')
    F_REG_REG_REG_REG = ('fpu_register', 'fpu_register', 'fpu_register', 'fpu_register')
    F_COMPARE = ('condition_code', 'fpu_register', 'fpu_register')
    F_ALIGN = ('fpu_register', 'fpu_register', 'fpu_register', 'register')
    F_LOADSTORE = ('fpu_register', 'offset', 'base')
    F_LOADSTORE_INDEXED = ('fpu_register', 'index_register', 'base_register')
    F_MOVE = ('register', 'fpu_register')
    F_CONTROL_MOVE = ('register', 'fpu_control_register')
    F_REG_REG_COND = ('fpu_register', 'fpu_register', 'condition_code')
    F_REG_REG_CONDREG = ('fpu_register', 'fpu_register', 'register')


def fmt_(instruction: str, options: tuple[str, ...], operands: OperandType) -> dict[str, OperandType]:
    return {f'{instruction[:instruction.rfind(".")]}.{fmt}': operands for fmt in options}


# def cond_fmt_(instruction: str, conditions: tuple(str, ...), options: tuple(str, ...), operands: OperandType) -> dict[str, OperandType]:
#     return {f'{instruction}.{cond}.{fmt}': operands for cond in conditions for fmt in options}


INSTRUCTION_OPERAND_TYPES = {
    'add': OperandType.REG_REG_REG,
    'addi': OperandType.REG_REG_IMM,
    'addiu': OperandType.REG_REG_IMM,
    'addu': OperandType.REG_REG_REG,
    'clo': OperandType.REG_REG,
    'clz': OperandType.REG_REG,
    'dadd': OperandType.REG_REG_REG,
    'daddi': OperandType.REG_REG_IMM,
    'daddiu': OperandType.REG_REG_IMM,
    'daddu': OperandType.REG_REG_REG,
    'dclo': OperandType.REG_REG,
    'dclz': OperandType.REG_REG,
    'ddiv': OperandType.REG_REG,
    'ddivu': OperandType.REG_REG,
    'div': OperandType.REG_REG,
    'divu': OperandType.REG_REG,
    'dmult': OperandType.REG_REG,
    'dmultu': OperandType.REG_REG,
    'dsub': OperandType.REG_REG_REG,
    'dsubu': OperandType.REG_REG_REG,
    'madd': OperandType.REG_REG,
    'maddu': OperandType.REG_REG,
    'msub': OperandType.REG_REG,
    'msubu': OperandType.REG_REG,
    'mul': OperandType.REG_REG_REG,
    'mult': OperandType.REG_REG,
    'multu': OperandType.REG_REG,
    'seb': OperandType.REG_REG,
    'seh': OperandType.REG_REG,
    'slt': OperandType.REG_REG_REG,
    'slti': OperandType.REG_REG_IMM,
    'sltiu': OperandType.REG_REG_IMM,
    'sltu': OperandType.REG_REG_REG,
    'sub': OperandType.REG_REG_REG,
    'subu': OperandType.REG_REG_REG,

    'ehb': OperandType.NO_OPERANDS,
    'nop': OperandType.NO_OPERANDS,
    'ssnop': OperandType.NO_OPERANDS,

    'lb': OperandType.LS_OPERANDS,
    'lbu': OperandType.LS_OPERANDS,
    'ld': OperandType.LS_OPERANDS,
    'ldl': OperandType.LS_OPERANDS,
    'ldr': OperandType.LS_OPERANDS,
    'lh': OperandType.LS_OPERANDS,
    'lhu': OperandType.LS_OPERANDS,
    'll': OperandType.LS_OPERANDS,
    'lld': OperandType.LS_OPERANDS,
    'lw': OperandType.LS_OPERANDS,
    'lwl': OperandType.LS_OPERANDS,
    'lwr': OperandType.LS_OPERANDS,
    'lwu': OperandType.LS_OPERANDS,
    'pref': OperandType.PREF_OPERANDS,
    'sb': OperandType.LS_OPERANDS,
    'sc': OperandType.LS_OPERANDS,
    'scd': OperandType.LS_OPERANDS,
    'sd': OperandType.LS_OPERANDS,
    'sdl': OperandType.LS_OPERANDS,
    'sdr': OperandType.LS_OPERANDS,
    'sh': OperandType.LS_OPERANDS,
    'sw': OperandType.LS_OPERANDS,
    'swl': OperandType.LS_OPERANDS,
    'swr': OperandType.LS_OPERANDS,
    'sync': OperandType.NO_OPERANDS,
    'synci': OperandType.SYNCI_OPERANDS,

    'and': OperandType.REG_REG_REG,
    'andi': OperandType.REG_REG_UIMM,
    'lui': OperandType.REG_IMM,
    'nor': OperandType.REG_REG_REG,
    'or': OperandType.REG_REG_REG,
    'ori': OperandType.REG_REG_UIMM,
    'xor': OperandType.REG_REG_REG,
    'xori': OperandType.REG_REG_UIMM,

    'dext': OperandType.EXTRACT_INSERT_OPERANDS,
    'dextm': OperandType.EXTRACT_INSERT_OPERANDS,
    'dextu': OperandType.EXTRACT_INSERT_OPERANDS,
    'dins': OperandType.EXTRACT_INSERT_OPERANDS,
    'dinsm': OperandType.EXTRACT_INSERT_OPERANDS,
    'dinsu': OperandType.EXTRACT_INSERT_OPERANDS,
    'dsbh': OperandType.REG_REG,
    'dshd': OperandType.REG_REG,
    'ext': OperandType.EXTRACT_INSERT_OPERANDS,
    'ins': OperandType.EXTRACT_INSERT_OPERANDS,
    'wsbh': OperandType.REG_REG,

    'mfhi': OperandType.REG,
    'mflo': OperandType.REG,
    'movf': OperandType.FLOAT_COND_MOVE_OPERANDS,
    'movn': OperandType.REG_REG_REG,
    'movt': OperandType.FLOAT_COND_MOVE_OPERANDS,
    'movz': OperandType.REG_REG_REG,
    'mthi': OperandType.REG,
    'mtlo': OperandType.REG,

    'drotr': OperandType.SHIFT_OPERANDS,
    'drotr32': OperandType.SHIFT_OPERANDS,
    'drotrv': OperandType.REG_REG_REG,
    'dsll': OperandType.SHIFT_OPERANDS,
    'dsll32': OperandType.SHIFT_OPERANDS,
    'dsllv': OperandType.REG_REG_REG,
    'dsra': OperandType.SHIFT_OPERANDS,
    'dsra32': OperandType.SHIFT_OPERANDS,
    'dsrav': OperandType.REG_REG_REG,
    'dsrl': OperandType.SHIFT_OPERANDS,
    'dsrl32': OperandType.SHIFT_OPERANDS,
    'dsrlv': OperandType.REG_REG_REG,
    'rotr': OperandType.SHIFT_OPERANDS,
    'rotrv': OperandType.REG_REG_REG,
    'sll': OperandType.SHIFT_OPERANDS,
    'sllv': OperandType.REG_REG_REG,
    'sra': OperandType.SHIFT_OPERANDS,
    'srav': OperandType.REG_REG_REG,
    'srl': OperandType.SHIFT_OPERANDS,
    'srlv': OperandType.REG_REG_REG,

    **fmt_('abs.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG),
    **fmt_('add.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG_REG),
    **fmt_('div.fmt', ('s', 'd'), OperandType.F_REG_REG_REG),
    **fmt_('madd.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG_REG_REG),
    **fmt_('msub.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG_REG_REG),
    **fmt_('mul.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG_REG),
    **fmt_('neg.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG),
    **fmt_('nmadd.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG_REG_REG),
    **fmt_('nmsub.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG_REG_REG),
    **fmt_('recip.fmt', ('s', 'd'), OperandType.F_REG_REG),
    **fmt_('rsqrt.fmt', ('s', 'd'), OperandType.F_REG_REG),
    **fmt_('sqrt.fmt', ('s', 'd'), OperandType.F_REG_REG),
    **fmt_('sub.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG_REG),

    'c.cond.fmt': OperandType.F_COMPARE,

    'alnv.ps': OperandType.F_ALIGN,
    **fmt_('ceil.l.fmt', ('s', 'd'), OperandType.F_REG_REG),
    **fmt_('ceil.w.fmt', ('s', 'd'), OperandType.F_REG_REG),
    **fmt_('cvt.d.fmt', ('s', 'w', 'l'), OperandType.F_REG_REG),
    **fmt_('cvt.l.fmt', ('s', 'd'), OperandType.F_REG_REG),
    'cvt.ps.s': OperandType.F_REG_REG_REG,
    'cvt.s.pl': OperandType.F_REG_REG,
    'cvt.s.pu': OperandType.F_REG_REG,
    **fmt_('cvt.s.fmt', ('d', 'w', 'l'), OperandType.F_REG_REG),
    **fmt_('cvt.w.fmt', ('s', 'd'), OperandType.F_REG_REG),
    **fmt_('floor.l.fmt', ('s', 'd'), OperandType.F_REG_REG),
    **fmt_('floor.w.fmt', ('s', 'd'), OperandType.F_REG_REG),
    'pll.ps': OperandType.F_REG_REG_REG,
    'plu.ps': OperandType.F_REG_REG_REG,
    'pul.ps': OperandType.F_REG_REG_REG,
    'puu.ps': OperandType.F_REG_REG_REG,
    **fmt_('round.l.fmt', ('s', 'd'), OperandType.F_REG_REG),
    **fmt_('round.w.fmt', ('s', 'd'), OperandType.F_REG_REG),
    **fmt_('trunc.l.fmt', ('s', 'd'), OperandType.F_REG_REG),
    **fmt_('trunc.w.fmt', ('s', 'd'), OperandType.F_REG_REG),

    'ldc1': OperandType.F_LOADSTORE,
    'ldxc1': OperandType.F_LOADSTORE_INDEXED,
    'luxc1': OperandType.F_LOADSTORE_INDEXED,
    'lwc1': OperandType.F_LOADSTORE,
    'lwxc1': OperandType.F_LOADSTORE_INDEXED,
    'prefx': OperandType.PREFX_OPERANDS,
    'sdc1': OperandType.F_LOADSTORE,
    'sdxc1': OperandType.F_LOADSTORE_INDEXED,
    'suxc1': OperandType.F_LOADSTORE_INDEXED,
    'swc1': OperandType.F_LOADSTORE,
    'swxc1': OperandType.F_LOADSTORE_INDEXED,

    'cfc1': OperandType.F_CONTROL_MOVE,
    'ctc1': OperandType.F_CONTROL_MOVE,
    'dmfc1': OperandType.F_MOVE,
    'dmtc1': OperandType.F_MOVE,
    'mfc1': OperandType.F_MOVE,
    'mfhc1': OperandType.F_MOVE,
    **fmt_('mov.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG),
    **fmt_('movf.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG_COND),
    **fmt_('movn.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG_CONDREG),
    **fmt_('movt.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG_COND),
    **fmt_('movz.fmt', ('s', 'd', 'ps'), OperandType.F_REG_REG_CONDREG),
    'mtc1': OperandType.F_MOVE,
    'mthc1': OperandType.F_MOVE,
}
# priveleged, trap, coprocessor2 and coprocessor0 instructions are unused for now

