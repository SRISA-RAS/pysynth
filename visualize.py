from __future__ import annotations

import pygraphviz as pgv

from graph import FlowGraph, FlowNode, FuncRet


def codeblock_to_html_(codeblock: FlowNode) -> str:
    result = ''
    for instr in codeblock.instructions:
        result = result + '<tr><td align="center">' + f'{instr.program_counter:#x}' + \
                 '</td><td align="left">' + instr.instruction + '</td></tr>'
    return result


def generate_dotfile(graph: FlowGraph, simplified: bool = False) -> pgv.AGraph:
    """
    Creates an object of a Graphviz dotfile for a given flow graph.

    :param FlowGraph graph: source flow graph
    :param bool simplified: create a simplified graph
    :return: AGraph graph object
    """
    dot_g = pgv.AGraph(strict=True, directed=True, concentrate=True, ranksep=(0.3 if simplified else 1.0))

    for addr, codeblock in graph.nodes.items():
        dot_g.add_node(addr)
        node = dot_g.get_node(addr)
        node.attr["shape"] = 'Mrecord'
        if simplified:
            node.attr["label"] = f'<<table border="0" cellborder="0" cellpadding="3">' \
                                 f'<tr><td>{addr:#016x}</td></tr>' \
                                 f'<tr><td>{sum(graph.nodes[addr].metrics.instr_by_type.values())} instructions' \
                                 f'</td></tr></table>>'
        else:
            node.attr["label"] = f'<<table border="0" cellborder="0" cellpadding="3">' \
                                 f'<tr><td align="center" colspan="2">{addr:#016x}</td>' \
                                 f'</tr>{codeblock_to_html_(codeblock)}</table>>'

    for node in graph.nodes.values():
        if isinstance(node.flowchange, FuncRet):
            for target_addr, count in node.flowchange.recorded_returns.items():
                dot_g.add_edge(node.address, target_addr, label=count, headport='n', tailport='s')
            continue
        for edge in node.flowchange:
            dot_g.add_edge(node.address, edge.to_addr, label=edge.count, headport='n', tailport='s')

    return dot_g


def visualize_graph(graph: pgv.AGraph, filename: str, dpi: int = 50):
    """
    Saves an image of a Graphviz formed flow graph.

    :param pgv.AGraph graph: flow graph
    :param str filename: output file name
    :param int dpi: output file resolution
    :return: None
    """
    # TODO: improve the readability of the resulting graph
    # splines=ortho
    # graph.graph_attr.update(splines="false")
    graph.graph_attr.update(dpi=dpi)
    graph.draw(filename, 'jpg', 'dot')

