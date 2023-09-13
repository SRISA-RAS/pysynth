import argparse
from pathlib import Path
import random
import sys

from graph_builder import GraphBuilder
from generate import generate_code_from_graph


def main():
    parser = argparse.ArgumentParser(
        description='Builds a Statistical Flow Graph of a program from an assembly trace'
    )
    parser.add_argument('source',
                        help='path to the trace file')
    parser.add_argument('-i', '--image',
                        help='save the visualization of a graph as an image into a given file')
    parser.add_argument('-o', '--out',
                        help='save an extracted synthesis of a given trace as an assembly code into a given directory')
    parser.add_argument('-v', '--verbose',
                        help='output program status to the terminal',
                        action='store_true')
    parser.add_argument('--loops',
                        help='choose loop reduction technique',
                        type=str, choices=['all', 'none'], default='all')
    parser.add_argument('-r', '--reduction',
                        help='reduce loops by this amount (WARNING: currently ignored)',
                        type=int, default=100)
    parser.add_argument('-n', '--name',
                        help='write this at the start of a resulting test file',
                        type=str, default=None)
    parser.add_argument('-s', '--seed',
                        help='a number to use as a seed for the random number generator',
                        type=int, default=random.randrange(sys.maxsize))

    args = parser.parse_args()
    with open(args.source) as file:
        if args.verbose:
            print('Parsing log...')
        builder = GraphBuilder(file)
        flowgraph = builder.build()
        if args.verbose:
            print('Generated graph...')
        if args.image:
            from visualize import generate_dotfile, visualize_graph
            if args.verbose:
                print('Graph visualization...')
            visualize_graph(generate_dotfile(flowgraph, True), args.image)
        if args.out:
            out_dir = Path(args.out)
            if not out_dir.is_dir():
                out_dir = Path.cwd() / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / 'test.S', 'w') as output, open(out_dir / 'btable.bin', 'wb') as out_branches,\
                    open(out_dir / 'jtable.bin', 'wb') as out_jumps, open(out_dir / 'ram_table.ini', 'w') as ram_table,\
		    open(out_dir / 'defines.h', 'w') as defines:
                if args.verbose:
                    print('Generating code...')
                generate_code_from_graph(flowgraph, output, out_branches, out_jumps, ram_table, defines, args.loops,
                                         args.reduction, args.name, args.seed, args.source)
        if args.verbose:
            print('Done!')


if __name__ == '__main__':
    main()

