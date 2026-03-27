# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "networkx>=3.2",
#     "matplotlib>=3.8",
#     "tabulate>=0.9",
# ]
# ///
"""Iteration graph analysis for f(x) = x^2 + c (mod p).

Each element x in Z_p maps to f(x) mod p, forming a directed graph (the
"iteration digraph"). This script builds those graphs for small primes and
constants, then analyzes their structure: cycles, fixed points, component
sizes, and in-degree distributions.

Run: uv run examples/iteration_graph_analysis.py
"""

from __future__ import annotations

from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate


# -- graph construction -------------------------------------------------------


def iteration_digraph(p: int, c: int) -> nx.DiGraph:
    """Build the iteration digraph for f(x) = x^2 + c over Z_p.

    Each node x in {0, ..., p-1} has exactly one outgoing edge to f(x) mod p.
    This is a functional graph: out-degree is always 1.

    graphops equivalent: build an adjacency list from a closure, then convert
    to a DiGraph. graphops stores this as a CSR-style edge list.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(p))
    for x in range(p):
        G.add_edge(x, (x * x + c) % p)
    return G


# -- analysis -----------------------------------------------------------------


def find_all_cycles(G: nx.DiGraph) -> list[list[int]]:
    """Find all simple cycles in the iteration digraph.

    graphops equivalent: Johnson's algorithm via graphops::algo::cycles,
    or SCC decomposition followed by cycle enumeration within each SCC.
    """
    return list(nx.simple_cycles(G))


def find_fixed_points(G: nx.DiGraph) -> list[int]:
    """Nodes where f(x) = x, i.e. self-loops.

    graphops equivalent: filter the edge list for (u, u) pairs.
    """
    return sorted(x for x in G.nodes() if G.has_edge(x, x))


def component_sizes(G: nx.DiGraph) -> list[int]:
    """Sizes of weakly connected components, sorted descending.

    graphops equivalent: graphops::algo::wcc (union-find based).
    """
    components = list(nx.weakly_connected_components(G))
    return sorted((len(c) for c in components), reverse=True)


def in_degree_distribution(G: nx.DiGraph) -> dict[int, int]:
    """Map from in-degree k to count of nodes with that in-degree.

    graphops equivalent: degree histogram from CSR column counts.
    """
    return dict(sorted(Counter(d for _, d in G.in_degree()).items()))


def analyze_graph(p: int, c: int) -> dict:
    """Full analysis of a single (p, c) iteration digraph."""
    G = iteration_digraph(p, c)
    cycles = find_all_cycles(G)
    cycle_lengths = sorted(len(cyc) for cyc in cycles)
    fixed = find_fixed_points(G)
    components = component_sizes(G)
    in_deg = in_degree_distribution(G)

    return {
        "p": p,
        "c": c,
        "graph": G,
        "cycles": cycles,
        "cycle_lengths": cycle_lengths,
        "fixed_points": fixed,
        "num_components": len(components),
        "component_sizes": components,
        "in_degree_dist": in_deg,
    }


# -- visualization ------------------------------------------------------------


def draw_iteration_graph(
    ax: plt.Axes,
    result: dict,
) -> None:
    """Draw the iteration digraph with cycles highlighted.

    Cycle nodes are colored red; non-cycle (tail) nodes are light blue.
    """
    G = result["graph"]
    p, c = result["p"], result["c"]
    cycles = result["cycles"]

    cycle_nodes = {node for cyc in cycles for node in cyc}
    tail_nodes = set(G.nodes()) - cycle_nodes

    # Spring layout works well for small functional graphs.
    pos = nx.spring_layout(G, seed=p * 100 + c, k=1.5)

    # Draw tail nodes and edges first.
    nx.draw_networkx_nodes(
        G, pos, nodelist=sorted(tail_nodes), node_color="#a8c4e0", node_size=400, ax=ax
    )
    # Draw cycle nodes.
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=sorted(cycle_nodes),
        node_color="#e06060",
        node_size=500,
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )

    # Separate cycle edges from tail edges for distinct styling.
    cycle_edges = []
    tail_edges = []
    for u, v in G.edges():
        if u in cycle_nodes and v in cycle_nodes:
            cycle_edges.append((u, v))
        else:
            tail_edges.append((u, v))

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=tail_edges,
        edge_color="#888888",
        arrows=True,
        arrowsize=15,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=cycle_edges,
        edge_color="#c03030",
        width=2.0,
        arrows=True,
        arrowsize=15,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    cycle_str = ", ".join(str(l) for l in result["cycle_lengths"]) or "none"
    fixed_str = ", ".join(str(x) for x in result["fixed_points"]) or "none"
    ax.set_title(
        f"p={p}, c={c}\ncycles: [{cycle_str}]  fixed: {{{fixed_str}}}",
        fontsize=10,
    )
    ax.axis("off")


# -- main ---------------------------------------------------------------------


def main() -> None:
    primes = [5, 7, 11, 13]
    constants = [0, 1, 2]

    # Analyze all combinations.
    results = []
    for p, c in product(primes, constants):
        results.append(analyze_graph(p, c))

    # Print summary table.
    table_rows = []
    for r in results:
        table_rows.append(
            [
                r["p"],
                r["c"],
                r["num_components"],
                str(r["component_sizes"]),
                str(r["cycle_lengths"]),
                str(r["fixed_points"]),
                str(r["in_degree_dist"]),
            ]
        )

    headers = [
        "p",
        "c",
        "#comp",
        "comp sizes",
        "cycle lengths",
        "fixed pts",
        "in-deg dist",
    ]
    print("Iteration digraphs for f(x) = x^2 + c (mod p)")
    print("=" * 80)
    print(tabulate(table_rows, headers=headers, tablefmt="simple"))
    print()

    # Visualize a selection: pick (p=5, c=0), (p=7, c=1), (p=11, c=2).
    # These show different structural patterns (fixed points, multi-cycles,
    # large tails).
    vis_params = [(5, 0), (7, 1), (11, 2)]
    vis_results = [r for r in results if (r["p"], r["c"]) in vis_params]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, result in zip(axes, vis_results):
        draw_iteration_graph(ax, result)

    fig.suptitle(
        "Iteration digraphs: f(x) = x^2 + c (mod p)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig("examples/iteration_graphs.png", dpi=150, bbox_inches="tight")
    print("Saved figure to examples/iteration_graphs.png")


if __name__ == "__main__":
    main()
