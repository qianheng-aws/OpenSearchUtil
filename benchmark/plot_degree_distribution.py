#!/usr/bin/env python3
"""
Query vertex degree distribution from OpenSearch and plot the chart.

Usage:
    python benchmark/plot_degree_distribution.py --host localhost --port 9200
    python benchmark/plot_degree_distribution.py --host localhost --port 9200 --username admin --password admin --use-ssl
    python benchmark/plot_degree_distribution.py --host localhost --port 9200 --edge-index person_knows_person
"""

import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import requests
from requests.auth import HTTPBasicAuth

BASE_DIR = '/Users/qianheng/Documents/Code/Python/OpenSearchIngestion/benchmark'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def execute_ppl(endpoint, query, auth=None, verify=False):
    """Execute a PPL query and return the result."""
    response = requests.post(
        endpoint,
        headers={'Content-Type': 'application/json'},
        json={'query': query},
        auth=auth,
        verify=verify,
        timeout=300
    )
    response.raise_for_status()
    return response.json()


def query_degree_distribution(endpoint, edge_index, auth=None, verify=False):
    """Query degree of each vertex using multisearch PPL."""
    query = (
        f"| multisearch "
        f"[source={edge_index} | fields source] "
        f"[source={edge_index} | eval source=target | fields source]\n"
        f"| stats count as cnt by source\n"
        f"| sort -cnt"
    )

    logger.info(f"Executing PPL query:\n{query}")
    result = execute_ppl(endpoint, query, auth, verify)

    # Parse result: each row is [cnt, source] or [source, cnt]
    degrees = []
    if 'datarows' in result:
        # Determine column order from schema
        schema = result.get('schema', [])
        cnt_idx = 0
        for i, col in enumerate(schema):
            if col.get('name') == 'cnt':
                cnt_idx = i
                break

        for row in result['datarows']:
            degrees.append(int(row[cnt_idx]))

    logger.info(f"Got degree data for {len(degrees)} vertices")
    return degrees


def plot_degree_distribution(degrees, output_path):
    """Plot degree distribution: histogram + CDF."""
    degrees = np.array(degrees)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Vertex Degree Distribution (person_knows_person)', fontsize=16, fontweight='bold', y=1.02)

    # Left: Degree histogram (log-log)
    max_deg = degrees.max()
    bins = np.logspace(0, np.log10(max_deg + 1), 50)
    ax1.hist(degrees, bins=bins, color='#2196F3', edgecolor='white', alpha=0.85)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Degree', fontsize=12)
    ax1.set_ylabel('# Vertices (count)', fontsize=12)
    ax1.set_title('Degree Histogram (log-log)', fontsize=14)
    ax1.grid(True, alpha=0.3, which='both')

    # Add stats annotation
    stats_text = (
        f"Vertices: {len(degrees)}\n"
        f"Min: {degrees.min()}\n"
        f"Max: {degrees.max()}\n"
        f"Mean: {degrees.mean():.1f}\n"
        f"Median: {np.median(degrees):.0f}"
    )
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

    # Right: CCDF (complementary cumulative distribution)
    sorted_deg = np.sort(degrees)[::-1]
    ccdf_y = np.arange(1, len(sorted_deg) + 1) / len(sorted_deg)
    ax2.plot(sorted_deg, ccdf_y, color='#F44336', linewidth=1.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Degree', fontsize=12)
    ax2.set_ylabel('P(X >= degree)', fontsize=12)
    ax2.set_title('CCDF (log-log)', fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot vertex degree distribution from OpenSearch')

    parser.add_argument('--host', default='localhost', help='OpenSearch host (default: localhost)')
    parser.add_argument('--port', type=int, default=9200, help='OpenSearch port (default: 9200)')
    parser.add_argument('--username', help='Username for authentication')
    parser.add_argument('--password', help='Password for authentication')
    parser.add_argument('--use-ssl', action='store_true', help='Use SSL connection')
    parser.add_argument('--verify-certs', action='store_true', help='Verify SSL certificates')
    parser.add_argument('--edge-index', default='person_knows_person',
                        help='Edge index name (default: person_knows_person)')
    parser.add_argument('--output', '-o', default=None, help='Output image path')

    args = parser.parse_args()

    protocol = 'https' if args.use_ssl else 'http'
    endpoint = f"{protocol}://{args.host}:{args.port}/_plugins/_ppl"
    auth = HTTPBasicAuth(args.username, args.password) if args.username and args.password else None
    verify = args.verify_certs if args.use_ssl else False

    output_path = args.output or f'{BASE_DIR}/degree_distribution.png'

    degrees = query_degree_distribution(endpoint, args.edge_index, auth, verify)

    if not degrees:
        print("No degree data returned. Check your connection and index name.")
        return

    plot_degree_distribution(degrees, output_path)

    # Also save raw data as JSON
    json_path = output_path.rsplit('.', 1)[0] + '.json'
    with open(json_path, 'w') as f:
        json.dump({'degrees': sorted(degrees, reverse=True)}, f)
    print(f"Raw data saved to: {json_path}")


if __name__ == '__main__':
    main()
