#!/usr/bin/env python3
"""Plot benchmark comparison charts - LDBC SNB SF-1 & SF-30."""

import matplotlib.pyplot as plt

BASE_DIR = '/Users/qianheng/Documents/Code/Python/OpenSearchIngestion/benchmark'

COLORS = {
    'OpenSearch (uni)':     '#2196F3',
    'OpenSearch (bi)':      '#FF9800',
    'OpenSearch PIT (uni)': '#4CAF50',
    'OpenSearch PIT (bi)':  '#F44336',
    'MongoDB':              '#9C27B0',
}
MARKERS = {
    'OpenSearch (uni)':     'o',
    'OpenSearch (bi)':      's',
    'OpenSearch PIT (uni)': '^',
    'OpenSearch PIT (bi)':  'D',
    'MongoDB':              'v',
}


def _plot_lines(ax, x, data):
    for label, latencies in data.items():
        ax.plot(x, latencies, marker=MARKERS[label], color=COLORS[label],
                label=label, linewidth=2, markersize=7)


def plot_sf1():
    """LDBC SNB SF-1: Single Value + Multi Value."""
    max_depths = [0, 1, 3, 5, 10, 30, 50]
    single = {
        'OpenSearch (uni)':     [33.27, 42.35, 122.46, 125.42, 132.10, 124.57, 120.05],
        'OpenSearch (bi)':      [16.43, 67.94, 271.19, 322.30, 336.41, 340.11, 337.63],
        'OpenSearch PIT (uni)': [16.94, 20.31, 73.74, 71.48, 78.27, 78.30, 77.32],
        'OpenSearch PIT (bi)':  [11.39, 73.86, 646.53, 645.95, 639.65, 641.67, 646.20],
        'MongoDB':              [851.00, 1211.60, 3478.00, 5321.60, 6400.90, 6398.30, 6395.70],
    }

    num_values = list(range(1, 11))
    multi = {
        'OpenSearch (uni)':     [274.38, 255.36, 242.09, 236.66, 232.70, 232.66, 248.25, 248.64, 251.86, 260.38],
        'OpenSearch (bi)':      [340.35, 311.57, 312.14, 310.69, 310.91, 310.20, 316.28, 318.84, 318.18, 324.34],
        'OpenSearch PIT (uni)': [251.32, 165.12, 163.61, 233.71, 225.65, 239.91, 231.27, 224.27, 229.60, 222.27],
        'OpenSearch PIT (bi)':  [831.11, 771.12, 604.46, 525.00, 515.61, 455.57, 411.31, 411.00, 410.14, 418.25],
        'MongoDB':              [5925.00, 5937.00, 5938.00, 6216.00, 6896.00, 6918.00, 17837.00, 27880.00, 33911.00, 33852.00],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('LDBC SNB SF-1 Benchmark', fontsize=16, fontweight='bold', y=1.02)

    _plot_lines(ax1, max_depths, single)
    ax1.set_yscale('log')
    ax1.set_xlabel('maxDepth', fontsize=12)
    ax1.set_ylabel('Avg Latency (ms) - log scale', fontsize=12)
    ax1.set_title('Single Value', fontsize=14)
    ax1.set_xticks(max_depths)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')

    _plot_lines(ax2, num_values, multi)
    ax2.set_yscale('log')
    ax2.set_xlabel('# Start Values', fontsize=12)
    ax2.set_ylabel('Latency (ms) - log scale', fontsize=12)
    ax2.set_title('Multi Value (maxDepth=3)', fontsize=14)
    ax2.set_xticks(num_values)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    path = f'{BASE_DIR}/benchmark_sf1.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {path}")


def plot_sf30():
    """LDBC SNB SF-30: Single Value + Multi Value (OpenSearch only, no MongoDB)."""
    max_depths = [0, 1, 3, 5, 10, 30, 50]
    single = {
        'OpenSearch (uni)':     [7.37, 76.14, 378.84, 667.67, 1207.68, 1261.20, 1235.24],
        'OpenSearch (bi)':      [9.44, 194.58, 822.43, 1393.13, 3142.56, 6708.96, 6846.42],
        'OpenSearch PIT (uni)': [30.87, 78.08, 2819.37, 3181.65, 3354.44, 3376.56, 3365.25],
        'OpenSearch PIT (bi)':  [16.03, 365.69, 32715.64, 33241.66, 33345.85, 33316.33, 33319.96],
    }

    num_values = list(range(1, 11))
    multi = {
        'OpenSearch (uni)':     [623.15, 644.44, 640.59, 643.38, 639.60, 641.39, 652.50, 640.00, 646.78, 659.97],
        'OpenSearch (bi)':      [855.60, 823.55, 849.10, 832.79, 840.01, 867.25, 858.93, 850.75, 876.47, 856.70],
        'OpenSearch PIT (uni)': [10116.41, 10159.91, 9912.53, 10197.35, 10211.33, 13342.55, 13338.66, 14373.45, 14281.39, 14347.10],
        'OpenSearch PIT (bi)':  [26899.16, 35928.29, 53917.51, 56978.91, 60448.94, 57930.62, 59624.47, 56592.04, 53925.67, 53847.08],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('LDBC SNB SF-30 Benchmark', fontsize=16, fontweight='bold', y=1.02)

    _plot_lines(ax1, max_depths, single)
    ax1.set_yscale('log')
    ax1.set_xlabel('maxDepth', fontsize=12)
    ax1.set_ylabel('Avg Latency (ms) - log scale', fontsize=12)
    ax1.set_title('Single Value', fontsize=14)
    ax1.set_xticks(max_depths)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')

    _plot_lines(ax2, num_values, multi)
    ax2.set_yscale('log')
    ax2.set_xlabel('# Start Values', fontsize=12)
    ax2.set_ylabel('Latency (ms) - log scale', fontsize=12)
    ax2.set_title('Multi Value (maxDepth=3)', fontsize=14)
    ax2.set_xticks(num_values)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    path = f'{BASE_DIR}/benchmark_sf30.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {path}")


if __name__ == '__main__':
    plot_sf1()
    plot_sf30()
