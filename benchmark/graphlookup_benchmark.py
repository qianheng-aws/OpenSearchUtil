#!/usr/bin/env python3
"""
GraphLookup Benchmark for OpenSearch PPL

This script benchmarks the graphLookup command in OpenSearch PPL.
It measures latency for different maxDepth values and direction settings.
"""

import time
import json
import logging
import argparse
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import requests
from requests.auth import HTTPBasicAuth


@dataclass
class BenchmarkConfig:
    """Configuration for graphLookup benchmark."""
    # OpenSearch connection
    host: str = 'localhost'
    port: int = 9200
    username: Optional[str] = None
    password: Optional[str] = None
    use_ssl: bool = False
    verify_certs: bool = False

    # Benchmark settings
    vertex_index: str = 'person'
    edge_index: str = 'connection'
    start_field: str = 'id'
    from_field: str = 'target'
    to_field: str = 'source'
    num_start_values: int = 10
    runs_per_test: int = 5
    max_depths: List[int] = field(default_factory=lambda: [0, 1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    directions: List[str] = field(default_factory=lambda: ['uni', 'bi'])


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    start_value: Any
    max_depth: int
    direction: str
    latencies: List[float]
    median_latency: float
    edge_count: int
    node_count: int
    error: Optional[str] = None


@dataclass
class MultiValueBenchmarkResult:
    """Result of a multi-value benchmark run."""
    num_start_values: int
    start_values: List[Any]
    max_depth: int
    direction: str
    latencies: List[float]
    median_latency: float
    edge_count: int
    node_count: int
    error: Optional[str] = None


class GraphLookupBenchmark:
    """
    Benchmark class for OpenSearch PPL graphLookup command.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark with configuration.

        Args:
            config: BenchmarkConfig instance with connection and benchmark settings
        """
        self.config = config

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Build base URL
        protocol = 'https' if config.use_ssl else 'http'
        self.base_url = f"{protocol}://{config.host}:{config.port}"
        self.ppl_endpoint = f"{self.base_url}/_plugins/_ppl"

        # Setup authentication
        self.auth = None
        if config.username and config.password:
            self.auth = HTTPBasicAuth(config.username, config.password)

        # SSL verification
        self.verify = config.verify_certs if config.use_ssl else False

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to OpenSearch."""
        try:
            response = requests.get(
                self.base_url,
                auth=self.auth,
                verify=self.verify,
                timeout=10
            )
            response.raise_for_status()
            info = response.json()
            self.logger.info(f"Connected to OpenSearch cluster: {info.get('cluster_name', 'unknown')}")
        except Exception as e:
            self.logger.error(f"Failed to connect to OpenSearch: {e}")
            raise

    def execute_ppl(self, query: str) -> Dict:
        """
        Execute a PPL query and return the result.

        Args:
            query: PPL query string

        Returns:
            Query result as dictionary
        """
        headers = {'Content-Type': 'application/json'}
        payload = {'query': query}

        response = requests.post(
            self.ppl_endpoint,
            headers=headers,
            json=payload,
            auth=self.auth,
            verify=self.verify,
            timeout=300  # 5 minutes timeout for long queries
        )
        response.raise_for_status()
        return response.json()

    def get_random_start_values(self) -> List[Any]:
        """
        Get random start values from the vertex index.

        Returns:
            List of random start values (IDs)
        """
        query = f"""source={self.config.vertex_index}
| eval a = rand()
| sort a
| fields {self.config.start_field}, a
| head {self.config.num_start_values}"""

        self.logger.info(f"Getting {self.config.num_start_values} random start values...")
        self.logger.debug(f"Query: {query}")

        result = self.execute_ppl(query)

        # Extract IDs from result
        start_values = []
        if 'datarows' in result:
            for row in result['datarows']:
                if row and len(row) > 0:
                    start_values.append(row[0])

        self.logger.info(f"Got {len(start_values)} start values: {start_values}")
        return start_values

    def run_graphlookup_query(self, start_value: Any, max_depth: int, direction: str) -> Tuple[float, int, int]:
        """
        Run a single graphLookup query and measure latency.

        Args:
            start_value: The starting vertex ID
            max_depth: Maximum traversal depth
            direction: Traversal direction ('uni' or 'bi')

        Returns:
            Tuple of (latency_ms, edge_count, node_count)
        """
        query = f"""source={self.config.vertex_index}
| where {self.config.start_field} = {start_value}
| graphLookup {self.config.edge_index}
    startField={self.config.start_field}
    fromField={self.config.from_field}
    toField={self.config.to_field}
    depthField=depth
    maxDepth={max_depth}
    direction={direction}
    as reports
| fields reports"""

        start_time = time.perf_counter()
        result = self.execute_ppl(query)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Extract edge count and node count from reports
        edge_count = 0
        node_count = 0

        if 'datarows' in result and result['datarows'] and result['datarows'][0]:
            reports = result['datarows'][0][0]
            if reports and isinstance(reports, list):
                edge_count = len(reports)

                # Extract unique nodes from edges
                # Each report is like "{creationDate, source, target, depth}"
                nodes = set()
                for report in reports:
                    if isinstance(report, str):
                        # Parse the string format: "{val1, val2, val3, val4}"
                        try:
                            parts = report.strip('{}').split(',')
                            if len(parts) >= 3:
                                source = parts[1].strip()
                                target = parts[2].strip()
                                nodes.add(source)
                                nodes.add(target)
                        except Exception:
                            pass
                    elif isinstance(report, dict):
                        # Handle dict format if returned as object
                        if 'source' in report:
                            nodes.add(str(report['source']))
                        if 'target' in report:
                            nodes.add(str(report['target']))
                        if self.config.to_field in report:
                            nodes.add(str(report[self.config.to_field]))
                        if self.config.from_field in report:
                            nodes.add(str(report[self.config.from_field]))

                node_count = len(nodes)

        return latency_ms, edge_count, node_count

    def benchmark_single_config(self, start_value: Any, max_depth: int, direction: str) -> BenchmarkResult:
        """
        Run benchmark for a single configuration (start_value, max_depth, direction).

        Args:
            start_value: The starting vertex ID
            max_depth: Maximum traversal depth
            direction: Traversal direction

        Returns:
            BenchmarkResult with latency statistics
        """
        latencies = []
        edge_count = 0
        node_count = 0
        error = None

        for run in range(self.config.runs_per_test):
            try:
                latency, edges, nodes = self.run_graphlookup_query(start_value, max_depth, direction)
                latencies.append(latency)
                edge_count = edges
                node_count = nodes
                self.logger.debug(f"  Run {run + 1}: {latency:.2f}ms, edges={edges}, nodes={nodes}")
            except Exception as e:
                error = str(e)
                self.logger.warning(f"  Run {run + 1} failed: {e}")

        median_latency = statistics.median(latencies) if latencies else 0

        return BenchmarkResult(
            start_value=start_value,
            max_depth=max_depth,
            direction=direction,
            latencies=latencies,
            median_latency=median_latency,
            edge_count=edge_count,
            node_count=node_count,
            error=error
        )

    def run_multivalue_graphlookup_query(self, start_values: List[Any], max_depth: int, direction: str) -> Tuple[float, int, int]:
        """
        Run a graphLookup query with multiple start values using IN clause.

        Args:
            start_values: List of starting vertex IDs
            max_depth: Maximum traversal depth
            direction: Traversal direction ('uni' or 'bi')

        Returns:
            Tuple of (latency_ms, edge_count, node_count)
        """
        # Build IN clause: where id in (val1, val2, ...)
        values_str = ', '.join(str(v) for v in start_values)

        query = f"""source={self.config.vertex_index}
| where {self.config.start_field} in ({values_str})
| graphLookup {self.config.edge_index}
    startField={self.config.start_field}
    fromField={self.config.from_field}
    toField={self.config.to_field}
    depthField=depth
    batchMode=true
    maxDepth={max_depth}
    direction={direction}
    as reports
| fields reports"""

        start_time = time.perf_counter()
        result = self.execute_ppl(query)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Extract edge count and node count from all reports rows
        total_edge_count = 0
        all_nodes = set()

        if 'datarows' in result:
            for row in result['datarows']:
                if row and row[0]:
                    reports = row[0]
                    if reports and isinstance(reports, list):
                        total_edge_count += len(reports)

                        # Extract unique nodes from edges
                        for report in reports:
                            if isinstance(report, str):
                                try:
                                    parts = report.strip('{}').split(',')
                                    if len(parts) >= 3:
                                        source = parts[1].strip()
                                        target = parts[2].strip()
                                        all_nodes.add(source)
                                        all_nodes.add(target)
                                except Exception:
                                    pass
                            elif isinstance(report, dict):
                                if 'source' in report:
                                    all_nodes.add(str(report['source']))
                                if 'target' in report:
                                    all_nodes.add(str(report['target']))
                                if self.config.to_field in report:
                                    all_nodes.add(str(report[self.config.to_field]))
                                if self.config.from_field in report:
                                    all_nodes.add(str(report[self.config.from_field]))

        return latency_ms, total_edge_count, len(all_nodes)

    def benchmark_multivalue_config(self, start_values: List[Any], max_depth: int, direction: str) -> MultiValueBenchmarkResult:
        """
        Run benchmark for a multi-value configuration.

        Args:
            start_values: List of starting vertex IDs
            max_depth: Maximum traversal depth
            direction: Traversal direction

        Returns:
            MultiValueBenchmarkResult with latency statistics
        """
        latencies = []
        edge_count = 0
        node_count = 0
        error = None

        for run in range(self.config.runs_per_test):
            try:
                latency, edges, nodes = self.run_multivalue_graphlookup_query(start_values, max_depth, direction)
                latencies.append(latency)
                edge_count = edges
                node_count = nodes
                self.logger.debug(f"  Run {run + 1}: {latency:.2f}ms, edges={edges}, nodes={nodes}")
            except Exception as e:
                error = str(e)
                self.logger.warning(f"  Run {run + 1} failed: {e}")

        median_latency = statistics.median(latencies) if latencies else 0

        return MultiValueBenchmarkResult(
            num_start_values=len(start_values),
            start_values=start_values,
            max_depth=max_depth,
            direction=direction,
            latencies=latencies,
            median_latency=median_latency,
            edge_count=edge_count,
            node_count=node_count,
            error=error
        )

    def run_multivalue_benchmark(self, max_depth: int = 3) -> List[MultiValueBenchmarkResult]:
        """
        Run multi-value benchmark: incrementally add start values and measure performance.

        Args:
            max_depth: Fixed maxDepth for all tests (default: 3)

        Returns:
            List of MultiValueBenchmarkResult for each batch size
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Multi-Value GraphLookup Benchmark")
        self.logger.info("=" * 60)
        self.logger.info(f"Vertex Index: {self.config.vertex_index}")
        self.logger.info(f"Edge Index: {self.config.edge_index}")
        self.logger.info(f"Fixed maxDepth: {max_depth}")
        self.logger.info(f"Directions: {self.config.directions}")
        self.logger.info(f"Runs per test: {self.config.runs_per_test}")
        self.logger.info(f"Max start values: {self.config.num_start_values}")
        self.logger.info("=" * 60)

        # Get random start values
        all_start_values = self.get_random_start_values()

        if not all_start_values:
            self.logger.error("No start values found. Aborting benchmark.")
            return []

        results = []
        total_tests = len(all_start_values) * len(self.config.directions)
        current_test = 0

        for direction in self.config.directions:
            self.logger.info(f"\n{'=' * 40}")
            self.logger.info(f"Testing direction: {direction}")
            self.logger.info(f"{'=' * 40}")

            # Incrementally add start values: 1, 2, 3, ..., num_start_values
            for num_values in range(1, len(all_start_values) + 1):
                current_test += 1
                start_values_subset = all_start_values[:num_values]

                self.logger.info(f"\n[{current_test}/{total_tests}] "
                                f"num_start_values={num_values}, maxDepth={max_depth}, direction={direction}")
                self.logger.info(f"  Start values: {start_values_subset}")

                result = self.benchmark_multivalue_config(start_values_subset, max_depth, direction)
                results.append(result)

                self.logger.info(f"  Median latency: {result.median_latency:.2f}ms, "
                                f"Edges: {result.edge_count}, Nodes: {result.node_count}")

        return results

    def generate_multivalue_report(self, results: List[MultiValueBenchmarkResult], max_depth: int = 3) -> Dict:
        """
        Generate a summary report from multi-value benchmark results.

        Args:
            results: List of MultiValueBenchmarkResult
            max_depth: The fixed maxDepth used

        Returns:
            Summary report as dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_type': 'multivalue',
            'config': {
                'vertex_index': self.config.vertex_index,
                'edge_index': self.config.edge_index,
                'max_depth': max_depth,
                'max_start_values': self.config.num_start_values,
                'runs_per_test': self.config.runs_per_test,
                'directions': self.config.directions
            },
            'summary': {},
            'details': []
        }

        # Group results by direction
        for direction in self.config.directions:
            report['summary'][direction] = {}

            for num_values in range(1, self.config.num_start_values + 1):
                matching_results = [
                    r for r in results
                    if r.direction == direction and r.num_start_values == num_values
                ]

                if matching_results:
                    r = matching_results[0]
                    report['summary'][direction][num_values] = {
                        'median_latency_ms': r.median_latency,
                        'edge_count': r.edge_count,
                        'node_count': r.node_count
                    }

        # Add detailed results
        for r in results:
            report['details'].append({
                'num_start_values': r.num_start_values,
                'start_values': r.start_values,
                'max_depth': r.max_depth,
                'direction': r.direction,
                'median_latency_ms': r.median_latency,
                'all_latencies_ms': r.latencies,
                'edge_count': r.edge_count,
                'node_count': r.node_count,
                'error': r.error
            })

        return report

    def print_multivalue_summary_table(self, results: List[MultiValueBenchmarkResult]):
        """
        Print a summary table of multi-value benchmark results.

        Args:
            results: List of MultiValueBenchmarkResult
        """
        print("\n" + "=" * 90)
        print("MULTI-VALUE BENCHMARK SUMMARY")
        print("=" * 90)

        for direction in self.config.directions:
            print(f"\n--- Direction: {direction} ---")
            print(f"{'# Values':<12} {'Median Latency (ms)':<22} {'Edges':<15} {'Nodes':<15}")
            print("-" * 90)

            for num_values in range(1, self.config.num_start_values + 1):
                matching_results = [
                    r for r in results
                    if r.direction == direction and r.num_start_values == num_values
                ]

                if matching_results:
                    r = matching_results[0]
                    print(f"{r.num_start_values:<12} "
                          f"{r.median_latency:<22.2f} "
                          f"{r.edge_count:<15} "
                          f"{r.node_count:<15}")

        print("\n" + "=" * 90)

    def run_benchmark(self) -> List[BenchmarkResult]:
        """
        Run the complete benchmark suite.

        Returns:
            List of BenchmarkResult for all configurations
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting GraphLookup Benchmark")
        self.logger.info("=" * 60)
        self.logger.info(f"Vertex Index: {self.config.vertex_index}")
        self.logger.info(f"Edge Index: {self.config.edge_index}")
        self.logger.info(f"Max Depths: {self.config.max_depths}")
        self.logger.info(f"Directions: {self.config.directions}")
        self.logger.info(f"Runs per test: {self.config.runs_per_test}")
        self.logger.info("=" * 60)

        # Get random start values
        start_values = self.get_random_start_values()

        if not start_values:
            self.logger.error("No start values found. Aborting benchmark.")
            return []

        results = []
        total_tests = len(start_values) * len(self.config.max_depths) * len(self.config.directions)
        current_test = 0

        for direction in self.config.directions:
            self.logger.info(f"\n{'=' * 40}")
            self.logger.info(f"Testing direction: {direction}")
            self.logger.info(f"{'=' * 40}")

            for max_depth in self.config.max_depths:
                self.logger.info(f"\n--- maxDepth={max_depth} ---")

                for start_value in start_values:
                    current_test += 1
                    self.logger.info(f"[{current_test}/{total_tests}] "
                                    f"start={start_value}, maxDepth={max_depth}, direction={direction}")

                    result = self.benchmark_single_config(start_value, max_depth, direction)
                    results.append(result)

                    self.logger.info(f"  Median latency: {result.median_latency:.2f}ms, "
                                    f"Edges: {result.edge_count}, Nodes: {result.node_count}")

        return results

    def generate_report(self, results: List[BenchmarkResult]) -> Dict:
        """
        Generate a summary report from benchmark results.

        Args:
            results: List of BenchmarkResult

        Returns:
            Summary report as dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'vertex_index': self.config.vertex_index,
                'edge_index': self.config.edge_index,
                'num_start_values': self.config.num_start_values,
                'runs_per_test': self.config.runs_per_test,
                'max_depths': self.config.max_depths,
                'directions': self.config.directions
            },
            'summary': {},
            'details': []
        }

        # Group results by direction and maxDepth
        for direction in self.config.directions:
            report['summary'][direction] = {}

            for max_depth in self.config.max_depths:
                matching_results = [
                    r for r in results
                    if r.direction == direction and r.max_depth == max_depth
                ]

                if matching_results:
                    latencies = [r.median_latency for r in matching_results]
                    edge_counts = [r.edge_count for r in matching_results]
                    node_counts = [r.node_count for r in matching_results]

                    report['summary'][direction][max_depth] = {
                        'avg_latency_ms': statistics.mean(latencies),
                        'median_latency_ms': statistics.median(latencies),
                        'min_latency_ms': min(latencies),
                        'max_latency_ms': max(latencies),
                        'avg_edge_count': statistics.mean(edge_counts),
                        'avg_node_count': statistics.mean(node_counts),
                        'num_tests': len(matching_results)
                    }

        # Add detailed results
        for r in results:
            report['details'].append({
                'start_value': r.start_value,
                'max_depth': r.max_depth,
                'direction': r.direction,
                'median_latency_ms': r.median_latency,
                'all_latencies_ms': r.latencies,
                'edge_count': r.edge_count,
                'node_count': r.node_count,
                'error': r.error
            })

        return report

    def print_summary_table(self, results: List[BenchmarkResult]):
        """
        Print a summary table of benchmark results.

        Args:
            results: List of BenchmarkResult
        """
        print("\n" + "=" * 100)
        print("BENCHMARK SUMMARY")
        print("=" * 100)

        for direction in self.config.directions:
            print(f"\n--- Direction: {direction} ---")
            print(f"{'maxDepth':<10} {'Avg Latency (ms)':<18} {'Median (ms)':<15} {'Min (ms)':<12} {'Max (ms)':<12} {'Avg Edges':<12} {'Avg Nodes':<12}")
            print("-" * 100)

            for max_depth in self.config.max_depths:
                matching_results = [
                    r for r in results
                    if r.direction == direction and r.max_depth == max_depth
                ]

                if matching_results:
                    latencies = [r.median_latency for r in matching_results]
                    edge_counts = [r.edge_count for r in matching_results]
                    node_counts = [r.node_count for r in matching_results]

                    print(f"{max_depth:<10} "
                          f"{statistics.mean(latencies):<18.2f} "
                          f"{statistics.median(latencies):<15.2f} "
                          f"{min(latencies):<12.2f} "
                          f"{max(latencies):<12.2f} "
                          f"{statistics.mean(edge_counts):<12.0f} "
                          f"{statistics.mean(node_counts):<12.0f}")

        print("\n" + "=" * 100)


def main():
    """Main function to run the benchmark from command line."""
    parser = argparse.ArgumentParser(description='Benchmark OpenSearch PPL graphLookup')

    # OpenSearch connection arguments
    parser.add_argument('--host', default='localhost', help='OpenSearch host (default: localhost)')
    parser.add_argument('--port', type=int, default=9200, help='OpenSearch port (default: 9200)')
    parser.add_argument('--username', help='Username for authentication')
    parser.add_argument('--password', help='Password for authentication')
    parser.add_argument('--use-ssl', action='store_true', help='Use SSL connection')
    parser.add_argument('--verify-certs', action='store_true', help='Verify SSL certificates')

    # Benchmark configuration
    parser.add_argument('--vertex-index', default='person', help='Vertex index name (default: person)')
    parser.add_argument('--edge-index', default='connection', help='Edge index name (default: connection)')
    parser.add_argument('--start-field', default='id', help='Start field name (default: id)')
    parser.add_argument('--from-field', default='target', help='From field name (default: target)')
    parser.add_argument('--to-field', default='source', help='To field name (default: source)')
    parser.add_argument('--num-start-values', type=int, default=10, help='Number of random start values (default: 10)')
    parser.add_argument('--runs-per-test', type=int, default=5, help='Runs per test for median calculation (default: 5)')
    parser.add_argument('--max-depths', default='0,1,3,5,10,15,20,25,30,35,40,45,50',
                        help='Comma-separated max depths to test (default: 0,1,3,5,10,...,50)')
    parser.add_argument('--directions', default='uni,bi',
                        help='Comma-separated directions to test (default: uni,bi)')

    # Benchmark mode
    parser.add_argument('--multivalue-mode', action='store_true',
                        help='Run multi-value benchmark (uses IN clause with incrementally added values)')
    parser.add_argument('--multivalue-depth', type=int, default=3,
                        help='Fixed maxDepth for multi-value benchmark (default: 3)')

    # Output options
    parser.add_argument('--output', '-o', help='Output file for JSON report')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress detailed output')

    args = parser.parse_args()

    # Parse list arguments
    max_depths = [int(x.strip()) for x in args.max_depths.split(',')]
    directions = [x.strip() for x in args.directions.split(',')]

    # Create configuration
    config = BenchmarkConfig(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        use_ssl=args.use_ssl,
        verify_certs=args.verify_certs,
        vertex_index=args.vertex_index,
        edge_index=args.edge_index,
        start_field=args.start_field,
        from_field=args.from_field,
        to_field=args.to_field,
        num_start_values=args.num_start_values,
        runs_per_test=args.runs_per_test,
        max_depths=max_depths,
        directions=directions
    )

    # Set log level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        # Create benchmark instance
        benchmark = GraphLookupBenchmark(config)

        if args.multivalue_mode:
            # Run multi-value benchmark
            results = benchmark.run_multivalue_benchmark(max_depth=args.multivalue_depth)

            # Print summary table
            benchmark.print_multivalue_summary_table(results)

            # Generate and save report
            report = benchmark.generate_multivalue_report(results, max_depth=args.multivalue_depth)
        else:
            # Run standard benchmark
            results = benchmark.run_benchmark()

            # Print summary table
            benchmark.print_summary_table(results)

            # Generate and save report
            report = benchmark.generate_report(results)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {args.output}")

        print("\nBenchmark completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
