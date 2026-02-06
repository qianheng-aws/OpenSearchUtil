#!/usr/bin/env python3
"""
CSV to OpenSearch Data Importer

This script reads CSV files and imports the data into OpenSearch using bulk indexing.
Supports various configuration options for OpenSearch connection and CSV parsing.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import RequestError, ConnectionError


class CSVToOpenSearchImporter:
    """
    A class to handle importing CSV file data into OpenSearch.
    """

    def __init__(self,
                 host: str = 'localhost',
                 port: int = 9200,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 use_ssl: bool = False,
                 verify_certs: bool = False,
                 ca_certs_path: Optional[str] = None):
        """
        Initialize the OpenSearch connection.

        Args:
            host: OpenSearch host
            port: OpenSearch port
            username: Authentication username
            password: Authentication password
            use_ssl: Whether to use SSL
            verify_certs: Whether to verify certificates
            ca_certs_path: Path to CA certificates
        """
        self.host = host
        self.port = port

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # OpenSearch client configuration
        client_config = {
            'hosts': [{'host': host, 'port': port}],
            'use_ssl': use_ssl,
            'verify_certs': verify_certs,
            'timeout': 60,
        }

        if username and password:
            client_config['http_auth'] = (username, password)

        if ca_certs_path:
            client_config['ca_certs'] = ca_certs_path

        try:
            self.client = OpenSearch(**client_config)
            # Test connection
            info = self.client.info()
            self.logger.info(f"Connected to OpenSearch cluster: {info['cluster_name']}")
        except Exception as e:
            self.logger.error(f"Failed to connect to OpenSearch: {e}")
            raise

    def read_csv_file(self,
                      file_path: str,
                      delimiter: str = ',',
                      encoding: str = 'utf-8',
                      header: Union[int, List[int], str] = 0,
                      skiprows: Optional[Union[int, List[int]]] = None,
                      nrows: Optional[int] = None,
                      usecols: Optional[List[str]] = None,
                      dtype: Optional[Dict[str, str]] = None,
                      parse_dates: Optional[Union[bool, List[str]]] = None,
                      na_values: Optional[List[str]] = None,
                      quotechar: str = '"',
                      escapechar: Optional[str] = None,
                      replace_dots: bool = False,
                      column_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Read CSV file and return pandas DataFrame.

        Args:
            file_path: Path to the CSV file
            delimiter: Field delimiter (default: comma)
            encoding: File encoding (default: utf-8)
            header: Row number(s) to use as column names
            skiprows: Line numbers to skip
            nrows: Number of rows to read (useful for large files)
            usecols: List of columns to read
            dtype: Dict of column -> type mappings
            parse_dates: List of columns to parse as dates
            na_values: Additional values to recognize as NA/NaN
            quotechar: Character used to denote quoted fields
            escapechar: Character used to escape delimiter
            replace_dots: Replace dots in column names with underscores (for OpenSearch compatibility)
            column_names: Custom column names to use instead of header row

        Returns:
            pandas DataFrame containing the data
        """
        try:
            self.logger.info(f"Reading CSV file: {file_path}")

            read_params = {
                'filepath_or_buffer': file_path,
                'delimiter': delimiter,
                'encoding': encoding,
                'header': header if header != 'none' else None,
                'quotechar': quotechar,
            }

            if skiprows is not None:
                read_params['skiprows'] = skiprows
            if nrows is not None:
                read_params['nrows'] = nrows
            if usecols is not None:
                read_params['usecols'] = usecols
            if dtype is not None:
                read_params['dtype'] = dtype
            if parse_dates is not None:
                read_params['parse_dates'] = parse_dates
            if na_values is not None:
                read_params['na_values'] = na_values
            if escapechar is not None:
                read_params['escapechar'] = escapechar

            df = pd.read_csv(**read_params)

            # Apply custom column names if provided
            if column_names:
                df.columns = column_names
                self.logger.info(f"Applied custom column names: {column_names}")

            # Replace dots in column names with underscores for OpenSearch compatibility
            if replace_dots:
                original_columns = list(df.columns)
                # Handle duplicate columns by adding suffix
                new_columns = []
                seen = {}
                for col in df.columns:
                    new_col = str(col).replace('.', '_')
                    if new_col in seen:
                        seen[new_col] += 1
                        new_col = f"{new_col}_{seen[new_col]}"
                    else:
                        seen[new_col] = 0
                    new_columns.append(new_col)
                df.columns = new_columns
                self.logger.info(f"Replaced dots in column names: {original_columns} -> {new_columns}")

            self.logger.info(f"Successfully read {len(df)} records from {file_path}")
            return df

        except Exception as e:
            self.logger.error(f"Error reading CSV file {file_path}: {e}")
            raise

    def create_index_if_not_exists(self, index_name: str, mapping: Optional[Dict] = None) -> bool:
        """
        Create OpenSearch index if it doesn't exist.

        Args:
            index_name: Name of the index
            mapping: Optional index mapping

        Returns:
            True if index was created or already exists
        """
        try:
            if self.client.indices.exists(index=index_name):
                self.logger.info(f"Index '{index_name}' already exists")
                return True

            index_config = {
                'settings': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'refresh_interval': '30s'
                }
            }

            if mapping:
                index_config['mappings'] = mapping

            self.client.indices.create(index=index_name, body=index_config)
            self.logger.info(f"Created index '{index_name}'")
            return True

        except RequestError as e:
            if e.error == 'resource_already_exists_exception':
                self.logger.info(f"Index '{index_name}' already exists")
                return True
            else:
                self.logger.error(f"Error creating index '{index_name}': {e}")
                raise

    def prepare_documents(self, df: pd.DataFrame, index_name: str, doc_id_column: Optional[str] = None) -> List[Dict]:
        """
        Prepare documents for bulk indexing.

        Args:
            df: DataFrame containing the data
            index_name: Target index name
            doc_id_column: Column to use as document ID (optional)

        Returns:
            List of documents formatted for bulk indexing
        """
        import numpy as np

        documents = []

        for idx, row in df.iterrows():
            # Convert row to dictionary and handle various data types
            doc = {}

            for key, value in row.items():
                try:
                    # Handle different data types properly
                    if value is None:
                        doc[key] = None
                    elif isinstance(value, (pd.Timestamp, datetime)):
                        doc[key] = value.isoformat()
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        # Handle array/list types
                        if hasattr(value, 'tolist'):
                            # Convert numpy arrays to lists
                            doc[key] = value.tolist()
                        else:
                            doc[key] = list(value)
                    elif isinstance(value, dict):
                        # Handle dictionary/object types
                        doc[key] = value
                    elif pd.isna(value) if not hasattr(value, '__len__') or len(str(value)) < 50 else False:
                        # Safe NaN check for scalar values only
                        doc[key] = None
                    elif isinstance(value, (int, float, str, bool)):
                        # Handle basic types
                        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                            doc[key] = None
                        else:
                            doc[key] = value
                    else:
                        # Convert other types to string
                        try:
                            # Try to convert to JSON-serializable format
                            if hasattr(value, 'item'):  # numpy scalars
                                doc[key] = value.item()
                            else:
                                doc[key] = str(value)
                        except Exception:
                            doc[key] = str(value)

                except Exception as e:
                    # Fallback: convert to string
                    self.logger.warning(f"Error processing field {key} with value type {type(value)}: {e}")
                    doc[key] = str(value) if value is not None else None

            # Prepare document for bulk API
            document = {
                '_index': index_name,
                '_source': doc
            }

            # Use specified column as document ID if provided
            if doc_id_column and doc_id_column in doc and doc[doc_id_column] is not None:
                document['_id'] = str(doc[doc_id_column])

            documents.append(document)

        return documents

    def bulk_index_documents(self, documents: List[Dict], chunk_size: int = 1000) -> Dict:
        """
        Bulk index documents to OpenSearch.

        Args:
            documents: List of documents to index
            chunk_size: Number of documents per batch

        Returns:
            Dictionary with indexing statistics
        """
        try:
            self.logger.info(f"Starting bulk indexing of {len(documents)} documents")

            success_count = 0
            failed_count = 0
            errors = []

            # Use helpers.bulk for efficient bulk indexing
            for success, info in helpers.parallel_bulk(
                self.client,
                documents,
                chunk_size=chunk_size,
                thread_count=4,
                timeout=60
            ):
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    errors.append(info)
                    self.logger.error(f"Failed to index document: {info}")

            # Refresh the index
            for doc in documents:
                index_name = doc['_index']
                self.client.indices.refresh(index=index_name)
                break

            stats = {
                'total_documents': len(documents),
                'successful': success_count,
                'failed': failed_count,
                'errors': errors
            }

            self.logger.info(f"Bulk indexing completed: {success_count} successful, {failed_count} failed")
            return stats

        except Exception as e:
            self.logger.error(f"Error during bulk indexing: {e}")
            raise

    def import_csv_to_opensearch(self,
                                 csv_file: str,
                                 index_name: str,
                                 doc_id_column: Optional[str] = None,
                                 chunk_size: int = 1000,
                                 mapping: Optional[Dict] = None,
                                 delimiter: str = ',',
                                 encoding: str = 'utf-8',
                                 header: Union[int, List[int], str] = 0,
                                 skiprows: Optional[Union[int, List[int]]] = None,
                                 nrows: Optional[int] = None,
                                 usecols: Optional[List[str]] = None,
                                 dtype: Optional[Dict[str, str]] = None,
                                 parse_dates: Optional[Union[bool, List[str]]] = None,
                                 na_values: Optional[List[str]] = None,
                                 replace_dots: bool = False,
                                 column_names: Optional[List[str]] = None) -> Dict:
        """
        Complete workflow to import CSV file to OpenSearch.

        Args:
            csv_file: Path to CSV file
            index_name: OpenSearch index name
            doc_id_column: Column to use as document ID
            chunk_size: Bulk indexing chunk size
            mapping: Optional index mapping
            delimiter: CSV field delimiter
            encoding: File encoding
            header: Row number(s) to use as column names
            skiprows: Line numbers to skip
            nrows: Number of rows to read
            usecols: List of columns to read
            dtype: Dict of column -> type mappings
            parse_dates: List of columns to parse as dates
            na_values: Additional values to recognize as NA/NaN
            replace_dots: Replace dots in column names with underscores
            column_names: Custom column names to use instead of header row

        Returns:
            Import statistics
        """
        try:
            # Read CSV file
            df = self.read_csv_file(
                file_path=csv_file,
                delimiter=delimiter,
                encoding=encoding,
                header=header,
                skiprows=skiprows,
                nrows=nrows,
                usecols=usecols,
                dtype=dtype,
                parse_dates=parse_dates,
                na_values=na_values,
                replace_dots=replace_dots,
                column_names=column_names
            )

            # Create index if needed
            self.create_index_if_not_exists(index_name, mapping)

            # Prepare documents
            documents = self.prepare_documents(df, index_name, doc_id_column)

            # Bulk index documents
            stats = self.bulk_index_documents(documents, chunk_size)

            self.logger.info(f"Import completed successfully for {csv_file}")
            return stats

        except Exception as e:
            self.logger.error(f"Import failed for {csv_file}: {e}")
            raise

    def get_index_info(self, index_name: str) -> Dict:
        """
        Get information about an index.

        Args:
            index_name: Name of the index

        Returns:
            Dictionary with index information
        """
        try:
            # Get index stats
            stats = self.client.indices.stats(index=index_name)

            # Get document count
            count_response = self.client.count(index=index_name)

            info = {
                'index_name': index_name,
                'document_count': count_response['count'],
                'size_in_bytes': stats['indices'][index_name]['total']['store']['size_in_bytes'],
                'status': 'exists'
            }

            return info

        except Exception as e:
            self.logger.error(f"Error getting index info for {index_name}: {e}")
            return {'index_name': index_name, 'status': 'error', 'error': str(e)}


def create_sample_mapping() -> Dict:
    """
    Create a sample mapping for common data types.
    Modify this based on your CSV file structure.
    """
    return {
        'properties': {
            'timestamp': {
                'type': 'date',
                'format': 'strict_date_optional_time||epoch_millis'
            },
            'id': {
                'type': 'keyword'
            },
            'text_field': {
                'type': 'text',
                'analyzer': 'standard'
            },
            'numeric_field': {
                'type': 'double'
            },
            'boolean_field': {
                'type': 'boolean'
            }
        }
    }


def main():
    """
    Main function to handle command line arguments and run the import.
    """
    parser = argparse.ArgumentParser(description='Import CSV files to OpenSearch')

    # Required arguments
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('index_name', help='OpenSearch index name')

    # OpenSearch connection arguments
    parser.add_argument('--host', default='localhost', help='OpenSearch host (default: localhost)')
    parser.add_argument('--port', type=int, default=9200, help='OpenSearch port (default: 9200)')
    parser.add_argument('--username', help='Username for authentication')
    parser.add_argument('--password', help='Password for authentication')
    parser.add_argument('--use-ssl', action='store_true', help='Use SSL connection')
    parser.add_argument('--verify-certs', action='store_true', help='Verify SSL certificates')
    parser.add_argument('--ca-certs', help='Path to CA certificates')

    # CSV parsing options
    parser.add_argument('--delimiter', default=',', help='CSV field delimiter (default: comma)')
    parser.add_argument('--encoding', default='utf-8', help='File encoding (default: utf-8)')
    parser.add_argument('--header', default='0', help='Header row number or "none" (default: 0)')
    parser.add_argument('--skiprows', type=int, help='Number of rows to skip at start')
    parser.add_argument('--nrows', type=int, help='Number of rows to read')
    parser.add_argument('--columns', help='Comma-separated list of columns to read')
    parser.add_argument('--parse-dates', help='Comma-separated list of date columns')
    parser.add_argument('--na-values', help='Comma-separated list of NA values')
    parser.add_argument('--replace-dots', action='store_true',
                        help='Replace dots in column names with underscores (for OpenSearch compatibility)')
    parser.add_argument('--column-names', help='Comma-separated list of custom column names')

    # Import options
    parser.add_argument('--doc-id-column', help='Column to use as document ID')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Bulk indexing chunk size (default: 1000)')
    parser.add_argument('--use-sample-mapping', action='store_true', help='Use sample mapping')

    # Info options
    parser.add_argument('--info', action='store_true', help='Show index info after import')

    args = parser.parse_args()

    try:
        # Initialize importer
        importer = CSVToOpenSearchImporter(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            use_ssl=args.use_ssl,
            verify_certs=args.verify_certs,
            ca_certs_path=args.ca_certs
        )

        # Prepare mapping if requested
        mapping = create_sample_mapping() if args.use_sample_mapping else None

        # Parse header argument
        header = None if args.header == 'none' else int(args.header)

        # Parse columns
        usecols = args.columns.split(',') if args.columns else None

        # Parse date columns
        parse_dates = args.parse_dates.split(',') if args.parse_dates else None

        # Parse NA values
        na_values = args.na_values.split(',') if args.na_values else None

        # Parse custom column names
        column_names = args.column_names.split(',') if args.column_names else None

        # Import data
        stats = importer.import_csv_to_opensearch(
            csv_file=args.csv_file,
            index_name=args.index_name,
            doc_id_column=args.doc_id_column,
            chunk_size=args.chunk_size,
            mapping=mapping,
            delimiter=args.delimiter,
            encoding=args.encoding,
            header=header,
            skiprows=args.skiprows,
            nrows=args.nrows,
            usecols=usecols,
            parse_dates=parse_dates,
            na_values=na_values,
            replace_dots=args.replace_dots,
            column_names=column_names
        )

        print("\n=== Import Statistics ===")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Successfully indexed: {stats['successful']}")
        print(f"Failed to index: {stats['failed']}")

        if stats['errors']:
            print(f"Errors: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")

        # Show index info if requested
        if args.info:
            print("\n=== Index Information ===")
            info = importer.get_index_info(args.index_name)
            for key, value in info.items():
                print(f"{key}: {value}")

        print(f"\nImport completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
