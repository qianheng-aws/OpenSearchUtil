#!/usr/bin/env python3
"""
Parquet to OpenSearch Data Importer

This script reads parquet files and imports the data into OpenSearch using bulk indexing.
Supports various configuration options for OpenSearch connection and data processing.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import RequestError, ConnectionError


class ParquetToOpenSearchImporter:
    """
    A class to handle importing parquet file data into OpenSearch.
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
    
    def read_parquet_file(self, file_path: str) -> pd.DataFrame:
        """
        Read parquet file and return pandas DataFrame.
        
        Args:
            file_path: Path to the parquet file
            
        Returns:
            pandas DataFrame containing the data
        """
        try:
            self.logger.info(f"Reading parquet file: {file_path}")
            df = pd.read_parquet(file_path)
            self.logger.info(f"Successfully read {len(df)} records from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error reading parquet file {file_path}: {e}")
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
    
    def import_parquet_to_opensearch(self, 
                                   parquet_file: str, 
                                   index_name: str,
                                   doc_id_column: Optional[str] = None,
                                   chunk_size: int = 1000,
                                   mapping: Optional[Dict] = None) -> Dict:
        """
        Complete workflow to import parquet file to OpenSearch.
        
        Args:
            parquet_file: Path to parquet file
            index_name: OpenSearch index name
            doc_id_column: Column to use as document ID
            chunk_size: Bulk indexing chunk size
            mapping: Optional index mapping
            
        Returns:
            Import statistics
        """
        try:
            # Read parquet file
            df = self.read_parquet_file(parquet_file)
            
            # Create index if needed
            self.create_index_if_not_exists(index_name, mapping)
            
            # Prepare documents
            documents = self.prepare_documents(df, index_name, doc_id_column)
            
            # Bulk index documents
            stats = self.bulk_index_documents(documents, chunk_size)
            
            self.logger.info(f"Import completed successfully for {parquet_file}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Import failed for {parquet_file}: {e}")
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
    Modify this based on your parquet file structure.
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
    parser = argparse.ArgumentParser(description='Import parquet files to OpenSearch')
    
    # Required arguments
    parser.add_argument('parquet_file', help='Path to parquet file')
    parser.add_argument('index_name', help='OpenSearch index name')
    
    # OpenSearch connection arguments
    parser.add_argument('--host', default='localhost', help='OpenSearch host (default: localhost)')
    parser.add_argument('--port', type=int, default=9200, help='OpenSearch port (default: 9200)')
    parser.add_argument('--username', help='Username for authentication')
    parser.add_argument('--password', help='Password for authentication')
    parser.add_argument('--use-ssl', action='store_true', help='Use SSL connection')
    parser.add_argument('--verify-certs', action='store_true', help='Verify SSL certificates')
    parser.add_argument('--ca-certs', help='Path to CA certificates')
    
    # Import options
    parser.add_argument('--doc-id-column', help='Column to use as document ID')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Bulk indexing chunk size (default: 1000)')
    parser.add_argument('--use-sample-mapping', action='store_true', help='Use sample mapping')
    
    # Info options
    parser.add_argument('--info', action='store_true', help='Show index info after import')
    
    args = parser.parse_args()
    
    try:
        # Initialize importer
        importer = ParquetToOpenSearchImporter(
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
        
        # Import data
        stats = importer.import_parquet_to_opensearch(
            parquet_file=args.parquet_file,
            index_name=args.index_name,
            doc_id_column=args.doc_id_column,
            chunk_size=args.chunk_size,
            mapping=mapping
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
