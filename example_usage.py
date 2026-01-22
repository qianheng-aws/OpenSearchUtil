#!/usr/bin/env python3
"""
Example usage of the Parquet to OpenSearch Importer

This script demonstrates how to use the ParquetToOpenSearchImporter class
to import parquet files into OpenSearch.
"""

import os
import pandas as pd
from parquet_to_opensearch import ParquetToOpenSearchImporter


def create_sample_parquet_file():
    """
    Create a sample parquet file for testing purposes.
    """
    # Sample data
    data = {
        'id': range(1, 101),
        'name': [f'item_{i}' for i in range(1, 101)],
        'value': [i * 10.5 for i in range(1, 101)],
        'category': ['A', 'B', 'C'] * 33 + ['A'],
        'active': [True if i % 2 == 0 else False for i in range(1, 101)],
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
    }
    
    df = pd.DataFrame(data)
    
    # Save as parquet file
    sample_file = 'sample_data.parquet'
    df.to_parquet(sample_file, index=False)
    print(f"Created sample parquet file: {sample_file}")
    return sample_file


def example_basic_import():
    """
    Example 1: Basic import without authentication
    """
    print("\n=== Example 1: Basic Import ===")
    
    # Create sample data
    sample_file = create_sample_parquet_file()
    
    try:
        # Initialize importer (assumes OpenSearch running on localhost:9200)
        importer = ParquetToOpenSearchImporter(
            host='localhost',
            port=9200,
            use_ssl=False
        )
        
        # Import the parquet file
        stats = importer.import_parquet_to_opensearch(
            parquet_file=sample_file,
            index_name='sample_data_index',
            doc_id_column='id',  # Use 'id' column as document ID
            chunk_size=50
        )
        
        print("Import Statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Successfully indexed: {stats['successful']}")
        print(f"  Failed to index: {stats['failed']}")
        
        # Get index information
        info = importer.get_index_info('sample_data_index')
        print("\nIndex Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)


def example_with_authentication():
    """
    Example 2: Import with authentication and SSL
    """
    print("\n=== Example 2: Import with Authentication ===")
    
    # Create sample data
    sample_file = create_sample_parquet_file()
    
    try:
        # Initialize importer with authentication
        importer = ParquetToOpenSearchImporter(
            host='your-opensearch-host.com',
            port=443,
            username='your_username',
            password='your_password',
            use_ssl=True,
            verify_certs=True
        )
        
        # Define custom mapping
        custom_mapping = {
            'properties': {
                'id': {'type': 'integer'},
                'name': {'type': 'text', 'analyzer': 'standard'},
                'value': {'type': 'double'},
                'category': {'type': 'keyword'},
                'active': {'type': 'boolean'},
                'timestamp': {'type': 'date'}
            }
        }
        
        # Import with custom mapping
        stats = importer.import_parquet_to_opensearch(
            parquet_file=sample_file,
            index_name='sample_data_with_auth',
            mapping=custom_mapping,
            chunk_size=100
        )
        
        print("Import completed with authentication")
        
    except Exception as e:
        print(f"Note: This example requires valid OpenSearch credentials: {e}")
    finally:
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)


def example_batch_import():
    """
    Example 3: Import multiple parquet files
    """
    print("\n=== Example 3: Batch Import ===")
    
    # Create multiple sample files
    files_to_import = []
    
    for i in range(3):
        # Create different sample data for each file
        data = {
            'id': range(i*100 + 1, (i+1)*100 + 1),
            'batch': [f'batch_{i}'] * 100,
            'value': [j * (i+1) for j in range(1, 101)],
            'timestamp': pd.date_range(f'2024-{i+1:02d}-01', periods=100, freq='H')
        }
        
        df = pd.DataFrame(data)
        filename = f'batch_{i}.parquet'
        df.to_parquet(filename, index=False)
        files_to_import.append(filename)
    
    try:
        # Initialize importer
        importer = ParquetToOpenSearchImporter()
        
        # Import each file to the same index
        total_stats = {'total_documents': 0, 'successful': 0, 'failed': 0}
        
        for file in files_to_import:
            print(f"Importing {file}...")
            stats = importer.import_parquet_to_opensearch(
                parquet_file=file,
                index_name='batch_import_index'
            )
            
            # Aggregate statistics
            total_stats['total_documents'] += stats['total_documents']
            total_stats['successful'] += stats['successful']
            total_stats['failed'] += stats['failed']
        
        print("\nTotal Import Statistics:")
        print(f"  Total documents: {total_stats['total_documents']}")
        print(f"  Successfully indexed: {total_stats['successful']}")
        print(f"  Failed to index: {total_stats['failed']}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        for file in files_to_import:
            if os.path.exists(file):
                os.remove(file)


def example_inspect_parquet():
    """
    Example 4: Inspect parquet file before import
    """
    print("\n=== Example 4: Inspect Parquet File ===")
    
    # Create sample data with different data types
    data = {
        'string_col': ['hello', 'world', 'test', None, 'data'],
        'int_col': [1, 2, 3, None, 5],
        'float_col': [1.1, 2.2, None, 4.4, 5.5],
        'bool_col': [True, False, True, None, False],
        'date_col': pd.date_range('2024-01-01', periods=5),
        'list_col': [['a', 'b'], ['c'], None, ['d', 'e', 'f'], ['g']]
    }
    
    df = pd.DataFrame(data)
    sample_file = 'inspect_sample.parquet'
    df.to_parquet(sample_file, index=False)
    
    try:
        # Read and inspect the parquet file
        print(f"Inspecting parquet file: {sample_file}")
        
        df_read = pd.read_parquet(sample_file)
        print(f"\nShape: {df_read.shape}")
        print(f"Columns: {list(df_read.columns)}")
        print("\nData types:")
        print(df_read.dtypes)
        print("\nFirst few rows:")
        print(df_read.head())
        print("\nNull values:")
        print(df_read.isnull().sum())
        
        # Initialize importer and import
        importer = ParquetToOpenSearchImporter()
        
        stats = importer.import_parquet_to_opensearch(
            parquet_file=sample_file,
            index_name='inspect_example_index'
        )
        
        print(f"\nImported {stats['successful']} documents successfully")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)


def main():
    """
    Run all examples
    """
    print("Parquet to OpenSearch Importer - Example Usage")
    print("=" * 50)
    
    # Check if we can use the existing parquet file
    existing_parquet = "/Volumes/workplace/community/neural-search/christmas/output/entities.parquet"
    if os.path.exists(existing_parquet):
        print(f"\n=== Example: Using Existing Parquet File ===")
        try:
            # Try to inspect the existing file
            df = pd.read_parquet(existing_parquet)
            print(f"Found existing parquet file with {len(df)} records")
            print(f"Columns: {list(df.columns)}")
            print(f"Shape: {df.shape}")
            print("\nFirst few rows:")
            print(df.head())
            
            # Import it to OpenSearch
            importer = ParquetToOpenSearchImporter()
            stats = importer.import_parquet_to_opensearch(
                parquet_file=existing_parquet,
                index_name='test.entities'
            )
            
            print(f"\nImported {stats['successful']} documents to 'parquetfilter_data' index")
            
        except Exception as e:
            print(f"Error with existing parquet file: {e}")
    
    # Run example scenarios
    # example_basic_import()
    # example_with_authentication()  # Will show error without real credentials
    # example_batch_import()
    # example_inspect_parquet()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use with your own data:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Start OpenSearch on localhost:9200")
    print("3. Run: python parquet_to_opensearch.py your_file.parquet your_index_name")


if __name__ == '__main__':
    main()
