# Parquet to OpenSearch Importer

一个用于读取 Parquet 文件并将数据导入 OpenSearch 的 Python 程序。

## 功能特性

- **高效读取**: 支持读取各种 Parquet 文件格式
- **批量导入**: 使用 OpenSearch 批量 API 进行高效数据导入
- **连接配置**: 支持多种 OpenSearch 连接选项（SSL、身份验证等）
- **数据映射**: 支持自定义字段映射和数据类型
- **错误处理**: 完善的错误处理和日志记录
- **灵活配置**: 支持命令行参数和程序化配置

## 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖

- `pandas`: 用于数据处理
- `opensearch-py`: OpenSearch Python 客户端
- `pyarrow`: Parquet 文件读取支持

## 快速开始

### 1. 命令行使用

```bash
# 基本用法
python parquet_to_opensearch.py data.parquet my_index

# 带认证的远程 OpenSearch
python parquet_to_opensearch.py data.parquet my_index \
    --host your-opensearch-host.com \
    --port 443 \
    --username admin \
    --password your_password \
    --use-ssl

# 使用自定义配置
python parquet_to_opensearch.py data.parquet my_index \
    --doc-id-column id \
    --chunk-size 500 \
    --use-sample-mapping \
    --info
```

### 2. 程序化使用

```python
from parquet_to_opensearch import ParquetToOpenSearchImporter

# 初始化导入器
importer = ParquetToOpenSearchImporter(
    host='localhost',
    port=9200,
    use_ssl=False
)

# 导入数据
stats = importer.import_parquet_to_opensearch(
    parquet_file='data.parquet',
    index_name='my_index',
    doc_id_column='id',
    chunk_size=1000
)

print(f"成功导入 {stats['successful']} 条记录")
```

## 配置选项

### OpenSearch 连接配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | str | localhost | OpenSearch 主机地址 |
| `port` | int | 9200 | OpenSearch 端口 |
| `username` | str | None | 认证用户名 |
| `password` | str | None | 认证密码 |
| `use_ssl` | bool | False | 是否使用 SSL |
| `verify_certs` | bool | False | 是否验证证书 |
| `ca_certs_path` | str | None | CA 证书路径 |

### 导入配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `doc_id_column` | str | None | 用作文档 ID 的列名 |
| `chunk_size` | int | 1000 | 批量导入的批次大小 |
| `mapping` | dict | None | 自定义字段映射 |

## 高级用法

### 1. 自定义字段映射

```python
# 定义字段映射
custom_mapping = {
    'properties': {
        'id': {'type': 'integer'},
        'name': {'type': 'text', 'analyzer': 'standard'},
        'timestamp': {'type': 'date'},
        'value': {'type': 'double'},
        'category': {'type': 'keyword'}
    }
}

# 使用自定义映射导入
stats = importer.import_parquet_to_opensearch(
    parquet_file='data.parquet',
    index_name='my_index',
    mapping=custom_mapping
)
```

### 2. 批量处理多个文件

```python
files = ['file1.parquet', 'file2.parquet', 'file3.parquet']

for file in files:
    stats = importer.import_parquet_to_opensearch(
        parquet_file=file,
        index_name='combined_index'
    )
    print(f"{file}: {stats['successful']} 条记录已导入")
```

### 3. 数据预览和检查

```python
import pandas as pd

# 读取并检查 parquet 文件
df = pd.read_parquet('data.parquet')
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print(f"数据类型:\n{df.dtypes}")
print(f"缺失值:\n{df.isnull().sum()}")
```

## 示例文件

运行 `example_usage.py` 查看完整的使用示例:

```bash
python example_usage.py
```

这个脚本包含以下示例:
- 基本导入示例
- 带认证的导入
- 批量文件导入
- 数据检查和预览
- 使用现有 parquet 文件

## 错误处理

程序包含完善的错误处理机制:

- **连接错误**: 自动检测 OpenSearch 连接问题
- **文件错误**: 处理 parquet 文件读取异常
- **导入错误**: 记录批量导入过程中的错误
- **数据类型**: 自动处理 pandas 数据类型转换

## 性能优化

### 1. 调整批次大小

根据数据大小和 OpenSearch 性能调整 `chunk_size`:
- 小数据集: 500-1000
- 大数据集: 1000-5000
- 高性能集群: 5000-10000

### 2. 并行处理

程序使用 `parallel_bulk` 进行并行导入，默认 4 个线程。

### 3. 索引设置

对于大量数据导入，建议临时调整索引设置:

```python
# 导入前设置
index_config = {
    'settings': {
        'number_of_replicas': 0,
        'refresh_interval': '30s',
        'merge.policy.max_merge_at_once': 30,
        'merge.policy.segments_per_tier': 30
    }
}
```

## 故障排除

### 常见问题

1. **连接失败**
   ```
   ConnectionError: Unable to connect to OpenSearch
   ```
   检查 OpenSearch 是否运行，端口是否正确。

2. **认证失败**
   ```
   AuthenticationException: Unauthorized
   ```
   检查用户名和密码是否正确。

3. **内存不足**
   ```
   MemoryError: Unable to allocate array
   ```
   减少 `chunk_size` 或增加系统内存。

4. **字段类型错误**
   ```
   RequestError: mapper_parsing_exception
   ```
   检查数据类型，使用自定义映射。

### 日志配置

程序使用 Python logging 模块，可以调整日志级别:

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # 详细日志
logging.basicConfig(level=logging.WARNING)  # 仅警告和错误
```

## 项目结构

```
Code/Python/
├── parquet_to_opensearch.py  # 主程序
├── example_usage.py          # 使用示例
├── requirements.txt          # 依赖列表
└── README.md                 # 文档
```

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个工具。

## 许可证

MIT License

---

## English Version

# Parquet to OpenSearch Importer

A Python program for reading Parquet files and importing data into OpenSearch.

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Basic usage
python parquet_to_opensearch.py data.parquet index_name

# With authentication
python parquet_to_opensearch.py data.parquet index_name \
    --host your-host --username admin --password pwd --use-ssl
```

### Programmatic Usage

```python
from parquet_to_opensearch import ParquetToOpenSearchImporter

importer = ParquetToOpenSearchImporter(host='localhost', port=9200)
stats = importer.import_parquet_to_opensearch('data.parquet', 'my_index')
```

See `example_usage.py` for complete examples and `parquet_to_opensearch.py` for full documentation.
