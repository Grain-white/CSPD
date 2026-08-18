import sys

import pyarrow.parquet as pq

for path in sys.argv[1:]:
    metadata = pq.ParquetFile(path).metadata
    print(path, "rows=", metadata.num_rows, "groups=", metadata.num_row_groups)
