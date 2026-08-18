"""Materialize the proposal's 17K training budget inside CSPD."""

from pathlib import Path

import pyarrow.parquet as pq

SOURCE = Path("/WORK/PUBLIC/alex_work/Meiqi.Gu/SDPO/datasets/dapo_hf/processed/dapo-math-17k.parquet")
DEST = Path("/home/fit/alex1/WORK/Meiqi.Gu/CSPD/data/dapo-math-17k-seed42.parquet")
N = 17_000

DEST.parent.mkdir(parents=True, exist_ok=True)
table = pq.read_table(SOURCE).slice(0, N)
if table.num_rows != N:
    raise RuntimeError(f"expected {N} rows, found {table.num_rows}")
pq.write_table(table, DEST, compression="zstd")
print(f"source={SOURCE}\nrows={table.num_rows}\ndestination={DEST}")
