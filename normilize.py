import argparse

from polars import col, scan_ipc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input arrow file")
    parser.add_argument("--engine", type=str, default="auto", help="Engine used to process the query")
    args = parser.parse_args()

    lf = scan_ipc(args.input)

    s2 = col("strain_magnitude") / col("strain_magnitude").mean()
    o2 = col("vorticity_magnitude") / col("vorticity_magnitude").mean()
    eps = col("eps") / col("eps").mean()

    df = lf.select([
        s2.alias("s2"),
        o2.alias("o2"),
        eps.alias("eps")
    ]).collect(engine=args.engine)

    file_name = str.replace(args.input, ".arrow", "")

    df.write_ipc(f"{file_name}_normalized.arrow")
    df.write_csv(f"{file_name}_normalized.csv")
