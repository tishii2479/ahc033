import subprocess
import sys

import pandas as pd

if __name__ == "__main__":
    seed = int(sys.argv[1])
    file = f"{seed:04}"

    proc = subprocess.run("cargo build --features local --release", shell=True)
    if proc.returncode != 0:
        print(f"failed to build: {proc}")
        exit(1)

    subprocess.run(
        "./target/release/solver" + f"< tools/in/{file}.txt > out/{file}.txt",
        shell=True,
    )
    subprocess.run(
        "./tools/target/release/vis" + f" tools/in/{file}.txt out/{file}.txt",
        shell=True,
    )
    subprocess.run(f"pbcopy < out/{file}.txt", shell=True)

    # 過去ログとの比較
    # df = pd.read_csv("./log/database.csv")
    # print(
    #     df[(df.input_file == f"tools/in/{file}.txt")][
    #         ["solver_version", "score"]
    #     ].sort_values(by="score")[:20]
    # )
