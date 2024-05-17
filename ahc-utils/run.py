import abc
import argparse
import dataclasses
import datetime
import json
import logging
import os
import subprocess
from logging import FileHandler, StreamHandler, getLogger
from typing import List, Optional, Type

import pandas as pd
from joblib import Parallel, delayed


@dataclasses.dataclass
class IResult:
    @abc.abstractmethod
    def __init__(self, stderr: str, input_file: str, solver_version: str) -> None:
        raise NotImplementedError()


class Col:
    INPUT_FILE: str = "input_file"
    SOLVER_VERSION: str = "solver_version"
    SCORE: str = "score"
    RELATIVE_SCORE: str = "relative_score"


class Runner:
    def __init__(
        self,
        result_class: Type[IResult],
        solver_cmd: str,
        solver_version: str,
        database_csv: str,
        log_file: str,
        is_maximize: bool = True,
        input_csv: Optional[str] = None,
        verbose: int = 10,
    ) -> None:
        self.result_class = result_class
        self.solver_cmd = solver_cmd
        self.solver_version = solver_version
        self.database_csv = database_csv
        self.input_csv = input_csv
        self.logger = self.setup_logger(log_file=log_file, verbose=verbose)
        self.is_maximize = is_maximize

    def run_case(self, input_file: str, output_file: str) -> IResult:
        cmd = f"{self.solver_cmd} < {input_file} > {output_file}"
        proc = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
        stderr = proc.stderr.decode("utf-8")
        result = self.result_class(stderr, input_file, self.solver_version)
        return result

    def run(
        self,
        cases: list[tuple[str, str]],
        n_jobs: int = -1,
        verbose: int = 10,
        ignore: bool = False,
    ) -> pd.DataFrame:
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self.run_case)(input_file, output_file)
            for input_file, output_file in cases
        )
        df = pd.DataFrame(list(map(lambda x: vars(x), results)))
        if not ignore:
            add_header = not os.path.exists(self.database_csv)
            df.to_csv(self.database_csv, mode="a", index=False, header=add_header)

        return df

    def evalute_relative_score(
        self,
        k: int = 10,
        bins: int = 6,
        benchmark_solver_version: Optional[str] = None,
        columns: Optional[List[str]] = None,
        eval_items: List[str] = [Col.SCORE, Col.RELATIVE_SCORE],
        ignore_solver_prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        self.logger.info(f"Evaluate: {self.solver_version}")
        df = self.read_database_df(ignore_solver_prefix=ignore_solver_prefix)
        if benchmark_solver_version is not None:
            df = pd.merge(
                df[df.solver_version == self.solver_version],
                df[df.solver_version == benchmark_solver_version][
                    [Col.INPUT_FILE, Col.SCORE, Col.RELATIVE_SCORE]
                ],
                how="left",
                on=Col.INPUT_FILE,
                suffixes=["", "_bench"],
            )
            df[Col.RELATIVE_SCORE] = (
                df[Col.RELATIVE_SCORE] - df[f"{Col.RELATIVE_SCORE}_bench"]
            )
            df = df.drop(f"{Col.RELATIVE_SCORE}_bench", axis=1)
            df[Col.SCORE] = df[Col.SCORE] - df[f"{Col.SCORE}_bench"]
            df = df.drop(f"{Col.SCORE}_bench", axis=1)

        df = df[df.solver_version == self.solver_version]
        self.logger.info(f"Raw score mean: {df.score.mean()}")
        self.logger.info(f"Relative score mean: {df[Col.RELATIVE_SCORE].mean()}")
        self.logger.info(f"Relative score median: {df[Col.RELATIVE_SCORE].median()}")
        self.logger.info(f"Top {k} improvements:")
        self.logger.info(
            df.sort_values(by=Col.RELATIVE_SCORE, ascending=self.is_maximize)[:k]
        )
        self.logger.info(f"Top {k} aggravations:")
        self.logger.info(
            df.sort_values(by=Col.RELATIVE_SCORE, ascending=not self.is_maximize)[:k]
        )

        if columns is not None:
            assert 1 <= len(columns) <= 2
            cut_columns = list(map(lambda col: f"{col}_cut", columns))
            for cut_column, column in zip(cut_columns, columns):
                df[cut_column] = pd.cut(df[column], bins=bins)
            if len(cut_columns) == 1:
                self.logger.info(df.groupby(cut_columns[0])[Col.RELATIVE_SCORE].mean())
            elif len(cut_columns) == 2:
                self.logger.info(
                    df[eval_items + cut_columns].pivot_table(
                        index=cut_columns[0], columns=cut_columns[1]
                    )
                )

        return df

    def list_solvers(
        self, ignore_solver_prefix: Optional[str] = None, top_k: int = 50
    ) -> pd.DataFrame:
        database_df = self.read_database_df(ignore_solver_prefix=ignore_solver_prefix)
        database_df = (
            database_df.groupby(Col.SOLVER_VERSION)[[Col.RELATIVE_SCORE, Col.SCORE]]
            .mean()
            .sort_values(by=Col.RELATIVE_SCORE, ascending=self.is_maximize)
        )
        self.logger.info(database_df[:top_k])

        return database_df

    def setup_logger(self, log_file: str, verbose: int) -> logging.Logger:
        logger = getLogger(__name__)
        logger.setLevel(logging.INFO)

        if verbose > 0:
            stream_handler = StreamHandler()
            stream_handler.setLevel(logging.DEBUG)
            logger.addHandler(stream_handler)

        file_handler = FileHandler(log_file, "a")
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        return logger

    def read_database_df(
        self, ignore_solver_prefix: Optional[str] = None
    ) -> pd.DataFrame:
        COL_BEST_SCORE = "best_score"
        df = pd.read_csv(self.database_csv)
        if self.input_csv is not None:
            df = pd.merge(
                df, pd.read_csv(self.input_csv), how="left", on=Col.INPUT_FILE
            )
        best_scores = (
            df.groupby(Col.INPUT_FILE)[Col.SCORE]
            .agg("max" if self.is_maximize else "min")
            .rename(COL_BEST_SCORE)
        )
        df = pd.merge(df, best_scores, on=Col.INPUT_FILE, how="left")
        if self.is_maximize:
            df[Col.RELATIVE_SCORE] = df[Col.SCORE] / df[COL_BEST_SCORE]
        else:
            df[Col.RELATIVE_SCORE] = df[COL_BEST_SCORE] / df[Col.SCORE]
        if ignore_solver_prefix is not None:
            df = df[
                (~df.solver_version.str.startswith(ignore_solver_prefix))
                | (df.solver_version == self.solver_version)
            ]
        df = df[
            (
                ~df.solver_version.str.startswith("optuna-")
                & ~df.solver_version.str.startswith("solver-")
            )
            | (df.solver_version == self.solver_version)
        ]
        return df


@dataclasses.dataclass
class Result(IResult):
    input_file: str
    solver_version: str
    score: int
    duration: float

    def __init__(self, stderr: str, input_file: str, solver_version: str):
        self.input_file = input_file
        self.solver_version = solver_version

        json_start = stderr.find("result:") + len("result:")
        result_str = stderr[json_start:]
        try:
            result_json = json.loads(result_str)
            self.score = result_json[Col.SCORE]
            self.duration = result_json["duration"]
        except json.JSONDecodeError as e:
            print(e)
            print(f"failed to parse result_str: {result_str}, input_file: {input_file}")


def parse_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="local")
    parser.add_argument("-d", "--data-dir", type=str, default="tools")
    parser.add_argument("-e", "--eval", action="store_true")
    parser.add_argument("--no-eval", action="store_false")
    parser.add_argument("-l", "--list-solver", action="store_true")
    parser.add_argument("-i", "--ignore", action="store_true")
    parser.add_argument("-n", "--case_num", type=int, default=100)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("-v", "--verbose", type=int, default=10)
    parser.add_argument(
        "-s",
        "--solver-path",
        type=str,
        default="./target/release/solver",
    )
    parser.add_argument(
        "-a",
        "--solver-version",
        type=str,
        default=f"solver-{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}",
    )
    parser.add_argument("-b", "--benchmark-solver-version", type=str, default=None)
    parser.add_argument("--database-csv", type=str, default="log/database.csv")
    parser.add_argument("--input-csv", type=str, default="log/input.csv")
    parser.add_argument("--log-file", type=str, default="log/a.log")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_config()
    runner = Runner(
        Result,
        solver_cmd=args.solver_path,
        solver_version=args.solver_version,
        database_csv=args.database_csv,
        input_csv=args.input_csv,
        log_file=args.log_file,
        is_maximize=False,
    )
    columns = ["n", "d"]

    if args.list_solver:
        runner.list_solvers()
    elif args.eval:
        runner.evalute_relative_score(
            benchmark_solver_version=args.benchmark_solver_version, columns=columns
        )
    else:
        subprocess.run(f"cargo build --features {args.mode} --release", shell=True)
        subprocess.run(
            f"python3 ahc-utils/expander.py > log/backup/{args.solver_version}.rs",
            shell=True,
        )
        cases = [
            (f"{args.data_dir}/in/{seed:04}.txt", f"out/{seed:04}.txt")
            for seed in range(args.start_case, args.start_case + args.case_num)
        ]
        runner.run(
            cases=cases, n_jobs=args.n_jobs, ignore=args.ignore, verbose=args.verbose
        )
        if not args.no_eval:
            runner.evalute_relative_score(
                benchmark_solver_version=args.benchmark_solver_version, columns=columns
            )
