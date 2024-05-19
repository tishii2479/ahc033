mod def;
mod solver;
mod util;

use itertools::iproduct;
use std::collections::VecDeque;

use proconio::input;

use crate::def::*;
use crate::solver::*;
use crate::util::*;

/*
struct State {
    terminal: Vec<Vec<Option<usize>>>,
    crane_p: Vec<(usize, usize)>,
    crane_status: Vec<CraneStatus>,
    next_cont: Vec<usize>,
}
*/

fn output_ans(moves: &Vec<Vec<Move>>) {
    let mut score = 0;
    for i in 0..N {
        let mut s = String::new();
        for m in moves[i].iter() {
            s += m.to_str();
        }
        println!("{s}");
        score = score.max(moves[i].len());
    }

    eprintln!(
        "result: {{\"score\": {}, \"duration\": {:.4}}}",
        score,
        time::elapsed_seconds(),
    );
}

fn main() {
    time::start_clock();
    input! {
        _: usize,
        a: [[usize; N]; N],
    }

    // in_order := 使用する搬入口の順序
    // 搬入順序の最適化
    let in_order = optimize_in_order(&a);

    // in_place[c] := コンテナcを退避させる場所
    // 搬入場所の最適化
    let in_place = optimize_in_place(&in_order, &a);

    // クレーンの操作
    let moves = solve(&in_order, &in_place, &a);

    output_ans(&moves);
}

#[derive(Clone, Copy, Debug)]
struct Task {
    from: (usize, usize),
    to: (usize, usize),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ContainerStatus {
    Await,    // ターミナル内にあるがアサインされていない
    Assigned, // アサインされた
    Out,      // ターミナルの外（未搬入か搬出後のいずれか）
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CraneStatus {
    Idle,    // タスクがアサインされていない
    Moving,  // コンテナを吊るさずに目的地に移動している
    Hanging, // コンテナを吊るしてに目的地に移動している
    Blown,   // 爆破した
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TileStatus {
    Empty,
    Crane(usize),
    Container,
}

#[derive(Clone, Copy)]
struct Container {
    status: ContainerStatus,
    pos: Option<(usize, usize)>,
}

#[derive(Clone)]
struct Crane {
    status: CraneStatus,
    task: Option<Task>,
    pos: (usize, usize),
    moves: Vec<Move>,
}

fn add_moves(
    terminal: &mut Vec<Vec<Vec<TileStatus>>>,
    crane: &mut Crane,
    moves: Vec<Move>,
    cur_pos: (usize, usize),
    crane_i: usize,
) {
    let (mut pi, mut pj) = cur_pos;
    for dt in 0..moves.len() {
        let t = crane.moves.len() + dt;
        let (di, dj) = moves[dt].to_d();
        let (ni, nj) = (pi + di, pj + dj);
        assert!(!matches!(terminal[t + 1][ni][nj], TileStatus::Crane(_)));
        terminal[t + 1][ni][nj] = TileStatus::Crane(crane_i);
        (pi, pj) = (ni, nj);
    }
    crane.moves.extend(moves);
}

fn get_moves_for_task(
    terminal: &Vec<Vec<Vec<TileStatus>>>,
    t: usize,
    cur_pos: (usize, usize),
    task: Task,
    is_crane_zero: bool,
) -> Vec<Move> {
    let mut moves = vec![];
    moves.extend(listup_valid_moves(terminal, t, cur_pos, task.from, true));
    moves.push(Move::Pick);
    moves.extend(listup_valid_moves(
        terminal,
        t + moves.len(),
        task.from,
        task.to,
        is_crane_zero,
    ));
    moves.push(Move::Drop);
    moves
}

fn restore_path(
    end_t: usize,
    from: (usize, usize),
    to: (usize, usize),
    dp: &Vec<Vec<Vec<bool>>>,
) -> Vec<Move> {
    // toに到達したので、fromに戻るまでの経路を復元する
    let mut moves = vec![];
    let mut cur = to;
    let mut cur_t = end_t;
    while cur != from {
        cur_t -= 1;
        for (i, &(di, dj)) in REV_D.iter().enumerate() {
            let (ni, nj) = (cur.0 + di, cur.1 + dj);
            if ni >= N || nj >= N {
                continue;
            }
            if dp[cur_t][ni][nj] {
                moves.push(D_MOVE[i]);
                cur = (ni, nj);
                break;
            }
        }
    }

    moves.reverse();
    return moves;
}

/// from->toの経路を探索する
/// 最後に拾う・落とすなどの操作をするため、時刻t+1に留まることができるか調べる必要がある
/// over_container := コンテナの上を移動できるかどうか
fn listup_valid_moves(
    terminal: &Vec<Vec<Vec<TileStatus>>>,
    start_t: usize,
    from: (usize, usize),
    to: (usize, usize),
    over_container: bool,
) -> Vec<Move> {
    if from == to && !matches!(terminal[start_t + 1][from.0][from.1], TileStatus::Crane(_)) {
        return vec![];
    }
    let mut dp = vec![vec![vec![false; N]; N]; T];
    dp[start_t][from.0][from.1] = true;
    for t in start_t.. {
        for (i, j) in iproduct!(0..N, 0..N) {
            if !dp[t][i][j] {
                continue;
            }
            for (di, dj) in D {
                let (ni, nj) = (i + di, j + dj);
                if ni >= N || nj >= N {
                    continue;
                }

                // 移動できる条件
                // 1. terminal[t + 1][ni][nj] == TileStatus::Empty || (terminal[t + 1][ni][nj] == TileStatus::Container && over_container)
                // 2. ! (terminal[t][ni][nj] == TileStatus::Crane(c) && terminal[t + 1][i][j] == TileStatus::Crane(c))
                // 3. j == 0
                if (terminal[t + 1][ni][nj] == TileStatus::Empty
                    || (terminal[t + 1][ni][nj] == TileStatus::Container && over_container))
                    && !(matches!(terminal[t][ni][nj], TileStatus::Crane(_))
                        && terminal[t][ni][nj] == terminal[t + 1][i][j])
                {
                    dp[t + 1][ni][nj] = true;
                    // 最後に拾う・落とすなどの操作をするため、時刻t+2に留まることができるか調べる必要がある
                    if (ni, nj) == to && !matches!(terminal[t + 2][ni][nj], TileStatus::Crane(_)) {
                        let moves = restore_path(t + 1, from, to, &dp);
                        assert_eq!(moves.len(), t + 1 - start_t);
                        return moves;
                    }
                }
            }
        }
    }

    panic!();
}

/// 搬入位置の最適化
/// NOTE: 搬入順序と一緒に最適化できそう
/// 評価関数
/// - 各時刻に仮置きしているコンテナのうち、到達不可能なコンテナの個数の総和
///     - 搬入口と搬出口から
/// - 搬入口から搬出口までの距離
fn optimize_in_place(in_order: &Vec<usize>, a: &Vec<Vec<usize>>) -> Vec<(usize, usize)> {
    let mut in_place = vec![(0, 0); N * N];
    for i in 0..N * N {
        in_place[i] = (rnd::gen_index(N), rnd::gen_range(1, N - 1));
    }
    let mut cur_score = eval_in_place(in_order, &in_place, a);
    for _t in 0..10000 {
        let i = rnd::gen_index(in_place.len());
        let prev_p = in_place[i];
        in_place[i] = (rnd::gen_index(N), rnd::gen_range(1, N - 1));
        let new_score = eval_in_place(in_order, &in_place, a);
        if new_score < cur_score {
            eprintln!("[{}] {} -> {}", _t, cur_score, new_score);
            cur_score = new_score;
        } else {
            in_place[i] = prev_p;
        }
    }

    in_place
}

/// 搬入順序の最適化
/// 各時刻に仮置きしているコンテナの個数の総和を最小化
fn optimize_in_order(a: &Vec<Vec<usize>>) -> Vec<usize> {
    let mut order = vec![];
    for i in 0..N {
        order.extend(vec![i; N]);
    }

    rnd::shuffle(&mut order);
    eprintln!("before:  {:?}", order);
    let mut cur_score = eval_in_order(&order, &a);
    for _t in 0..10000 {
        let (i, j) = (rnd::gen_index(order.len()), rnd::gen_index(order.len()));
        if order[i] == order[j] {
            continue;
        }
        order.swap(i, j);
        let new_score = eval_in_order(&order, &a);
        if new_score < cur_score {
            eprintln!("[{}] {} -> {}", _t, cur_score, new_score);
            cur_score = new_score;
        } else {
            order.swap(i, j);
        }
    }
    eprintln!("after:   {:?}", order);

    order
}

fn eval_in_place(
    in_order: &Vec<usize>,
    in_place: &Vec<(usize, usize)>,
    a: &Vec<Vec<usize>>,
) -> i64 {
    let mut moved = vec![false; N * N]; // moved[c] := コンテナcを搬入したかどうか
    let mut used_count = vec![vec![0; N]; N];
    let mut in_count = vec![0; N]; // in_count[i] := 搬入口iから搬入した個数
    let mut out_count = vec![0; N]; // out_count[i] := 搬出口iから搬出した個数

    let mut conflict_count = 0;

    for &i in in_order.iter() {
        let c = a[i][in_count[i]];
        let (pi, pj) = in_place[c];
        in_count[i] += 1;
        moved[c] = true;

        // TODO: 搬入できる場合は(pi, pj)を置き換える
        used_count[pi][pj] += 1;

        // (pi, pj)にコンテナが存在しないかどうか
        if used_count[pi][pj] > 1 {
            conflict_count += 1;
        }

        // TODO: (i, 0)から(pi, pj)に到達可能かどうか

        // 搬出できるコンテナを搬出する
        let (out_i, out_j) = (c / N, c % N);
        for j in out_j..N {
            let c = out_i * N + j;
            if !moved[c] || out_count[out_i] != j {
                break;
            }

            // (pi, pj) := in_place[c]
            // TODO: (pi, pj)から(out_i, 4)が到達可能かどうか
            let (pi, pj) = in_place[c];
            used_count[pi][pj] -= 1;

            out_count[out_i] += 1;
        }
    }

    conflict_count
}

fn eval_in_order(order: &Vec<usize>, a: &Vec<Vec<usize>>) -> i64 {
    let mut moved = vec![false; N * N]; // moved[c] := コンテナcを搬入したかどうか
    let mut in_count = vec![0; N]; // in_count[i] := 搬入口iから搬入した個数
    let mut out_count = vec![0; N]; // out_count[i] := 搬出口iから搬出した個数
    let mut evacuate_count = 0;
    let mut eval = 0;

    for &i in order {
        // 搬入する
        let c = a[i][in_count[i]];
        in_count[i] += 1;
        moved[c] = true;
        evacuate_count += 1;

        // 退避したコンテナのうち、搬出できるものは搬出する
        let (out_i, out_j) = (c / N, c % N);

        // 同じ搬出口のコンテナが退避されていないか調べる
        for j in out_j..N {
            let c = out_i * N + j;
            if !moved[c] || out_count[out_i] != j {
                break;
            }
            out_count[out_i] += 1;
            evacuate_count -= 1;
        }

        // 退避してるコンテナの数を数える
        eval += evacuate_count;
    }

    eval
}
