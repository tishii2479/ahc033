mod def;
mod util;

use proconio::input;

use crate::def::*;
use crate::util::*;

/*
#[derive(Clone, Copy)]
enum CraneStatus {
    Idle,
    Hanging,
    Blowed,
}

#[derive(Clone, Copy)]
enum ContainerStatus {
    Await,
    Evacuated,
    Hunging,
    Complete,
}

struct State {
    terminal: Vec<Vec<Option<usize>>>,
    crane_p: Vec<(usize, usize)>,
    crane_status: Vec<CraneStatus>,
    next_cont: Vec<usize>,
}
*/

#[derive(Clone, Copy)]
enum Move {
    Pick,  // P
    Quit,  // Q
    Up,    // U
    Down,  // D
    Left,  // L
    Right, // R
    Stop,  // .
    Blow,  // B
}

impl Move {
    fn to_str(&self) -> &str {
        match self {
            Move::Pick => "P",
            Move::Quit => "Q",
            Move::Up => "U",
            Move::Down => "D",
            Move::Left => "L",
            Move::Right => "R",
            Move::Stop => "S",
            Move::Blow => "B",
        }
    }
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
    let mut moves = vec![vec![]; N];
    moves[0] = solve_crane0(&in_order, &in_place, &a);
    for i in 1..N {
        moves[i].push(Move::Blow);
    }

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

fn get_move(from: (usize, usize), to: (usize, usize)) -> Vec<Move> {
    let mut v = vec![];
    if from.0 > to.0 {
        v.extend(vec![Move::Up; from.0 - to.0]);
    } else {
        v.extend(vec![Move::Down; to.0 - from.0]);
    }
    if from.1 > to.1 {
        v.extend(vec![Move::Left; from.1 - to.1]);
    } else {
        v.extend(vec![Move::Right; to.1 - from.1]);
    }
    v
}

fn solve_crane0(
    in_order: &Vec<usize>,
    in_place: &Vec<(usize, usize)>,
    a: &Vec<Vec<usize>>,
) -> Vec<Move> {
    let mut moved = vec![false; N * N]; // moved[c] := コンテナcを搬入したかどうか
    let mut used_count = vec![vec![0; N]; N];
    let mut in_count = vec![0; N]; // in_count[i] := 搬入口iから搬入した個数
    let mut out_count = vec![0; N]; // out_count[i] := 搬出口iから搬出した個数

    let mut moves = vec![];
    let mut cur_pos = (0, 0);

    for &i in in_order.iter() {
        let c = a[i][in_count[i]];
        let (pi, pj) = in_place[c];
        in_count[i] += 1;
        moved[c] = true;

        // (i, 0)から(pi, pj)に運ぶ
        moves.extend(get_move(cur_pos, (i, 0)));
        moves.push(Move::Pick);
        moves.extend(get_move((i, 0), (pi, pj)));
        moves.push(Move::Quit);
        cur_pos = (pi, pj);

        // 搬出できるコンテナを搬出する
        let (out_i, out_j) = (c / N, c % N);
        for j in out_j..N {
            let c = out_i * N + j;
            if !moved[c] || out_count[out_i] != j {
                break;
            }

            // (pi, pj) := in_place[c]
            // (pi, pj)から(out_i, 4)に運ぶ
            let (pi, pj) = in_place[c];
            used_count[pi][pj] -= 1;
            moves.extend(get_move(cur_pos, (pi, pj)));
            moves.push(Move::Pick);
            moves.extend(get_move((pi, pj), (out_i, 4)));
            moves.push(Move::Quit);
            cur_pos = (out_i, 4);

            out_count[out_i] += 1;
        }
    }

    moves
}

/// 搬入位置の最適化
/// 搬入順序と一緒に最適化できそう
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
