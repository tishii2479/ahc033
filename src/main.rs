mod def;
mod solver;
mod util;

use std::collections::VecDeque;

use itertools::iproduct;
use proconio::input;

use crate::def::*;
use crate::util::*;

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

    let jobs = listup_jobs(&a);
    // for job in jobs.iter() {
    //     eprintln!("{:?}", job);
    // }
    // eprintln!("{}", jobs.len());
    let state = State::initialize(&jobs, &a);
    // for t in 0..100 {
    //     eprintln!("t={t}");
    //     for i in 0..N {
    //         for j in 0..N {
    //             eprint!(" {:?}", state.container_occupations[t][i][j]);
    //         }
    //         eprintln!();
    //     }
    // }
    dbg!(&state.in_t, &state.out_t);
    let moves = state.to_moves();
    output_ans(&moves);
}

#[derive(Clone, Copy, Debug)]
struct Schedule {
    t: usize,
    p: (usize, usize),
    c: usize,
    job_i: usize,
}

struct State {
    crane_log: Vec<Vec<(usize, usize)>>,
    crane_schedules: Vec<Vec<Schedule>>,
    container_occupations: Vec<Vec<Vec<Option<usize>>>>,
    in_t: Vec<Vec<usize>>,
    out_t: Vec<Vec<usize>>,
    c_to_ij: Vec<(usize, usize)>,
}

impl State {
    fn initialize(jobs: &Vec<Job>, a: &Vec<Vec<usize>>) -> State {
        let mut c_to_in_ij = vec![(0, 0); N * N];
        for (i, j) in iproduct!(0..N, 0..N) {
            c_to_in_ij[a[i][j]] = (i, j);
        }

        // 初期解は全てクレーン0にアサインする
        // TODO: 初期界を貪欲法で作る
        let mut crane_log: Vec<Vec<(usize, usize)>> = (0..N).map(|i| vec![(i, 0)]).collect();
        let mut crane_schedules = vec![vec![]; N];
        let mut container_occupations = vec![vec![vec![None; N]; N]; MAX_T];
        let mut in_t = vec![vec![0; N]; N];
        let mut out_t = vec![vec![0; N]; N];

        for (t, i) in iproduct!(0..MAX_T, 0..N) {
            // 搬入口はコンテナを持った小クレーンは通れなくする
            // TODO: 搬入口のコンテナ状況も更新する
            container_occupations[t][i][0] = Some(N * N);
        }

        let mut cur_pos = (0, 0);
        for (job_i, job) in jobs.iter().enumerate() {
            let path = get_straight_path(cur_pos, job.from);
            crane_log[0].extend(path);
            crane_schedules[0].push(Schedule {
                t: crane_log[0].len(),
                p: job.from,
                c: job.c,
                job_i,
            });
            if job.is_in_job() {
                let (i, j) = c_to_in_ij[job.c];
                in_t[i][j] = crane_log[0].len();
            } else {
                for t in crane_log[0].len()..MAX_T {
                    container_occupations[t][job.from.0][job.from.1] = None;
                }
            }
            crane_log[0].push(job.from); // P

            let path = get_straight_path(job.from, job.to);
            crane_log[0].extend(path);
            crane_schedules[0].push(Schedule {
                t: crane_log[0].len(),
                p: job.to,
                c: job.c,
                job_i,
            });
            if job.is_out_job() {
                let (i, j) = (job.c / N, job.c % N);
                out_t[i][j] = crane_log[0].len();
            } else {
                for t in crane_log[0].len()..MAX_T {
                    container_occupations[t][job.to.0][job.to.1] = Some(job.c);
                }
            }
            crane_log[0].push(job.to); // Q

            cur_pos = job.to;
        }

        State {
            crane_log,
            crane_schedules,
            container_occupations,
            in_t,
            out_t,
            c_to_ij: c_to_in_ij,
        }
    }

    fn find_new_path(&self, ci: usize, t: usize, from: (usize, usize), to: (usize, usize)) {}

    fn to_moves(&self) -> Vec<Vec<Move>> {
        let mut moves = vec![vec![]; N];
        for i in 0..N {
            for t in 0..self.crane_log[i].len() - 1 {
                let d = (
                    self.crane_log[i][t + 1].0 - self.crane_log[i][t].0,
                    self.crane_log[i][t + 1].1 - self.crane_log[i][t].1,
                );
                moves[i].push(Move::from_d(d));
            }
            for (j, schedule) in self.crane_schedules[i].iter().enumerate() {
                let m = if j % 2 == 0 { Move::Pick } else { Move::Drop };
                moves[i][schedule.t - 1] = m;
            }
        }

        let t = (0..N).map(|i| moves[i].len()).max().unwrap();
        for i in 0..N {
            if moves[i].len() < t {
                moves[i].push(Move::Blow);
            }
        }
        moves
    }
}

#[test]
fn test_get_straight_path() {
    dbg!(get_straight_path((0, 0), (3, 2)));
    dbg!(get_straight_path((3, 2), (0, 0)));
    dbg!(get_straight_path((3, 0), (0, 2)));
    dbg!(get_straight_path((0, 2), (3, 0)));
}

fn get_straight_path(from: (usize, usize), to: (usize, usize)) -> Vec<(usize, usize)> {
    let mut v = vec![];
    if from.0 > to.0 {
        v.extend(
            (to.0..from.0)
                .map(|i| (i, from.1))
                .rev()
                .collect::<Vec<(usize, usize)>>(),
        );
    } else {
        v.extend(
            (from.0 + 1..=to.0)
                .map(|i| (i, from.1))
                .collect::<Vec<(usize, usize)>>(),
        );
    }
    if from.1 > to.1 {
        v.extend(
            (to.1..from.1)
                .map(|j| (to.0, j))
                .rev()
                .collect::<Vec<(usize, usize)>>(),
        );
    } else {
        v.extend(
            (from.1 + 1..=to.1)
                .map(|j| (to.0, j))
                .collect::<Vec<(usize, usize)>>(),
        );
    }
    v
}

fn optimize_schedule(state: State) {
    unimplemented!();
}

struct PathFinder {
    id: usize,
    dp: Vec<Vec<Vec<usize>>>,
}

impl PathFinder {
    fn new() -> PathFinder {
        PathFinder {
            id: 0,
            dp: vec![vec![vec![0; N]; N]; MAX_T],
        }
    }

    fn find_path(
        &mut self,
        ci: usize,
        t: usize,
        from: (usize, usize),
        to: (usize, usize),
        crane_log: Vec<Vec<(usize, usize)>>,
        container_occupations: Vec<Vec<Vec<Option<usize>>>>,
    ) {
        self.id += 1;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TileStatus {
    Empty,
    Crane(usize),
    Container,
}

fn add_moves(
    tiles: &mut Vec<Vec<Vec<TileStatus>>>,
    crane_moves: &mut Vec<Move>,
    add_moves: Vec<Move>,
    cur_pos: (usize, usize),
    crane_i: usize,
) {
    let (mut pi, mut pj) = cur_pos;
    for dt in 0..add_moves.len() {
        let t = crane_moves.len() + dt;
        let (di, dj) = add_moves[dt].to_d();
        let (ni, nj) = (pi + di, pj + dj);
        tiles[t + 1][ni][nj] = TileStatus::Crane(crane_i);
        (pi, pj) = (ni, nj);
    }
    crane_moves.extend(add_moves);
}

fn restore_path(
    start_t: usize,
    end_t: usize,
    from: (usize, usize),
    to: (usize, usize),
    dp: &Vec<Vec<Vec<bool>>>,
) -> Vec<Move> {
    // toに到達したので、fromに戻るまでの経路を復元する
    let mut moves = vec![];
    let mut cur = to;
    let mut cur_t = end_t;
    while cur != from || start_t != cur_t {
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
    tiles: &Vec<Vec<Vec<TileStatus>>>,
    start_t: usize,
    from: (usize, usize),
    to: (usize, usize),
    over_container: bool,
) -> Vec<Move> {
    if from == to && !matches!(tiles[start_t + 1][from.0][from.1], TileStatus::Crane(_)) {
        return vec![];
    }
    let mut dp = vec![vec![vec![false; N]; N]; MAX_T];
    dp[start_t][from.0][from.1] = true;
    for t in start_t..MAX_T - 1 {
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
                // 1. tiles[t + 1][ni][nj] == TileStatus::Empty || (tiles[t + 1][ni][nj] == TileStatus::Container && over_container)
                // 2. !(tiles[t][ni][nj] == TileStatus::Crane(c) && tiles[t + 1][i][j] == TileStatus::Crane(c))
                // 3. && !((di != 0 || dj != 0) && nj == 0)
                if (tiles[t + 1][ni][nj] == TileStatus::Empty
                    || (tiles[t + 1][ni][nj] == TileStatus::Container && over_container))
                    && !(matches!(tiles[t][ni][nj], TileStatus::Crane(_))
                        && tiles[t + 1][ni][nj] == tiles[t][i][j])
                    && !((di != 0 || dj != 0) && nj == 0 && !over_container)
                {
                    dp[t + 1][ni][nj] = true;
                    // 最後に拾う・落とすなどの操作をするため、時刻t+2に留まることができるか調べる必要がある
                    if (ni, nj) == to && !matches!(tiles[t + 2][ni][nj], TileStatus::Crane(_)) {
                        let moves = restore_path(start_t, t + 1, from, to, &dp);
                        return moves;
                    }
                }
            }
        }
    }

    // for t in start_t..start_t + 5 {
    //     for i in 0..N {
    //         for j in 0..N {
    //             if dp[t][i][j] {
    //                 eprint!(" ");
    //             } else {
    //                 eprint!("#");
    //             }
    //         }
    //         eprintln!();
    //     }
    // }
    panic!();
}

fn dump_tiles(tiles: &Vec<Vec<TileStatus>>) {
    for i in 0..N {
        for j in 0..N {
            match tiles[i][j] {
                TileStatus::Empty => eprint!("."),
                TileStatus::Container => eprint!("c"),
                TileStatus::Crane(ci) => eprint!("{ci}"),
            }
        }
        eprintln!();
    }
}

fn get_moves_for_task(
    tiles: &Vec<Vec<Vec<TileStatus>>>,
    t: usize,
    cur_pos: (usize, usize),
    job: &Job,
    is_crane_zero: bool,
) -> Vec<Move> {
    let mut moves = vec![];
    moves.extend(listup_valid_moves(tiles, t, cur_pos, job.from, true));
    moves.push(Move::Pick);
    moves.extend(listup_valid_moves(
        tiles,
        t + moves.len(),
        job.from,
        job.to,
        is_crane_zero,
    ));
    moves.push(Move::Drop);
    moves
}

fn listup_jobs(a: &Vec<Vec<usize>>) -> Vec<Job> {
    fn eval(order: &Vec<usize>, place: &Vec<(usize, usize)>, a: &Vec<Vec<usize>>) -> i64 {
        let mut moved = vec![false; N * N]; // moved[c] := コンテナcを搬入したかどうか
        let mut used_count = vec![vec![0; N]; N];
        let mut in_count = vec![0; N]; // in_count[i] := 搬入口iから搬入した個数
        let mut out_count = vec![0; N]; // out_count[i] := 搬出口iから搬出した個数

        let mut conflict_count = 0;
        let mut dist_sum = 0;

        for i in 0..N {
            used_count[i][0] = 1;
        }

        fn dist_to(from: (usize, usize), to: (usize, usize), used_count: &Vec<Vec<usize>>) -> i64 {
            let mut dist = vec![vec![1_000_000; N]; N];
            let mut q = VecDeque::new();
            dist[from.0][from.1] = 0;
            q.push_back(from);
            while let Some((vi, vj)) = q.pop_front() {
                for (di, dj) in D {
                    let (ni, nj) = (vi + di, vj + dj);
                    if ni >= N || nj >= N {
                        continue;
                    }
                    if used_count[ni][nj] > 0 || dist[ni][nj] <= dist[vi][vj] + 1 {
                        continue;
                    }
                    dist[ni][nj] = dist[vi][vj] + 1;
                    q.push_back((ni, nj));
                }
            }
            dist[to.0][to.1]
        }

        for &i in order.iter() {
            let c = a[i][in_count[i]];
            let (pi, pj) = place[c];
            in_count[i] += 1;
            moved[c] = true;

            let (out_i, mut out_j) = (c / N, c % N);
            if out_count[out_i] == out_j {
                dist_sum += dist_to((i, 0), (out_i, N - 1), &used_count);
                out_count[out_i] += 1;
                out_j += 1;
            } else {
                dist_sum += dist_to((i, 0), (pi, pj), &used_count);
                used_count[pi][pj] += 1;
                if used_count[pi][pj] > 1 {
                    conflict_count += 1;
                }
            }

            // 搬出できるコンテナを搬出する
            for j in out_j..N {
                let c = out_i * N + j;
                if !moved[c] || out_count[out_i] != j {
                    break;
                }

                let (pi, pj) = place[c];
                used_count[pi][pj] -= 1;
                dist_sum += dist_to((pi, pj), (out_i, N - 1), &used_count);

                out_count[out_i] += 1;
            }
        }

        conflict_count * 1_000_000_000 + dist_sum as i64
    }

    let mut order = vec![];
    let mut place = vec![(0, 0); N * N];
    for i in 0..N {
        order.extend(vec![i; N]);
    }
    for i in 0..N * N {
        place[i] = (rnd::gen_index(N), rnd::gen_range(1, N - 1));
    }

    let mut cur_score = eval(&order, &place, a);
    for _t in 0..10000 {
        let p = rnd::nextf();
        if p < 0.5 {
            let (i, j) = (rnd::gen_index(order.len()), rnd::gen_index(order.len()));
            if order[i] == order[j] {
                continue;
            }
            order.swap(i, j);

            let new_score = eval(&order, &place, a);
            if new_score < cur_score {
                eprintln!("[{}] a: {} -> {}", _t, cur_score, new_score);
                cur_score = new_score;
            } else {
                order.swap(i, j);
            }
        } else {
            let ci = rnd::gen_index(place.len());
            let prev_p = place[ci];
            place[ci] = (rnd::gen_index(N), rnd::gen_range(1, N - 1));

            let new_score = eval(&order, &place, a);
            if new_score < cur_score {
                eprintln!("[{}] b: {} -> {}", _t, cur_score, new_score);
                cur_score = new_score;
            } else {
                place[ci] = prev_p;
            }
        }
    }

    convert_in_order_and_in_place_to_jobs(&order, &place, a)
}

fn convert_in_order_and_in_place_to_jobs(
    in_order: &Vec<usize>,
    in_place: &Vec<(usize, usize)>,
    a: &Vec<Vec<usize>>,
) -> Vec<Job> {
    let mut jobs = vec![];
    let mut moved = vec![false; N * N]; // moved[c] := コンテナcを搬入したかどうか
    let mut in_count = vec![0; N]; // in_count[i] := 搬入口iから搬入した個数
    let mut out_count = vec![0; N]; // out_count[i] := 搬出口iから搬出した個数

    for &i in in_order.iter() {
        let c = a[i][in_count[i]];
        let (pi, pj) = in_place[c];
        in_count[i] += 1;
        moved[c] = true;

        let (out_i, mut out_j) = (c / N, c % N);
        // 搬出できる場合は直接搬出する
        if out_count[out_i] == out_j {
            jobs.push(Job {
                c,
                from: (i, 0),
                to: (out_i, N - 1),
            });
            out_count[out_i] += 1;
            out_j += 1;
        } else {
            jobs.push(Job {
                c,
                from: (i, 0),
                to: (pi, pj),
            });
        }

        // 搬出できるコンテナを搬出する
        for j in out_j..N {
            let c = out_i * N + j;
            if !moved[c] || out_count[out_i] != j {
                break;
            }

            // (pi, pj) := in_place[c]
            let (pi, pj) = in_place[c];
            jobs.push(Job {
                c,
                from: (pi, pj),
                to: (out_i, N - 1),
            });

            out_count[out_i] += 1;
        }
    }

    assert_eq!(out_count.iter().min().unwrap(), &N);

    jobs
}

#[derive(Clone, Copy, Debug)]
struct Job {
    c: usize,
    from: (usize, usize),
    to: (usize, usize),
}

impl Job {
    fn is_in_job(&self) -> bool {
        self.from.1 == 0
    }

    fn is_out_job(&self) -> bool {
        self.to.1 == N - 1
    }
}
