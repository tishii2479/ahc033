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
    let (crane_schedules, container_occupations) = optimize_upper_level(jobs, &a);
    let state = optimize_lower_level(crane_schedules, container_occupations);
    let moves = state.to_moves();
    output_ans(&moves);
}

fn jobs_to_schedules(jobs: Vec<Job>, cur_pos: (usize, usize), start_t: usize) -> Vec<Schedule> {
    let mut cur_pos = cur_pos;
    let mut cur_t = 0;
    let mut schedules = vec![];
    for job in jobs {
        cur_t += cur_pos.0.abs_diff(job.from.0) + cur_pos.1.abs_diff(job.from.1);
        let start_t = cur_t;
        cur_t += 1; // P
        cur_t += job.from.0.abs_diff(job.to.0) + job.from.1.abs_diff(job.to.1);
        let end_t = cur_t;
        cur_t += 1; // Q
        cur_pos = job.to;
        schedules.push(Schedule {
            start_t,
            end_t,
            job,
        })
    }
    schedules
}

/// 評価関数（移動経路とクレーンの衝突を無視したシミュレーション）
fn eval_schedules(
    schedules: &Vec<Vec<Schedule>>,
    a: &Vec<Vec<usize>>,
) -> (i64, Vec<Vec<Vec<Option<usize>>>>) {
    let mut container_occupations = vec![vec![vec![None; N]; N]; MAX_T];
    for ci in 0..N {}

    (0, container_occupations)
}

fn optimize_upper_level(
    jobs: Vec<Job>,
    a: &Vec<Vec<usize>>,
) -> (Vec<Vec<Schedule>>, Vec<Vec<Vec<Option<usize>>>>) {
    let mut schedules: Vec<Vec<Schedule>> = vec![vec![]; N];
    schedules[0] = jobs_to_schedules(jobs, (0, 0), 0);
    let (_, container_occupations) = eval_schedules(&schedules, a, false);
    (schedules, container_occupations)
}

fn optimize_lower_level(
    crane_schedules: Vec<Vec<Schedule>>,
    container_occupations: Vec<Vec<Vec<Option<usize>>>>,
) -> State {
    let mut state = State::initialize(crane_schedules, container_occupations);
    state
}

#[derive(Clone, Copy, Debug)]
struct Schedule {
    start_t: usize,
    end_t: usize,
    job: Job,
}

fn is_moveable(
    ci: usize,
    t: usize,
    v: (usize, usize),
    d: (usize, usize),
    crane_log: &Vec<Vec<(usize, usize)>>,
    container_occupations: &Vec<Vec<Vec<Option<usize>>>>,
) -> bool {
    let (ni, nj) = (v.0 + d.0, v.1 + d.1);
    if ni >= N || nj >= N {
        return false;
    }
    if container_occupations[t + 1][ni][nj].is_some() {
        return false;
    }
    for cj in 0..N {
        if ci == cj {
            continue;
        }
        if t + 1 >= crane_log[cj].len() {
            continue;
        }
        if crane_log[cj][t + 1] == (ni, nj) {
            return false;
        }
        if crane_log[cj][t] == (ni, nj) && crane_log[cj][t + 1] == v {
            return false;
        }
    }
    true
}

fn find_path_for_schedule(
    ci: usize,
    last_t: usize,
    start_pos: (usize, usize),
    s: &Schedule,
    path_finder: &mut PathFinder,
    crane_log: &Vec<Vec<(usize, usize)>>,
    container_occupations: &Vec<Vec<Vec<Option<usize>>>>,
) -> Vec<(usize, usize)> {
    let mut path = vec![];

    // last_t -> s.start_t の間に start_pos -> s.job.from に移動する
    let path1 = path_finder.find_path(
        ci,
        last_t,
        s.start_t,
        start_pos,
        s.job.from,
        &crane_log,
        &container_occupations,
    );
    path.extend(path1.unwrap());
    path.push(s.job.from); // P

    // s.start_t + 1 -> s.end_t の間に s.job.from -> s.job.to に移動する
    let path2 = path_finder.find_path(
        ci,
        s.start_t + 1,
        s.end_t,
        s.job.from,
        s.job.to,
        &crane_log,
        &container_occupations,
    );
    path.extend(path2.unwrap());
    path.push(s.job.to); // Q

    path
}

struct PathFinder {
    id: usize,
    dp: Vec<Vec<Vec<usize>>>, // id
}

impl PathFinder {
    fn new() -> PathFinder {
        PathFinder {
            id: 0,
            dp: vec![vec![vec![0; N]; N]; MAX_T],
        }
    }

    /// start_tにfromから開始して、end_tにtoに居るような経路を探索する
    fn find_path(
        &mut self,
        ci: usize,
        start_t: usize,
        end_t: usize,
        from: (usize, usize),
        to: (usize, usize),
        crane_log: &Vec<Vec<(usize, usize)>>,
        container_occupations: &Vec<Vec<Vec<Option<usize>>>>,
    ) -> Option<Vec<(usize, usize)>> {
        self.id += 1;
        self.dp[start_t][from.0][from.1] = self.id;
        for t in start_t..end_t {
            for i in 0..N {
                for j in 0..N {
                    if self.dp[t][i][j] != self.id {
                        continue;
                    }
                    for d in D {
                        if !is_moveable(ci, t, (i, j), d, crane_log, container_occupations) {
                            continue;
                        }
                        self.dp[t + 1][i + d.0][j + d.1] = self.id;
                    }
                }
            }
        }

        // NOTE: 最後に拾う・落とすなどの操作をするため、時刻t+1に留まることができるか調べる必要がある？
        if self.dp[end_t][to.0][to.1] != self.id {
            return None;
        }
        Some(self.restore_path(start_t, end_t, from, to))
    }

    fn restore_path(
        &self,
        start_t: usize,
        end_t: usize,
        from: (usize, usize),
        to: (usize, usize),
    ) -> Vec<(usize, usize)> {
        let mut path = vec![];
        let mut cur = to;
        let mut cur_t = end_t;
        while cur != from || start_t != cur_t {
            cur_t -= 1;
            for &(di, dj) in REV_D.iter() {
                let (ni, nj) = (cur.0 + di, cur.1 + dj);
                if ni >= N || nj >= N {
                    continue;
                }
                if self.dp[cur_t][ni][nj] == self.id {
                    path.push(cur);
                    cur = (ni, nj);
                    break;
                }
            }
        }
        path.reverse();
        path
    }
}

struct State {
    crane_log: Vec<Vec<(usize, usize)>>,
    crane_schedules: Vec<Vec<Schedule>>,
    container_occupations: Vec<Vec<Vec<Option<usize>>>>,
    path_finder: PathFinder,
}

impl State {
    fn initialize(
        crane_schedules: Vec<Vec<Schedule>>,
        container_occupations: Vec<Vec<Vec<Option<usize>>>>,
    ) -> State {
        let mut state = State {
            crane_log: (0..N).map(|ci| vec![(ci, 0)]).collect(),
            crane_schedules,
            container_occupations,
            path_finder: PathFinder::new(),
        };
        for ci in 0..N {
            for s in state.crane_schedules[ci].iter() {
                let path = find_path_for_schedule(
                    ci,
                    state.crane_log[ci].len() - 1,
                    *state.crane_log[ci].last().unwrap(),
                    s,
                    &mut state.path_finder,
                    &state.crane_log,
                    &state.container_occupations,
                );
                state.crane_log[ci].extend(path);
            }
        }
        state
    }

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
            for s in self.crane_schedules[i].iter() {
                moves[i][s.start_t - 1] = Move::Pick;
                moves[i][s.end_t - 1] = Move::Drop;
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
                idx: jobs.len(),
                c,
                from: (i, 0),
                to: (out_i, N - 1),
            });
            out_count[out_i] += 1;
            out_j += 1;
        } else {
            jobs.push(Job {
                idx: jobs.len(),
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
                idx: jobs.len(),
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
    idx: usize,
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
