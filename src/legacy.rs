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
        moves.push(Move::Drop);
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
            moves.push(Move::Drop);
            cur_pos = (out_i, 4);

            out_count[out_i] += 1;
        }
    }

    moves
}

fn solve(
    in_order: &Vec<usize>,
    in_place: &Vec<(usize, usize)>,
    a: &Vec<Vec<usize>>,
) -> Vec<Vec<Move>> {
    let mut tasks: VecDeque<Task> = VecDeque::new();
    let mut in_count = vec![0; N];
    let mut out_count = vec![0; N];

    for &i in in_order.iter() {
        let c = a[i][in_count[i]];
        let (pi, pj) = in_place[c];
        in_count[i] += 1;
        tasks.push_back(Task {
            from: (i, 0),
            to: (pi, pj),
        });
    }

    let mut cranes: Vec<Crane> = (0..N)
        .map(|i| Crane {
            status: CraneStatus::Idle,
            task: None,
            pos: (i, 0),
            moves: vec![],
        })
        .collect();
    let mut containers: Vec<Container> = vec![
        Container {
            status: ContainerStatus::Out,
            pos: None
        };
        N * N
    ];
    let mut terminal = vec![vec![vec![TileStatus::Empty; N]; N]; T];

    // TODO: 状態の初期化

    for t in 0..1 {
        // 1. 新たに発生したタスクを列挙する
        // - 搬出できるコンテナが発生したら、タスクの先頭に加える
        for c in 0..N * N {
            if containers[c].status != ContainerStatus::Await {
                continue;
            }
            let (i, j) = (c / N, c % N);
            if out_count[i] != j {
                continue;
            }
            if let Some(cp) = containers[c].pos {
                containers[c].status = ContainerStatus::Assigned;
                tasks.push_front(Task {
                    from: cp,
                    to: (i, N - 1),
                });
            }
        }

        // 2. タスクがないクレーンにタスクを割り当てる
        for i in 0..N {
            if cranes[i].status == CraneStatus::Blown || cranes[i].task.is_some() {
                continue;
            }
            let Some(task) = tasks.pop_front() else {
                continue;
            };

            let cur_pos = cranes[i].pos;
            let moves = get_moves_for_task(&terminal, t, cur_pos, task, i == 0);
            add_moves(&mut terminal, &mut cranes[i], moves, cur_pos, i);
            cranes[i].task = Some(task);
        }

        for i in 0..N {}

        // 3. 各クレーンについての操作
        for i in 0..N {
            /*
            if cranes[i].status == CraneStatus::Blown {
                continue;
            }
            let Some(task) = cranes[i].task else {
                cranes[i].moves.push(Move::Idle);
                continue;
            };

            // 1. 目的地にいれば積み下ろしを行う
            if task.from == cranes[i].pos && cranes[i].status == CraneStatus::Moving {
                cranes[i].moves.push(Move::Pick);
                cranes[i].status = CraneStatus::Hanging;
                continue;
            } else if task.to == cranes[i].pos && cranes[i].status == CraneStatus::Hanging {
                cranes[i].moves.push(Move::Drop);
                cranes[i].status = CraneStatus::Idle;
                continue;
            }

            // 2. 目的地に近づくような操作のうち、実行可能な操作を列挙する
            assert!(
                cranes[i].status == CraneStatus::Moving || cranes[i].status == CraneStatus::Hanging
            );
            let dest = if cranes[i].status == CraneStatus::Moving {
                task.from
            } else {
                task.to
            };
            let over_container = i == 0 || cranes[i].status == CraneStatus::Moving;
            let moves = listup_valid_moves(&terminal, t, cranes[i].pos, dest, over_container);

            // 3. 2.で列挙した操作からランダムに一つ選択し、実行する
            */
        }
    }

    (0..N).map(|i| cranes[i].moves.clone()).collect()
}

use itertools::iproduct;

use crate::{def::*, rnd};

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

#[derive(Clone, Copy, Debug)]
struct Job {
    c: usize,
    from: (usize, usize),
    to: (usize, usize),
}

pub fn solve(
    in_order: &Vec<usize>,
    in_place: &Vec<(usize, usize)>,
    a: &Vec<Vec<usize>>,
) -> Vec<Vec<Move>> {
    let mut jobs: Vec<Vec<Job>> = vec![vec![]; N];
    eprintln!("in_place: {:?}", in_place);
    jobs[0] = convert_in_order_and_in_place_to_jobs(in_order, in_place, a);

    let mut cur_score = simulate(&jobs, a, false);
    for _t in 0..100000 {
        let p = rnd::nextf();

        if p < 0.2 {
            // swap(tasks[i][j1], tasks[i][j2])
            let i = rnd::gen_index(N);
            if jobs[i].len() < 2 {
                continue;
            }
            let (j1, j2) = (rnd::gen_index(jobs[i].len()), rnd::gen_index(jobs[i].len()));
            if j1 == j2 {
                continue;
            }
            jobs[i].swap(j1, j2);
            let new_score = simulate(&jobs, a, false);
            if new_score < cur_score {
                eprintln!("[{_t}] a: {cur_score} -> {new_score}");
                cur_score = new_score;
            } else {
                jobs[i].swap(j1, j2);
            }
        } else if p < 0.6 {
            // swap(tasks[i1][j1], tasks[i2][j2])
            let (i1, i2) = (rnd::gen_index(N), rnd::gen_index(N));
            if jobs[i1].len() == 0 || jobs[i2].len() == 0 {
                continue;
            }
            let (j1, j2) = (
                rnd::gen_index(jobs[i1].len()),
                rnd::gen_index(jobs[i2].len()),
            );
            (jobs[i1][j1], jobs[i2][j2]) = (jobs[i2][j2], jobs[i1][j1]);
            let new_score = simulate(&jobs, a, false);
            if new_score < cur_score {
                eprintln!("[{_t}] b: {cur_score} -> {new_score}");
                cur_score = new_score;
            } else {
                (jobs[i1][j1], jobs[i2][j2]) = (jobs[i2][j2], jobs[i1][j1]);
            }
        } else {
            // move(tasks[i1][j1], tasks[i2][j2])
            let (i1, i2) = (rnd::gen_index(N), rnd::gen_index(N));
            if jobs[i1].len() == 0 {
                continue;
            }
            if i1 == i2 {
                continue;
            }
            let (j1, j2) = (
                rnd::gen_index(jobs[i1].len()),
                rnd::gen_index(jobs[i2].len() + 1),
            );
            let job = jobs[i1].remove(j1);
            jobs[i2].insert(j2, job);
            let new_score = simulate(&jobs, a, false);
            if new_score < cur_score {
                eprintln!("[{_t}] c: {cur_score} -> {new_score}");
                cur_score = new_score;
            } else {
                let job = jobs[i2].remove(j2);
                jobs[i1].insert(j1, job);
            }
        }
    }

    for i in 0..N {
        eprintln!("{}:", i);
        for job in jobs[i].iter() {
            eprintln!(" {:?}", job);
        }
    }

    let t = simulate(&jobs, a, true);
    eprintln!("{t}");

    unimplemented!()
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

/// 評価関数（移動経路とクレーンの衝突を無視したシミュレーション）
fn simulate(jobs: &Vec<Vec<Job>>, a: &Vec<Vec<usize>>, debug: bool) -> usize {
    const NOT_FEASIBLE: usize = 123456;
    let mut c_to_ij = vec![(0, 0); N * N];
    for (i, j) in iproduct!(0..N, 0..N) {
        c_to_ij[a[i][j]] = (i, j);
    }

    // jobsが整合性を保っているか確認する
    // NOTE: a2->b1, b2->a1みたいなケースは弾けない
    for ci in 0..N {
        let mut max_in = vec![0; N];
        let mut max_out = vec![0; N];
        for job in jobs[ci].iter() {
            let (i, j) = c_to_ij[job.c];
            let (out_i, out_j) = (job.c / N, job.c % N);
            if job.from == (i, 0) {
                if max_in[i] > j {
                    return NOT_FEASIBLE;
                }
                max_in[i] = j;
            }
            if job.to == (out_i, N - 1) {
                if max_out[out_i] > out_j {
                    return NOT_FEASIBLE;
                }
                max_out[out_i] = out_j;
            }
        }
    }

    let mut t_in = vec![None; N * N];
    let mut t_out = vec![None; N * N];
    let mut cur_job = vec![0; N];
    let mut busy_till = vec![0; N];
    let mut in_job = vec![false; N];

    for ci in 0..N {
        if jobs[ci].len() == 0 {
            continue;
        }
        busy_till[ci] = ci.abs_diff(jobs[ci][0].from.0) + (0_usize).abs_diff(jobs[ci][0].from.1);
    }

    for t in 0..1000 {
        for ci in 0..N {
            if cur_job[ci] >= jobs[ci].len() {
                continue;
            }
            if t < busy_till[ci] {
                continue;
            }

            let job = &jobs[ci][cur_job[ci]];
            let (i, j) = c_to_ij[job.c];
            let (out_i, out_j) = (job.c / N, job.c % N);
            if !in_job[ci] {
                // 現在のタスクを開始できるか判定
                let is_in_job = job.from == (i, 0);
                if j > 0 && is_in_job {
                    let Some(prev_t_in) = t_in[a[i][j - 1]] else {
                        continue;
                    };
                    if t < prev_t_in + 2 {
                        continue;
                    }
                }
                if is_in_job {
                    t_in[job.c] = Some(t);
                }
                in_job[ci] = true;
                busy_till[ci] = t
                    + jobs[ci][cur_job[ci]]
                        .from
                        .0
                        .abs_diff(jobs[ci][cur_job[ci]].to.0)
                    + jobs[ci][cur_job[ci]]
                        .from
                        .1
                        .abs_diff(jobs[ci][cur_job[ci]].to.1)
                    + 2;
            } else {
                // 現在のタスクを終了できるか判定
                let is_out_job = job.to == (out_i, N - 1);
                if is_out_job && out_j > 0 {
                    let Some(prev_t_out) = t_out[job.c - 1] else {
                        continue;
                    };
                    if t < prev_t_out + 2 {
                        continue;
                    }
                }
                if is_out_job {
                    t_out[job.c] = Some(t);
                }
                in_job[ci] = false;
                cur_job[ci] += 1;
                if cur_job[ci] < jobs[ci].len() {
                    busy_till[ci] = t
                        + jobs[ci][cur_job[ci] - 1]
                            .to
                            .0
                            .abs_diff(jobs[ci][cur_job[ci]].from.0)
                        + jobs[ci][cur_job[ci] - 1]
                            .to
                            .1
                            .abs_diff(jobs[ci][cur_job[ci]].from.1);
                }
            }
        }

        if debug {
            for ci in 0..N {
                if in_job[ci] || t < busy_till[ci] {
                    eprint!("{:3}", jobs[ci][cur_job[ci]].c);
                } else {
                    eprint!("   ");
                }
            }
            eprintln!();
        }

        let finish_count = (0..N)
            .map(|ci| (cur_job[ci] >= jobs[ci].len()) as usize)
            .sum::<usize>();
        if finish_count == N {
            return t;
        }
    }

    NOT_FEASIBLE
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
    let mut dist_sum = 0;

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

        dist_sum += pi.abs_diff(i) + pj;
        // TODO: (i, 0)から(pi, pj)に到達可能かどうか

        // 搬出できるコンテナを搬出する
        let (out_i, out_j) = (c / N, c % N);
        for j in out_j..N {
            let c = out_i * N + j;
            if !moved[c] || out_count[out_i] != j {
                break;
            }

            // (pi, pj) := in_place[c]
            // TODO: (pi, pj)から(out_i, N - 1)が到達可能かどうか
            let (pi, pj) = in_place[c];
            dist_sum += pi.abs_diff(out_i) + pj.abs_diff(N - 1);
            used_count[pi][pj] -= 1;

            out_count[out_i] += 1;
        }
    }

    conflict_count
    // conflict_count * 100000 + dist_sum as i64
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
