use itertools::iproduct;

use crate::def::*;

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
// tasks[i] := クレーンiのタスクの順序

// 近傍
// move(tasks[i1][j1], tasks[i2][j2])
// swap(tasks[i][j1], tasks[i][j2])
// swap(tasks[i1][j1], tasks[i2][j2])
// swap(tasks[i1][j1:], tasks[i2][j2:])
// swap(tasks[i1][:j1], tasks[i2][:j2])

// 制約
// s_{i, j} := コンテナa[i,j]が搬入される時刻
// t_{c} := コンテナcが搬出される時刻
//
// s_{i, j1} + 2 <= s_{i, j2} where j1 < j2
// t_{c1} + 2 <= t_{c2} where c1 < c2 and g(c1) == g(c2)
// 中継場所は固定
pub fn solve(
    in_order: &Vec<usize>,
    in_place: &Vec<(usize, usize)>,
    a: &Vec<Vec<usize>>,
) -> Vec<Vec<Move>> {
    let mut jobs: Vec<Vec<Job>> = vec![vec![]; N];
    eprintln!("in_place: {:?}", in_place);
    jobs[0] = convert_in_order_and_in_place_to_jobs(in_order, in_place, a);
    for job in jobs[0].iter() {
        eprintln!("{:?}", job);
    }
    eprintln!("{}", jobs[0].len());
    let t_end = simulate(&jobs, a);
    eprintln!("{}", t_end);

    let mut cur_pos = (0, 0);
    let mut moves = vec![vec![]; N];
    for i in 1..N {
        moves[i].push(Move::Blow);
    }
    for job in jobs[0].iter() {
        moves[0].extend(get_move(cur_pos, job.from));
        moves[0].push(Move::Pick);
        moves[0].extend(get_move(job.from, job.to));
        moves[0].push(Move::Drop);
        cur_pos = job.to;
    }
    moves
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

// 評価関数（移動経路とクレーンの衝突を無視したシミュレーション）
fn simulate(jobs: &Vec<Vec<Job>>, a: &Vec<Vec<usize>>) -> usize {
    let mut t = 0;
    let mut t_in = vec![None; N * N];
    let mut t_out = vec![None; N * N];
    let mut cur_task = vec![0; N];
    let mut busy_till = vec![0; N];
    let mut in_job = vec![false; N];
    let mut c_to_ij = vec![(0, 0); N * N];
    for (i, j) in iproduct!(0..N, 0..N) {
        c_to_ij[a[i][j]] = (i, j);
    }

    for ci in 0..N {
        if jobs[ci].len() == 0 {
            continue;
        }
        busy_till[ci] = ci.abs_diff(jobs[ci][0].from.0) + (0_usize).abs_diff(jobs[ci][0].from.1);
    }

    loop {
        for ci in 0..N {
            if cur_task[ci] >= jobs[ci].len() {
                continue;
            }
            if t < busy_till[ci] {
                continue;
            }

            let job = &jobs[ci][cur_task[ci]];
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
                eprintln!("start: {:?} {}", job, t);
                if is_in_job {
                    t_in[job.c] = Some(t);
                }
                in_job[ci] = true;
                busy_till[ci] = t
                    + jobs[ci][cur_task[ci]]
                        .from
                        .0
                        .abs_diff(jobs[ci][cur_task[ci]].to.0)
                    + jobs[ci][cur_task[ci]]
                        .from
                        .1
                        .abs_diff(jobs[ci][cur_task[ci]].to.1)
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
                eprintln!("finish: {:?} {}", job, t);
                if is_out_job {
                    t_out[job.c] = Some(t);
                }
                in_job[ci] = false;
                cur_task[ci] += 1;
                if cur_task[ci] < jobs[ci].len() {
                    busy_till[ci] = t
                        + jobs[ci][cur_task[ci] - 1]
                            .to
                            .0
                            .abs_diff(jobs[ci][cur_task[ci]].from.0)
                        + jobs[ci][cur_task[ci] - 1]
                            .to
                            .1
                            .abs_diff(jobs[ci][cur_task[ci]].from.1);
                }
            }
        }
        let finish_count = (0..N)
            .map(|ci| (cur_task[ci] >= jobs[ci].len()) as usize)
            .sum::<usize>();
        if finish_count == N {
            break;
        }
        t += 1;
    }

    for i in 0..N * N {
        eprintln!("{i}: {} {}", t_in[i].unwrap(), t_out[i].unwrap());
    }
    t
}
