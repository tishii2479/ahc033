use crate::def::*;
use crate::util::*;

use std::collections::VecDeque;

/*
ジョブの作成
*/

pub fn listup_jobs(input: &Input) -> Vec<Job> {
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

    let mut cur_score = eval(&order, &place, &input.a);
    for _t in 0..10000 {
        let p = rnd::nextf();
        if p < 0.5 {
            let (i, j) = (rnd::gen_index(order.len()), rnd::gen_index(order.len()));
            if order[i] == order[j] {
                continue;
            }
            order.swap(i, j);

            let new_score = eval(&order, &place, &input.a);
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

            let new_score = eval(&order, &place, &input.a);
            if new_score < cur_score {
                eprintln!("[{}] b: {} -> {}", _t, cur_score, new_score);
                cur_score = new_score;
            } else {
                place[ci] = prev_p;
            }
        }
    }

    convert_in_order_and_in_place_to_jobs(&order, &place, &input.a)
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
