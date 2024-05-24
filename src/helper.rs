use itertools::iproduct;

use crate::def::*;
use crate::util::*;

pub fn dist(a: (usize, usize), b: (usize, usize)) -> usize {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
}

pub fn to_moves(
    crane_log: &Vec<Vec<(usize, usize)>>,
    schedules: &Vec<Vec<Schedule>>,
) -> Vec<Vec<Move>> {
    let mut moves = vec![vec![]; N];
    for i in 0..N {
        for t in 0..crane_log[i].len() - 1 {
            let d = (
                crane_log[i][t + 1].0 - crane_log[i][t].0,
                crane_log[i][t + 1].1 - crane_log[i][t].1,
            );
            moves[i].push(Move::from_d(d));
        }
        for s in schedules[i].iter() {
            moves[i][s.start_t] = Move::Pick;
            moves[i][s.end_t] = Move::Drop;
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

pub fn output_ans(moves: &Vec<Vec<Move>>) {
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

pub struct PathFinder {
    id: usize,
    dp: Vec<Vec<Vec<(usize, usize, (usize, usize))>>>, // id, dist, par_p
}

impl PathFinder {
    pub fn new() -> PathFinder {
        PathFinder {
            id: 0,
            dp: vec![vec![vec![(0, 0, (0, 0)); N]; N]; MAX_T],
        }
    }

    /// start_tにfromから開始して、end_tにtoに居るような経路を探索する
    pub fn find_path(
        &mut self,
        ci: usize,
        start_t: usize,
        end_t: usize,
        from: (usize, usize),
        to: (usize, usize),
        over_container: bool,
        crane_log: &Vec<Vec<(usize, usize)>>,
        container_occupations: &Vec<Vec<Vec<(usize, usize, usize)>>>,
    ) -> (Vec<(usize, usize)>, i64) {
        fn move_cost(
            ci: usize,
            t: usize,
            v: (usize, usize),
            nv: (usize, usize),
            over_container: bool,
            crane_log: &Vec<Vec<(usize, usize)>>,
            container_occupations: &Vec<Vec<Vec<(usize, usize, usize)>>>,
        ) -> usize {
            let (ni, nj) = nv;
            let mut collide = 0;
            if !over_container && !(nj == 0 && v == nv) {
                for &(l, r, _) in container_occupations[ni][nj].iter() {
                    if l < t && t <= r {
                        collide += 1;
                    }
                }
            }
            if nj == 0 && collide > 0 {
                for cj in 0..N {
                    if ci == cj {
                        continue;
                    }
                    if t + 1 >= crane_log[cj].len() {
                        continue;
                    }
                    if crane_log[cj][t] == nv && crane_log[cj][t + 1] == (ni, 1) {
                        collide -= 1;
                    }
                }
            }
            for cj in 0..N {
                if ci == cj {
                    continue;
                }
                if t + 1 >= crane_log[cj].len() {
                    continue;
                }
                if crane_log[cj][t + 1] == (ni, nj) {
                    collide += 1;
                }
                if crane_log[cj][t] == (ni, nj) && crane_log[cj][t + 1] == v {
                    collide += 1;
                }
            }
            collide
        }

        self.id += 1;
        self.dp[start_t][from.0][from.1] = (self.id, 0, from);
        for (t, i, j) in iproduct!(start_t..end_t, 0..N, 0..N) {
            if self.dp[t][i][j].0 != self.id {
                continue;
            }
            for d in D {
                let (ni, nj) = (i + d.0, j + d.1);
                if ni >= N || nj >= N {
                    continue;
                }
                let cost = move_cost(
                    ci,
                    t,
                    (i, j),
                    (ni, nj),
                    over_container,
                    crane_log,
                    container_occupations,
                );
                let next = self.dp[t][i][j].1 + cost;
                if self.dp[t + 1][ni][nj].0 != self.id || next < self.dp[t + 1][ni][nj].1 {
                    self.dp[t + 1][ni][nj] = (self.id, next, (i, j));
                }
            }
        }

        // NOTE: 最後に拾う・落とすなどの操作をするため、時刻t+1に留まることができるか調べる必要がある？
        assert_eq!(
            self.dp[end_t][to.0][to.1].0, self.id,
            "{} {} {:?} {:?}",
            start_t, end_t, from, to
        );
        let path = self.restore_path(start_t, end_t, from, to);
        (path, self.dp[end_t][to.0][to.1].1 as i64)
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
            let par = self.dp[cur_t][cur.0][cur.1].2;
            // assert!(self.dp[cur_t - 1][par.0][par.1].1 < self.dp[cur_t][cur.0][cur.1].1);
            path.push(cur);
            cur = par;
            cur_t -= 1;
        }
        path.reverse();
        path
    }
}

pub fn jobs_to_schedules(jobs: &Vec<Job>, assigned_jobs: Vec<Vec<usize>>) -> Vec<Vec<Schedule>> {
    let mut schedules = vec![vec![]; N];
    for (ci, job_indices) in assigned_jobs.into_iter().enumerate() {
        let mut cur_pos: (usize, usize) = (ci, 0);
        let mut cur_t = 0;
        for job_idx in job_indices {
            cur_t +=
                cur_pos.0.abs_diff(jobs[job_idx].from.0) + cur_pos.1.abs_diff(jobs[job_idx].from.1);
            let start_t = cur_t;
            cur_t += 1; // P
            cur_t += jobs[job_idx].from.0.abs_diff(jobs[job_idx].to.0)
                + jobs[job_idx].from.1.abs_diff(jobs[job_idx].to.1);
            let end_t = cur_t;
            cur_t += 1; // Q
            cur_pos = jobs[job_idx].to;
            schedules[ci].push(Schedule {
                start_t,
                end_t,
                job_idx,
            })
        }
    }
    schedules
}

pub fn create_container_occupations(
    jobs: &Vec<Job>,
    schedules: &Vec<Vec<Schedule>>,
    input: &Input,
) -> Vec<Vec<Vec<(usize, usize, usize)>>> {
    let mut container_time_range = vec![(None, None); N * N];
    let mut container_pos = vec![None; N * N];
    let mut t_in = vec![vec![0; N]; N];

    for ci in 0..N {
        for s in schedules[ci].iter() {
            if jobs[s.job_idx].is_in_job() {
                let (i, j) = input.c_to_a_ij[jobs[s.job_idx].c];
                t_in[i][j] = s.start_t;
            } else {
                container_time_range[jobs[s.job_idx].c].1 = Some(s.start_t);
            }

            if !jobs[s.job_idx].is_out_job() {
                container_time_range[jobs[s.job_idx].c].0 = Some(s.end_t);
                container_pos[jobs[s.job_idx].c] = Some(jobs[s.job_idx].to);
            }
        }
    }

    let mut occupations = vec![vec![vec![]; N]; N];

    for c in 0..N * N {
        let (l, r) = container_time_range[c];
        let Some(l) = l else { continue };
        let r = r.unwrap();
        let p = container_pos[c].unwrap();
        occupations[p.0][p.1].push((l, r, c));
    }

    for i in 0..N {
        let mut l = 0;
        for (j, &r) in t_in[i].iter().enumerate() {
            occupations[i][0].push((l, r, input.a[i][j]));
            l = r;
        }
    }

    occupations
}
