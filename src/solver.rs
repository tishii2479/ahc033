use itertools::iproduct;

use crate::def::*;
use crate::helper::*;
use crate::lower::*;
use crate::util::*;

#[derive(Clone, Copy, Debug, Default)]
struct Score {
    raw_score: i64,
    length_sum: i64,
    constraint_penalty: i64,
    container_occupation_penalty: i64,
    schedule_feasibility_penalty: i64,
}

impl Score {
    #[inline]
    fn to_score(&self) -> i64 {
        self.raw_score * 1_000
            + self.constraint_penalty * 1_000_000_000_000
            + self.container_occupation_penalty * 1_000_000_000
            + self.schedule_feasibility_penalty * 1_000_000
        // + self.length_sum
    }
}

pub struct Solver {
    jobs: Vec<Job>,
    schedules: Vec<Vec<Schedule>>,
    constraints: Vec<Constraint>,
    path_finder: PathFinder,
    score: Score,
}

impl Solver {
    pub fn new(jobs: Vec<Job>, input: &Input) -> Solver {
        let path_finder = PathFinder::new();
        let constraints = create_constraints(&jobs, input);
        let mut assigned_jobs: Vec<Vec<usize>> = vec![vec![]; N];
        for job in jobs.iter() {
            assigned_jobs[rnd::gen_index(N)].push(job.idx);
        }
        let schedules = jobs_to_schedules(&jobs, assigned_jobs);

        let mut solver = Solver {
            jobs,
            schedules,
            constraints,
            path_finder,
            score: Score::default(),
        };
        solver.score = solver.eval_schedules(input);
        solver
    }

    pub fn solve(&mut self, iteration: usize, input: &Input) -> Vec<Vec<Move>> {
        eprintln!("[start]  upper-level-score: {:?}", self.score);

        let mut cnt = vec![0; 7];
        for _t in 0..iteration {
            if _t % 1000 == 0 {
                eprintln!("{:?}", cnt);
                cnt = vec![0; 7];
                eprintln!(
                    "[{:7}, {:.8}] {:?} {}",
                    _t,
                    time::elapsed_seconds(),
                    self.score,
                    self.score.to_score()
                );
                for ci in 0..N {
                    eprintln!("ci = {ci} ({})", self.schedules[ci].len());
                    let mut cur_pos = (ci, 0);
                    let mut last_t = 0;
                    for s in self.schedules[ci].iter() {
                        eprint!(
                            "(min = {}, cur = {}, d = {})",
                            dist(cur_pos, self.jobs[s.job_idx].from),
                            s.start_t - last_t,
                            s.start_t - last_t - dist(cur_pos, self.jobs[s.job_idx].from)
                        );
                        eprintln!(
                            "  {:?} (min = {}, cur = {}, d = {}) {:?} {:?}",
                            s,
                            dist(self.jobs[s.job_idx].from, self.jobs[s.job_idx].to) + 1,
                            s.end_t - s.start_t,
                            (s.end_t - s.start_t)
                                - (dist(self.jobs[s.job_idx].from, self.jobs[s.job_idx].to) + 1),
                            self.jobs[s.job_idx].from,
                            self.jobs[s.job_idx].to
                        );
                        last_t = s.end_t + 1;
                        cur_pos = self.jobs[s.job_idx].to;
                    }
                }
            }
            let p = rnd::nextf();
            let threshold = if self.score.to_score() > 1_000_000_000_000 {
                10_000_000_000
            } else if self.score.to_score() > 1_000_000_000 {
                10_000_000
            } else if self.score.to_score() > 1_000_000 {
                10_000
            } else {
                1
            };

            // 一時点のスケジュールを全てのクレーンで伸ばす
            if p < 0.05 {
                if self.action_shift_all_time(threshold, input) {
                    cnt[0] += 1;
                }
            } else if p < 0.1 {
                // 一つのスケジュールの時間を伸ばす・減らす
                if self.action_shift_one_time(threshold, input) {
                    cnt[1] += 1;
                }
            } else if p < 0.3 {
                // 一つのコンテナの置く位置を変更する
                if self.action_move_container(threshold, input) {
                    cnt[2] += 1;
                }
            } else if p < 0.7 {
                // コンテナを置く位置を入れ替える
                if self.action_swap_container(threshold, input) {
                    cnt[3] += 1;
                }
            } else if p < 0.8 {
                // 一つのジョブを移動する
                if self.action_move_one_job(threshold, input) {
                    cnt[4] += 1;
                }
            } else if p < 0.9 {
                // クレーン間でジョブをスワップする
                if self.action_swap_job_between_cranes(threshold, input) {
                    cnt[5] += 1;
                }
            } else {
                // クレーン内でジョブをスワップする
                if self.action_swap_job_in_crane(threshold, input) {
                    cnt[6] += 1;
                }
            }
        }

        eprintln!("[end]    upper-level-score: {:?}", self.score);
        self.eval_schedules(input);
        for ci in 0..N {
            eprintln!("{ci}: {:?}", self.schedules[ci]);
        }

        let container_occupations =
            create_container_occupations(&self.jobs, &self.schedules, input);
        let (mut crane_log, _) = optimize_lower_level(
            &self.jobs,
            &self.schedules,
            &container_occupations,
            &mut self.path_finder,
        );
        for ci in (0..N).rev() {
            crane_log[ci].clear();
            crane_log[ci].push((ci, 0));
            find_path_for_crane(
                ci,
                &self.jobs,
                &self.schedules,
                &mut crane_log,
                &container_occupations,
                &mut self.path_finder,
            );
        }
        to_moves(&crane_log, &self.schedules)
    }

    fn action_swap_container(&mut self, threshold: i64, input: &Input) -> bool {
        let (c1, c2) = (rnd::gen_index(N * N), rnd::gen_index(N * N));
        if c1 == c2 {
            return false;
        }
        let (mut prev_p1, mut prev_p2) = (None, None);
        for job in self.jobs.iter_mut() {
            if job.c == c1 {
                if !job.is_out_job() {
                    prev_p1 = Some(job.to);
                }
            }
            if job.c == c2 {
                if !job.is_out_job() {
                    prev_p2 = Some(job.to);
                }
            }
        }
        let Some(prev_p1) = prev_p1 else { return false };
        let Some(prev_p2) = prev_p2 else { return false };
        for job in self.jobs.iter_mut() {
            if job.c == c1 {
                if !job.is_out_job() {
                    job.to = prev_p2;
                }
                if !job.is_in_job() {
                    job.from = prev_p2;
                }
            }
            if job.c == c2 {
                if !job.is_out_job() {
                    job.to = prev_p1;
                }
                if !job.is_in_job() {
                    job.from = prev_p1;
                }
            }
        }

        let new_score = self.eval_schedules(input);
        let score_diff = new_score.to_score() - self.score.to_score();
        let adopt = score_diff < threshold;
        if adopt {
            self.score = new_score;
        } else {
            for job in self.jobs.iter_mut() {
                if job.c == c1 {
                    if !job.is_out_job() {
                        job.to = prev_p1;
                    }
                    if !job.is_in_job() {
                        job.from = prev_p1;
                    }
                }
                if job.c == c2 {
                    if !job.is_out_job() {
                        job.to = prev_p2;
                    }
                    if !job.is_in_job() {
                        job.from = prev_p2;
                    }
                }
            }
        }
        adopt
    }

    fn action_swap_job_in_crane(&mut self, threshold: i64, input: &Input) -> bool {
        let ci = rnd::gen_index(N);
        if self.schedules[ci].len() < 2 {
            return false;
        }
        let si = rnd::gen_index(self.schedules[ci].len() - 1);
        let sj = si + 1;
        if si == sj {
            return false;
        }
        let cloned_s = self.schedules.clone();
        (
            self.schedules[ci][si].job_idx,
            self.schedules[ci][sj].job_idx,
        ) = (
            self.schedules[ci][sj].job_idx,
            self.schedules[ci][si].job_idx,
        );

        let new_score = self.eval_schedules(input);
        let score_diff = new_score.to_score() - self.score.to_score();
        let adopt = score_diff < threshold;
        if adopt {
            // eprintln!("{:?} -> {:?}", self.score, new_score);
            self.score = new_score;
        } else {
            self.schedules = cloned_s;
        }
        adopt
    }

    fn action_swap_job_between_cranes(&mut self, threshold: i64, input: &Input) -> bool {
        let (ci, cj) = (rnd::gen_index(N), rnd::gen_index(N));
        if ci == cj || self.schedules[ci].len() == 0 || self.schedules[cj].len() == 0 {
            return false;
        }
        let (si, sj) = (
            rnd::gen_index(self.schedules[ci].len()),
            rnd::gen_index(self.schedules[cj].len()),
        );
        let cloned_s = self.schedules.clone();
        (
            self.schedules[ci][si].job_idx,
            self.schedules[cj][sj].job_idx,
        ) = (
            self.schedules[cj][sj].job_idx,
            self.schedules[ci][si].job_idx,
        );

        let new_score = self.eval_schedules(input);
        let score_diff = new_score.to_score() - self.score.to_score();
        let adopt = score_diff < threshold;
        if adopt {
            // eprintln!("{:?} -> {:?}", self.score, new_score);
            self.score = new_score;
        } else {
            self.schedules = cloned_s;
        }
        adopt
    }

    fn action_move_one_job(&mut self, threshold: i64, input: &Input) -> bool {
        let (ci, cj) = (rnd::gen_index(N), rnd::gen_index(N));
        if ci == cj || self.schedules[ci].len() == 0 {
            return false;
        }
        let (si, sj) = (
            rnd::gen_index(self.schedules[ci].len()),
            rnd::gen_index(self.schedules[cj].len() + 1),
        );
        let cloned_s = self.schedules.clone();
        let s = self.schedules[ci].remove(si);
        // s.end_t = s.start_t + dist(self.jobs[s.job_idx].from, self.jobs[s.job_idx].to) + 1;
        // if sj < self.schedules[cj].len() {
        //     self.schedules[cj][sj].start_t = s.end_t
        //         + dist(
        //             self.jobs[s.job_idx].to,
        //             self.jobs[self.schedules[cj][sj].job_idx].to,
        //         )
        //         + 1;
        // }
        self.schedules[cj].insert(sj, s);

        let new_score = self.eval_schedules(input);
        let score_diff = new_score.to_score() - self.score.to_score();
        let adopt = score_diff < threshold;
        if adopt {
            // eprintln!("{:?} -> {:?}", self.score, new_score);
            self.score = new_score;
        } else {
            self.schedules = cloned_s;
        }
        adopt
    }

    fn action_shift_all_time(&mut self, threshold: i64, input: &Input) -> bool {
        let ci = rnd::gen_index(N);
        if self.schedules[ci].len() == 0 {
            return false;
        }
        let d = if rnd::nextf() < 0.5 { 1 } else { !0 };
        // let d = rnd::gen_range(0, 8) - 5;
        let t = if rnd::nextf() < 0.2 {
            rnd::gen_index(self.schedules[ci].last().unwrap().end_t)
        } else {
            self.schedules[ci][rnd::gen_index(self.schedules[ci].len())].start_t + 1
        };
        let t = if d < 1_000 { t } else { t.max(0 - d + 1) }; // オーバーフロー対策
        let cloned_s = self.schedules.clone();
        for i in 0..N {
            for s in self.schedules[i].iter_mut() {
                if s.start_t >= t {
                    s.start_t += d;
                }
                if s.end_t >= t && s.start_t < s.end_t + d {
                    s.end_t += d;
                }
            }
        }
        let new_score = self.eval_schedules(input);
        let score_diff = new_score.to_score() - self.score.to_score();
        let adopt = score_diff < threshold;
        if adopt {
            // eprintln!("{:?} -> {:?}", self.score, new_score);
            self.score = new_score;
        } else {
            self.schedules = cloned_s;
        }
        adopt
    }

    fn action_shift_one_time(&mut self, threshold: i64, input: &Input) -> bool {
        let ci = rnd::gen_index(N);
        if self.schedules[ci].len() == 0 {
            return false;
        }
        let d = if rnd::nextf() < 0.5 { 1 } else { !0 };
        // let d = rnd::gen_range(0, 8) - 5;
        let t = if rnd::nextf() < 0.2 {
            rnd::gen_index(self.schedules[ci].last().unwrap().end_t)
        } else {
            self.schedules[ci][rnd::gen_index(self.schedules[ci].len())].start_t + 1
        };
        let t = if d < 1_000 { t } else { t.max(0 - d + 1) }; // オーバーフロー対策
        let cloned_s = self.schedules.clone();
        for s in self.schedules[ci].iter_mut() {
            if s.start_t >= t {
                assert!(s.start_t + d < 1000, "{} {} {}", s.start_t, d, t);
                s.start_t += d;
            }
            if s.end_t >= t && s.start_t < s.end_t + d {
                s.end_t += d;
            }
        }
        let new_score = self.eval_schedules(input);
        let score_diff = new_score.to_score() - self.score.to_score();
        let adopt = score_diff < threshold;

        if adopt {
            // eprintln!("{:?} -> {:?}", self.score, new_score);
            self.score = new_score;
        } else {
            self.schedules = cloned_s;
        }
        adopt
    }

    fn action_move_container(&mut self, threshold: i64, input: &Input) -> bool {
        let c = rnd::gen_index(N * N);
        let mut prev_p = None;
        let new_p = (rnd::gen_index(N), rnd::gen_range(1, N - 1));
        for job in self.jobs.iter_mut() {
            if job.c != c {
                continue;
            }
            if !job.is_out_job() {
                prev_p = Some(job.to);
                job.to = new_p;
            }
            if !job.is_in_job() {
                prev_p = Some(job.from);
                job.from = new_p;
            }
        }
        let Some(prev_p) = prev_p else { return false };

        let new_score = self.eval_schedules(input);
        let score_diff = new_score.to_score() - self.score.to_score();
        let adopt = score_diff < threshold;
        if adopt {
            // eprintln!("[{_t}] {} ->s {}", self.score, new_score);
            self.score = new_score;
        } else {
            for job in self.jobs.iter_mut() {
                if job.c != c {
                    continue;
                }
                if !job.is_out_job() {
                    job.to = prev_p;
                }
                if !job.is_in_job() {
                    job.from = prev_p;
                }
            }
        }
        adopt
    }

    fn eval_schedules(&mut self, input: &Input) -> Score {
        let raw_score = (0..N)
            .filter(|&ci| self.schedules[ci].len() > 0)
            .map(|ci| self.schedules[ci].last().unwrap().end_t + 1)
            .max()
            .unwrap() as i64;
        let length_sum = (0..N)
            .map(|ci| self.schedules[ci].last().unwrap().end_t + 1)
            .sum::<usize>() as i64;

        let constraint_penalty = self.eval_constraints();
        if constraint_penalty > 0 {
            return Score {
                raw_score,
                length_sum,
                constraint_penalty,
                container_occupation_penalty: 0,
                schedule_feasibility_penalty: 0,
            };
        }

        let container_occupations =
            create_container_occupations(&self.jobs, &self.schedules, input);
        let container_occupation_penalty = self.eval_container_occupation(&container_occupations);
        if container_occupation_penalty > 0 {
            return Score {
                raw_score,
                length_sum,
                constraint_penalty,
                container_occupation_penalty,
                schedule_feasibility_penalty: 0,
            };
        }

        let schedule_feasibility_penalty = self.eval_schedule_feasibility(&container_occupations);

        let score = Score {
            raw_score,
            length_sum,
            constraint_penalty,
            container_occupation_penalty,
            schedule_feasibility_penalty,
        };
        score
    }

    fn eval_schedule_feasibility(
        &mut self,
        container_occupations: &Vec<Vec<Vec<(usize, usize, usize)>>>,
    ) -> i64 {
        // return 0;
        let (_, penalty) = optimize_lower_level(
            &self.jobs,
            &self.schedules,
            &container_occupations,
            &mut self.path_finder,
        );
        penalty as i64
    }

    fn eval_container_occupation(&self, occupations: &Vec<Vec<Vec<(usize, usize, usize)>>>) -> i64 {
        let mut penalty = 0;
        for (i, j) in iproduct!(0..N, 1..N) {
            for k1 in 0..occupations[i][j].len() {
                for k2 in k1 + 1..occupations[i][j].len() {
                    let (l1, r1, _) = occupations[i][j][k1];
                    let (l2, r2, _) = occupations[i][j][k2];
                    let (l, r) = (l1.max(l2), r1.min(r2));
                    if r > l {
                        penalty += r - l;
                    }
                }
            }
        }
        penalty as i64
    }

    fn eval_constraints(&self) -> i64 {
        // TODO: .clone()しない
        let mut constraints = self.constraints.clone();
        for i in 0..N {
            for j in 0..self.schedules[i].len() {
                if j == 0 {
                    constraints.push(Constraint::FirstJob(self.schedules[i][j].job_idx));
                } else {
                    constraints.push(Constraint::Consecutive(
                        self.schedules[i][j - 1].job_idx,
                        self.schedules[i][j].job_idx,
                    ));
                }
            }
        }

        let mut mp = vec![(0, 0); self.jobs.len()];
        for i in 0..N {
            for (j, s) in self.schedules[i].iter().enumerate() {
                mp[s.job_idx] = (i, j);
            }
        }

        let mut penalty = 0;
        for &c in constraints.iter() {
            match c {
                Constraint::Start(prev_job_i, next_job_i) => {
                    let (prev_s, next_s) = (
                        &self.schedules[mp[prev_job_i].0][mp[prev_job_i].1],
                        &self.schedules[mp[next_job_i].0][mp[next_job_i].1],
                    );
                    let interval = 2;
                    if prev_s.start_t + interval > next_s.start_t {
                        penalty += prev_s.start_t + interval - next_s.start_t;
                    }
                }
                Constraint::End(prev_job_i, next_job_i) => {
                    let (prev_s, next_s) = (
                        &self.schedules[mp[prev_job_i].0][mp[prev_job_i].1],
                        &self.schedules[mp[next_job_i].0][mp[next_job_i].1],
                    );
                    let interval = 2;
                    if prev_s.end_t + interval > next_s.end_t {
                        penalty += prev_s.end_t + interval - next_s.end_t;
                    }
                }
                Constraint::Consecutive(prev_job_i, next_job_i) => {
                    let (prev_s, next_s) = (
                        &self.schedules[mp[prev_job_i].0][mp[prev_job_i].1],
                        &self.schedules[mp[next_job_i].0][mp[next_job_i].1],
                    );
                    let interval = dist(self.jobs[prev_job_i].to, self.jobs[next_job_i].from) + 1;
                    if prev_s.end_t + interval > next_s.start_t {
                        penalty += prev_s.end_t + interval - next_s.start_t;
                    }
                }
                Constraint::FirstJob(job_i) => {
                    let s = &self.schedules[mp[job_i].0][mp[job_i].1];
                    let interval = dist((mp[job_i].0, 0), self.jobs[s.job_idx].from);
                    if s.start_t < interval {
                        penalty += interval - s.start_t;
                    }
                }
                Constraint::Job(job_i) => {
                    let s = &self.schedules[mp[job_i].0][mp[job_i].1];
                    let interval = dist(self.jobs[job_i].from, self.jobs[job_i].to) + 1;
                    assert_eq!(self.jobs[s.job_idx].idx, job_i);
                    if s.start_t + interval > s.end_t {
                        penalty += s.start_t + interval - s.end_t;
                    }
                    if s.start_t + interval + 5 < s.end_t {
                        penalty += s.end_t - (s.start_t + interval + 5);
                    }
                }
            }
        }

        penalty as i64
    }
}

fn create_constraints(jobs: &Vec<Job>, input: &Input) -> Vec<Constraint> {
    let mut constraints = vec![];
    let mut prev_in: Vec<Option<usize>> = vec![None; N];
    let mut prev_out: Vec<Option<usize>> = vec![None; N];
    let mut prev_consecutive: Vec<Option<usize>> = vec![None; N * N];

    for (job_i, job) in jobs.iter().enumerate() {
        if job.is_in_job() {
            let (i, _) = input.c_to_a_ij[job.c];
            if let Some(prev_job_i) = prev_in[i] {
                constraints.push(Constraint::Start(prev_job_i, job_i));
            }
            prev_in[i] = Some(job_i);
        } else {
            let prev_job_i = prev_consecutive[job.c].unwrap();
            constraints.push(Constraint::Consecutive(prev_job_i, job_i));
        }

        if job.is_out_job() {
            let i = job.c / N;
            if let Some(prev_job_i) = prev_out[i] {
                constraints.push(Constraint::End(prev_job_i, job_i));
            }
            prev_out[i] = Some(job_i);
        } else {
            prev_consecutive[job.c] = Some(job_i);
        }
    }

    for i in 0..jobs.len() {
        constraints.push(Constraint::Job(i));
    }

    constraints
}
