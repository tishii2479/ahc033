use itertools::iproduct;

use crate::def::*;
use crate::helper::*;
use crate::lower::*;
use crate::util::*;

#[derive(Clone, Copy, Debug, Default)]
struct Score {
    raw_score: i64,
    constraint_penalty: i64,
    container_occupation_penalty: i64,
    schedule_feasibility_penalty: i64,
}

impl Score {
    fn to_score(&self) -> i64 {
        self.raw_score
            + self.constraint_penalty * 1_000_000
            + self.container_occupation_penalty * 1_000_000
            + self.schedule_feasibility_penalty * 1_000
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
        let mut assigned_jobs: Vec<Vec<Job>> = vec![vec![]; N];
        for job in jobs.iter() {
            // assigned_jobs[rnd::gen_index(N)].push(job.clone());
            assigned_jobs[0].push(job.clone());
        }
        let schedules = jobs_to_schedules(assigned_jobs);

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

        for _t in 0..iteration {
            if _t % 1000 == 0 {
                eprintln!("{}", _t);
            }
            let p = rnd::nextf();
            let start_temp = if self.score.to_score() > 1_000_000 {
                10_000
            } else if self.score.to_score() > 1000 {
                100
            } else {
                1
            } as f64;
            let progress = _t as f64 / iteration as f64;
            let cur_temp = start_temp.powf(1. - progress);
            let threshold = -(cur_temp * progress) * rnd::nextf().ln();
            let threshold = threshold.round() as i64;
            // dbg!(self.score, threshold);
            let threshold = if self.score.to_score() > 1_000_000 {
                1_000
            } else {
                1
            };
            if p < 0.7 {
                // 一つのスケジュールの時間を伸ばす・減らす
                self.action_shift_one_time(threshold, input);
            } else if p < 0.8 {
                // 一つのコンテナの置く位置を変更する
                self.action_change_container_p(threshold, input);
            } else if p < 0.9 {
                // 一つのジョブを移動する
                self.action_move_one_job(threshold, input);
            } else if p < 0.95 {
                // クレーン間でジョブをスワップする
                self.action_swap_job_between_cranes(threshold, input);
            } else {
                // クレーン内でジョブをスワップする
                self.action_swap_job_in_crane(threshold, input);
            }
            // 一時点のスケジュールを全てのクレーンで伸ばす
            // コンテナを置く位置を入れ替える
        }

        eprintln!("[end]    upper-level-score: {:?}", self.score);
        // assert!(self.eval_schedules() < 1_000input);
        self.eval_schedules(input);

        let container_occupations = create_container_occupations(&self.schedules, input);
        let (mut crane_log, _) = optimize_lower_level(&self.schedules, &container_occupations);
        for ci in (0..N).rev() {
            crane_log[ci].clear();
            crane_log[ci].push((ci, 0));
            find_path_for_crane(
                ci,
                &self.schedules,
                &mut crane_log,
                &container_occupations,
                &mut self.path_finder,
                true,
            );
        }
        to_moves(&crane_log, &self.schedules)
    }

    fn action_swap_job_in_crane(&mut self, threshold: i64, input: &Input) {
        let ci = rnd::gen_index(N);
        if self.schedules[ci].len() < 2 {
            return;
        }
        let si = rnd::gen_index(self.schedules[ci].len() - 1);
        let sj = si + 1;
        // let (si, sj) = (
        //     rnd::gen_index(self.schedules[ci].len()),
        //     rnd::gen_index(self.schedules[ci].len()),
        // );
        if si == sj {
            return;
        }
        (self.schedules[ci][si].job, self.schedules[ci][sj].job) =
            (self.schedules[ci][sj].job, self.schedules[ci][si].job);

        let new_score = self.eval_schedules(input);
        if new_score.to_score() - self.score.to_score() < threshold {
            eprintln!("{:?} -> {:?}", self.score, new_score);
            self.score = new_score;
        } else {
            (self.schedules[ci][si].job, self.schedules[ci][sj].job) =
                (self.schedules[ci][sj].job, self.schedules[ci][si].job);
        }
    }

    fn action_swap_job_between_cranes(&mut self, threshold: i64, input: &Input) {
        let (ci, cj) = (rnd::gen_index(N), rnd::gen_index(N));
        if ci == cj || self.schedules[ci].len() == 0 || self.schedules[cj].len() == 0 {
            return;
        }
        let (si, sj) = (
            rnd::gen_index(self.schedules[ci].len()),
            rnd::gen_index(self.schedules[cj].len()),
        );
        (self.schedules[ci][si].job, self.schedules[cj][sj].job) =
            (self.schedules[cj][sj].job, self.schedules[ci][si].job);

        let new_score = self.eval_schedules(input);
        if new_score.to_score() - self.score.to_score() < threshold {
            eprintln!("{:?} -> {:?}", self.score, new_score);
            self.score = new_score;
        } else {
            (self.schedules[ci][si].job, self.schedules[cj][sj].job) =
                (self.schedules[cj][sj].job, self.schedules[ci][si].job);
        }
    }

    fn action_move_one_job(&mut self, threshold: i64, input: &Input) {
        let (ci, cj) = (rnd::gen_index(N), rnd::gen_index(N));
        if ci == cj || self.schedules[ci].len() == 0 {
            return;
        }
        let (si, sj) = (
            rnd::gen_index(self.schedules[ci].len()),
            rnd::gen_index(self.schedules[cj].len() + 1),
        );
        let s = self.schedules[ci].remove(si);
        self.schedules[cj].insert(sj, s);

        let new_score = self.eval_schedules(input);
        if new_score.to_score() - self.score.to_score() < threshold {
            eprintln!("{:?} -> {:?}", self.score, new_score);
            self.score = new_score;
        } else {
            let s = self.schedules[cj].remove(sj);
            self.schedules[ci].insert(si, s);
        }
    }

    fn action_shift_one_time(&mut self, threshold: i64, input: &Input) {
        let ci = rnd::gen_index(N);
        if self.schedules[ci].len() == 0 {
            return;
        }
        let d = if rnd::nextf() < 0.5 { 1 } else { !0 };
        let t = rnd::gen_index(self.schedules[ci].last().unwrap().end_t);
        let t = if d == 1 { t } else { t.max(1) }; // オーバーフロー対策
        let a = self.schedules[ci].clone();
        for s in self.schedules[ci].iter_mut() {
            if s.start_t >= t {
                s.start_t += d;
            }
            if s.end_t >= t && s.start_t < s.end_t + d {
                s.end_t += d;
            }
        }
        let new_score = self.eval_schedules(input);

        if new_score.to_score() - self.score.to_score() < threshold {
            eprintln!("{:?} -> {:?}", self.score, new_score);
            self.score = new_score;
        } else {
            self.schedules[ci] = a;
        }
    }

    fn action_change_container_p(&mut self, threshold: i64, input: &Input) {
        let c = rnd::gen_index(N * N);
        let mut prev_p = None;
        let new_p = (rnd::gen_index(N), rnd::gen_range(1, N - 1));
        for i in 0..N {
            for s in self.schedules[i].iter_mut() {
                if s.job.c != c {
                    continue;
                }
                if !s.job.is_out_job() {
                    prev_p = Some(s.job.to);
                    s.job.to = new_p;
                }
                if !s.job.is_in_job() {
                    prev_p = Some(s.job.from);
                    s.job.from = new_p;
                }
            }
        }
        let Some(prev_p) = prev_p else { return };

        let new_score = self.eval_schedules(input);
        if new_score.to_score() - self.score.to_score() < threshold {
            // eprintln!("[{_t}] {} ->s {}", self.score, new_score);
            self.score = new_score;
        } else {
            for i in 0..N {
                for s in self.schedules[i].iter_mut() {
                    if s.job.c != c {
                        continue;
                    }
                    if !s.job.is_out_job() {
                        s.job.to = prev_p;
                    }
                    if !s.job.is_in_job() {
                        s.job.from = prev_p;
                    }
                }
            }
        }
    }

    fn eval_schedules(&mut self, input: &Input) -> Score {
        let raw_score = (0..N)
            .filter(|&ci| self.schedules[ci].len() > 0)
            .map(|ci| self.schedules[ci].last().unwrap().end_t)
            .max()
            .unwrap() as i64;

        let constraint_penalty = self.eval_constraints();
        if constraint_penalty > 0 {
            return Score {
                raw_score,
                constraint_penalty,
                container_occupation_penalty: 0,
                schedule_feasibility_penalty: 0,
            };
        }

        let container_occupations = create_container_occupations(&self.schedules, input);
        let container_occupation_penalty = self.eval_container_occupation(&container_occupations);
        if container_occupation_penalty > 0 {
            return Score {
                raw_score,
                constraint_penalty,
                container_occupation_penalty,
                schedule_feasibility_penalty: 0,
            };
        }

        let schedule_feasibility_penalty = self.eval_schedule_feasibility(&container_occupations);

        let score = Score {
            raw_score,
            constraint_penalty,
            container_occupation_penalty,
            schedule_feasibility_penalty,
        };
        eprintln!("{:?}", score);
        score
    }

    fn eval_schedule_feasibility(
        &mut self,
        container_occupations: &Vec<Vec<Vec<(usize, usize, usize)>>>,
    ) -> i64 {
        let (_, penalty) = optimize_lower_level(&self.schedules, &container_occupations);
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
                    constraints.push(Constraint::FirstJob(self.schedules[i][j].job.idx));
                } else {
                    constraints.push(Constraint::Consecutive(
                        self.schedules[i][j - 1].job.idx,
                        self.schedules[i][j].job.idx,
                    ));
                }
            }
        }

        let mut mp = vec![(0, 0); self.jobs.len()];
        for i in 0..N {
            for (j, s) in self.schedules[i].iter().enumerate() {
                mp[s.job.idx] = (i, j);
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
                    let interval = dist(prev_s.job.to, next_s.job.from) + 1;
                    if prev_s.end_t + interval > next_s.start_t {
                        penalty += prev_s.end_t + interval - next_s.start_t;
                    }
                }
                Constraint::FirstJob(job_i) => {
                    let s = &self.schedules[mp[job_i].0][mp[job_i].1];
                    let interval = dist((mp[job_i].0, 0), s.job.from);
                    if s.start_t < interval {
                        penalty += interval - s.start_t;
                    }
                }
                Constraint::Job(job_i) => {
                    let s = &self.schedules[mp[job_i].0][mp[job_i].1];
                    let interval = dist(s.job.from, s.job.to) + 2;
                    assert_eq!(s.job.idx, job_i);
                    if s.start_t + interval > s.end_t {
                        penalty += s.start_t + interval - s.end_t;
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
