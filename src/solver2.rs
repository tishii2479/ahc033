use itertools::iproduct;

use crate::def::*;
use crate::helper::*;
use crate::lower::*;
use crate::util::*;

#[allow(unused)]
fn print_schedules(schedules: &Vec<Vec<Schedule>>) {
    for (ci, v) in schedules.iter().enumerate() {
        eprintln!("ci = {};", ci);
        for s in v.iter() {
            eprintln!("{:?}", s);
        }
        eprintln!();
    }
}

#[allow(unused)]
#[derive(Clone, Copy, Debug, Default)]
struct Score {
    raw_score: i64,
    length_sum: i64,
    constraint_penalty: i64,
    container_occupation_penalty: i64,
    violations: i64,
}

impl Score {
    #[inline]
    fn to_score(&self) -> i64 {
        self.raw_score
            + self.constraint_penalty * 1_000_000_000
            + self.container_occupation_penalty * 1_000_000
    }

    #[inline]
    fn to_selection_score(&self) -> i64 {
        self.to_score() + self.violations * 1_000
    }
}

#[derive(Clone)]
struct State {
    jobs: Vec<Job>,
    job_order: Vec<Vec<usize>>,
    schedules: Vec<Vec<Schedule>>,
    constraints: Vec<Constraint>,
    additional_constraints: Vec<(usize, usize)>,
    score: Score,
}

fn init_state(jobs: Vec<Job>, input: &Input) -> State {
    let constraints = create_constraints(&jobs, &input);
    let mut job_order: Vec<Vec<usize>> = vec![vec![]; N];
    for job in jobs.iter() {
        job_order[rnd::gen_index(N)].push(job.idx);
    }
    let schedules = jobs_to_schedules(&jobs, &job_order);
    let additional_constraints = vec![(0, 0); jobs.len()];
    let mut state = State {
        jobs,
        job_order,
        schedules,
        constraints,
        additional_constraints,
        score: Score::default(),
    };
    state.score = state.eval(&input);
    state
}

pub struct Solver2 {
    input: Input,
}

impl Solver2 {
    pub fn new(input: Input) -> Solver2 {
        Solver2 { input }
    }

    pub fn solve(&mut self, jobs: Vec<Job>) -> Vec<Vec<Move>> {
        let mut path_finder = PathFinder::new();
        let mut best_state = init_state(jobs, &self.input);
        for _ in 0..1_000 {
            let mut state = best_state.clone();
            // 上位問題の最適化
            state = optimize_upper_level(&mut state, &self.input);

            // 下位問題の最適化
            let (_, violations) = optimize_lower_level(&state, &mut path_finder, &self.input);

            // 制約の更新
            update_additional_constraints(&mut state, &violations);
            state.score.violations = violations.len() as i64;

            eprintln!("best_score:  {:?}", best_state.score);
            eprintln!("new_score:   {:?}", state.score);
            if state.score.to_selection_score() < best_state.score.to_selection_score() {
                best_state = state;
            }
        }

        optimize_upper_level(&mut best_state, &self.input);
        let (crane_log, _) = optimize_lower_level(&best_state, &mut path_finder, &self.input);
        to_moves(&crane_log, &best_state.schedules)
    }
}

fn update_additional_constraints(state: &mut State, violations: &Vec<Violation>) {
    for v in violations {
        match v {
            Violation::PickUp(job_idx) => {
                if rnd::nextf() < 0.1 {
                    // let c = state.jobs[*job_idx].c;
                    // let new_p = (rnd::gen_index(N), rnd::gen_range(1, N - 1));
                    // for job in state.jobs.iter_mut() {
                    //     if job.c != c {
                    //         continue;
                    //     }
                    //     if !job.is_in_job() {
                    //         job.from = new_p;
                    //     }
                    //     if !job.is_out_job() {
                    //         job.to = new_p;
                    //     }
                    // }
                } else if rnd::nextf() < 0.5 {
                    state.additional_constraints[*job_idx].0 += 1;
                }
            }
            Violation::Carry(job_idx) => {
                if rnd::nextf() < 0.1 {
                    // let c = state.jobs[*job_idx].c;
                    // let new_p = (rnd::gen_index(N), rnd::gen_range(1, N - 1));
                    // for job in state.jobs.iter_mut() {
                    //     if job.c != c {
                    //         continue;
                    //     }
                    //     if !job.is_in_job() {
                    //         job.from = new_p;
                    //     }
                    //     if !job.is_out_job() {
                    //         job.to = new_p;
                    //     }
                    // }
                } else if rnd::nextf() < 0.5 {
                    state.additional_constraints[*job_idx].1 += 1;
                }
            }
        }
    }

    for ac in state.additional_constraints.iter_mut() {
        if rnd::nextf() < 0.1 && ac.0 > 0 {
            ac.0 -= 1;
        }
        if rnd::nextf() < 0.1 && ac.1 > 0 {
            ac.1 -= 1;
        }
    }
}

fn optimize_upper_level(state: &mut State, input: &Input) -> State {
    optimize_schedule(state, input);
    let mut best_state = state.clone();
    for _ in 0..100 {
        let p = rnd::nextf();
        let threshold = 1;
        if p < 0.2 {
            let (c1, c2) = (rnd::gen_index(N), rnd::gen_index(N));
            if state.job_order[c1].len() == 0 || state.job_order[c2].len() == 0 {
                continue;
            }
            let (s1, s2) = (
                rnd::gen_index(state.job_order[c1].len()),
                rnd::gen_index(state.job_order[c2].len()),
            );
            (state.job_order[c1][s1], state.job_order[c2][s2]) =
                (state.job_order[c2][s2], state.job_order[c1][s1]);

            optimize_schedule(state, input);
            let score_diff = state.score.to_score() - best_state.score.to_score();
            if score_diff < threshold {
                best_state = state.clone();
            } else {
                (state.job_order[c1][s1], state.job_order[c2][s2]) =
                    (state.job_order[c2][s2], state.job_order[c1][s1]);
            }
        } else if p < 0.4 {
            let (c1, c2) = (rnd::gen_index(N), rnd::gen_index(N));
            if state.job_order[c1].len() == 0 || state.job_order[c2].len() == 0 {
                continue;
            }
            let (s1, s2) = (
                rnd::gen_index(state.job_order[c1].len()),
                rnd::gen_index(state.job_order[c2].len()),
            );
            let (mut sv1, mut sv2) = (vec![], vec![]);
            while state.job_order[c1].len() > s1 {
                sv1.push(state.job_order[c1].remove(s1));
            }
            while state.job_order[c2].len() > s2 {
                sv2.push(state.job_order[c2].remove(s2));
            }
            for &si in sv1.iter() {
                state.job_order[c2].push(si);
            }
            for &si in sv2.iter() {
                state.job_order[c1].push(si);
            }

            optimize_schedule(state, input);
            let score_diff = state.score.to_score() - best_state.score.to_score();
            if score_diff < threshold {
                best_state = state.clone();
            } else {
                while state.job_order[c1].len() > s1 {
                    state.job_order[c1].remove(s1);
                }
                while state.job_order[c2].len() > s2 {
                    state.job_order[c2].remove(s2);
                }
                for si in sv1 {
                    state.job_order[c1].push(si);
                }
                for si in sv2 {
                    state.job_order[c2].push(si);
                }
            }
        } else if p < 0.6 {
            let (c1, c2) = (rnd::gen_index(N), rnd::gen_index(N));
            if c1 == c2 || state.job_order[c1].len() == 0 {
                continue;
            }
            let (s1, s2) = (
                rnd::gen_index(state.job_order[c1].len()),
                rnd::gen_index(state.job_order[c2].len() + 1),
            );
            let s = state.job_order[c1].remove(s1);
            state.job_order[c2].insert(s2, s);

            optimize_schedule(state, input);
            let score_diff = state.score.to_score() - best_state.score.to_score();
            if score_diff < threshold {
                best_state = state.clone();
            } else {
                let s = state.job_order[c2].remove(s2);
                state.job_order[c1].insert(s1, s);
            }
        } else if p < 0.8 {
            let c = rnd::gen_index(N * N);
            let mut prev_p = None;
            let new_p = (rnd::gen_index(N), rnd::gen_range(1, N - 1));
            for job in state.jobs.iter_mut() {
                if job.c != c {
                    continue;
                }
                if !job.is_in_job() {
                    prev_p = Some(job.from);
                    job.from = new_p;
                }
                if !job.is_out_job() {
                    prev_p = Some(job.to);
                    job.to = new_p;
                }
            }
            let Some(prev_p) = prev_p else {
                continue;
            };
            optimize_schedule(state, input);
            let score_diff = state.score.to_score() - best_state.score.to_score();
            if score_diff < threshold {
                best_state = state.clone();
            } else {
                for job in state.jobs.iter_mut() {
                    if job.c != c {
                        continue;
                    }
                    if !job.is_in_job() {
                        job.from = prev_p;
                    }
                    if !job.is_out_job() {
                        job.to = prev_p;
                    }
                }
            }
        } else {
            let (c1, c2) = (rnd::gen_index(N * N), rnd::gen_index(N * N));
            let (mut prev_p1, mut prev_p2) = (None, None);
            for job in state.jobs.iter_mut() {
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
            let Some(prev_p1) = prev_p1 else { continue; };
            let Some(prev_p2) = prev_p2 else { continue; };
            for job in state.jobs.iter_mut() {
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

            optimize_schedule(state, input);
            let score_diff = state.score.to_score() - best_state.score.to_score();
            if score_diff < threshold {
                best_state = state.clone();
            } else {
                for job in state.jobs.iter_mut() {
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
        }
    }
    best_state
}

fn optimize_schedule(state: &mut State, input: &Input) {
    // eprintln!("a: {:?}", state.job_order);
    state.schedules = jobs_to_schedules(&state.jobs, &state.job_order);
    state.score = state.eval(input);
    for _ in 0..10_000 {
        let p = rnd::nextf();
        let threshold = if state.score.to_score() > 1_000_000_000 {
            10_000_000
        } else if state.score.to_score() > 1_000_000 {
            10_000
        } else if state.score.to_score() > 1_000 {
            10
        } else {
            1
        };

        let _ = if p < 0.3 {
            // 一時点のスケジュールを全てのクレーンで伸ばす
            action_shift_all_time(state, threshold, input)
        } else {
            // 一つのスケジュールの時間を伸ばす・減らす
            action_shift_one_time(state, threshold, input)
        };
    }

    // print_schedules(&state.schedules);
    // eprintln!("{:?}", state.score);
}

fn optimize_lower_level(
    state: &State,
    path_finder: &mut PathFinder,
    input: &Input,
) -> (Vec<Vec<(usize, usize)>>, Vec<Violation>) {
    let container_occupations = create_container_occupations(&state.jobs, &state.schedules, &input);
    let (crane_log, violations) = search_crane_log(
        &state.jobs,
        &state.schedules,
        &container_occupations,
        path_finder,
    );
    (crane_log, violations)
}

/*
評価用関数
*/
impl State {
    fn eval(&self, input: &Input) -> Score {
        let raw_score = (0..N)
            .filter(|&ci| self.schedules[ci].len() > 0)
            .map(|ci| self.schedules[ci].last().unwrap().end_t + 1)
            .max()
            .unwrap() as i64;
        let length_sum = (0..N)
            .filter(|&ci| self.schedules[ci].len() > 0)
            .map(|ci| self.schedules[ci].last().unwrap().end_t + 1)
            .sum::<usize>() as i64;

        let constraint_penalty = self.eval_constraints();
        if constraint_penalty > 0 {
            return Score {
                raw_score,
                length_sum,
                constraint_penalty,
                container_occupation_penalty: 0,
                violations: 0,
            };
        }

        let container_occupations =
            create_container_occupations(&self.jobs, &self.schedules, &input);
        let container_occupation_penalty = self.eval_container_occupation(&container_occupations);

        Score {
            raw_score,
            length_sum,
            constraint_penalty,
            container_occupation_penalty,
            violations: 0,
        }
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
                    let interval = interval + self.additional_constraints[next_job_i].0;
                    if prev_s.end_t + interval > next_s.start_t {
                        penalty += prev_s.end_t + interval - next_s.start_t;
                    }
                }
                Constraint::FirstJob(job_i) => {
                    let s = &self.schedules[mp[job_i].0][mp[job_i].1];
                    let interval = dist((mp[job_i].0, 0), self.jobs[s.job_idx].from);
                    let interval = interval + self.additional_constraints[job_i].0;
                    if s.start_t < interval {
                        penalty += interval - s.start_t;
                    }
                }
                Constraint::Job(job_i) => {
                    let s = &self.schedules[mp[job_i].0][mp[job_i].1];
                    let interval = dist(self.jobs[job_i].from, self.jobs[job_i].to) + 1;
                    let interval = interval + self.additional_constraints[job_i].1;
                    if s.start_t + interval > s.end_t {
                        penalty += (s.start_t + interval - s.end_t) * 100;
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

/*
焼きなましの近傍
*/
fn action_shift_all_time(state: &mut State, threshold: i64, input: &Input) -> bool {
    let ci = rnd::gen_index(N);
    if state.schedules[ci].len() == 0 {
        return false;
    }
    let d = if rnd::nextf() < 0.5 { 1 } else { !0 };
    let t = if rnd::nextf() < 0.2 {
        rnd::gen_index(state.schedules[ci].last().unwrap().end_t)
    } else {
        state.schedules[ci][rnd::gen_index(state.schedules[ci].len())].start_t + 1
    };
    let t = if d < 1_000 { t } else { t.max(0 - d + 1) }; // オーバーフロー対策
    let cloned_s = state.schedules.clone();
    for i in 0..N {
        for s in state.schedules[i].iter_mut() {
            if s.start_t >= t {
                s.start_t += d;
            }
            if s.end_t >= t && s.start_t < s.end_t + d {
                s.end_t += d;
            }
        }
    }
    let new_score = state.eval(&input);
    let score_diff = new_score.to_score() - state.score.to_score();
    let adopt = score_diff < threshold;
    if adopt {
        state.score = new_score;
    } else {
        state.schedules = cloned_s;
    }
    adopt
}

fn action_shift_one_time(state: &mut State, threshold: i64, input: &Input) -> bool {
    let ci = rnd::gen_index(N);
    if state.schedules[ci].len() == 0 {
        return false;
    }

    let d = if rnd::nextf() < 0.5 { 1 } else { !0 };
    let t = if rnd::nextf() < 0.2 {
        rnd::gen_index(state.schedules[ci].last().unwrap().end_t)
    } else if rnd::nextf() < 0.6 {
        state.schedules[ci][rnd::gen_index(state.schedules[ci].len())].start_t + 1
    } else {
        state.schedules[ci][rnd::gen_index(state.schedules[ci].len())].end_t + 1
    };
    let t = if d < 1_000_000 { t } else { t.max(0 - d + 1) }; // オーバーフロー対策
    let mut last_t = t - 1;
    let cloned_s = state.schedules.clone();
    for s in state.schedules[ci].iter_mut() {
        if s.start_t >= t && s.start_t + d != last_t {
            assert!(s.start_t + d < 1_000_000, "{} {} {}", s.start_t, d, t);
            s.start_t += d;
        }
        if s.end_t >= t && s.start_t < s.end_t + d {
            s.end_t += d;
        }
        last_t = s.end_t;
    }
    let new_score = state.eval(&input);
    let score_diff = new_score.to_score() - state.score.to_score();
    let adopt = score_diff < threshold;

    if adopt {
        state.score = new_score;
    } else {
        state.schedules = cloned_s;
    }
    adopt
}
