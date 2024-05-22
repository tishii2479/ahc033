mod def;
mod solver;
mod util;

use std::collections::VecDeque;

use itertools::iproduct;
use proconio::input;

use crate::def::*;
use crate::util::*;

struct Input {
    a: Vec<Vec<usize>>,
    c_to_a_ij: Vec<(usize, usize)>,
}

impl Input {
    fn new(a: Vec<Vec<usize>>) -> Input {
        let mut c_to_a_ij = vec![(0, 0); N * N];
        for (i, j) in iproduct!(0..N, 0..N) {
            c_to_a_ij[a[i][j]] = (i, j);
        }
        Input { a, c_to_a_ij }
    }
}

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

    let input = Input::new(a);
    let jobs = listup_jobs(&input);
    // for job in jobs.iter() {
    //     eprintln!("{:?}", job);
    // }
    // eprintln!("{}", jobs.len());
    let (crane_schedules, container_occupations) = optimize_upper_level(jobs, &input);
    let state = optimize_lower_level(crane_schedules, container_occupations);

    let moves = state.to_moves();
    output_ans(&moves);
}

fn jobs_to_schedules(jobs: Vec<Vec<Job>>) -> Vec<Vec<Schedule>> {
    let mut schedules = vec![vec![]; N];
    for (ci, jobs_c) in jobs.into_iter().enumerate() {
        let mut cur_pos: (usize, usize) = (ci, 0);
        let mut cur_t = 0;
        for job in jobs_c {
            cur_t += cur_pos.0.abs_diff(job.from.0) + cur_pos.1.abs_diff(job.from.1);
            let start_t = cur_t;
            cur_t += 1; // P
            cur_t += job.from.0.abs_diff(job.to.0) + job.from.1.abs_diff(job.to.1);
            let end_t = cur_t;
            cur_t += 1; // Q
            cur_pos = job.to;
            schedules[ci].push(Schedule {
                start_t,
                end_t,
                job,
            })
        }
    }
    schedules
}

fn create_container_occupations(
    schedules: &Vec<Vec<Schedule>>,
    input: &Input,
) -> Vec<Vec<Vec<Option<usize>>>> {
    let mut container_occupations = vec![vec![vec![None; N]; N]; MAX_T];
    let mut t_in = vec![vec![0; N]; N];

    let mut container_time_range = vec![(None, None); N * N];
    let mut container_pos = vec![None; N * N];
    for ci in 0..N {
        for s in schedules[ci].iter() {
            if s.job.is_in_job() {
                let (i, j) = input.c_to_a_ij[s.job.c];
                t_in[i][j] = s.start_t;
            } else {
                container_time_range[s.job.c].1 = Some(s.start_t);
            }

            if !s.job.is_out_job() {
                container_time_range[s.job.c].0 = Some(s.end_t);
                container_pos[s.job.c] = Some(s.job.to);
            }
        }
    }

    for c in 0..N * N {
        let (l, r) = container_time_range[c];
        let Some(l) = l else { continue };
        let r = r.unwrap();
        let p = container_pos[c].unwrap();
        for t in l + 1..r {
            assert!(container_occupations[t][p.0][p.1].is_none());
            container_occupations[t][p.0][p.1] = Some(c);
        }
    }

    for i in 0..N {
        let mut l = 0;
        for (j, &r) in t_in[i].iter().enumerate() {
            for t in l..r {
                container_occupations[t][i][0] = Some(input.a[i][j]);
            }
            l = r;
        }
    }

    container_occupations
}

#[derive(Clone, Copy, Debug)]
enum Constraint {
    Start(usize, usize),
    End(usize, usize),
    FirstJob(usize),
    Consecutive(usize, usize),
    Job(usize),
}

fn dist(a: (usize, usize), b: (usize, usize)) -> usize {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
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

    for (job_i, job) in jobs.iter().enumerate() {
        constraints.push(Constraint::Job(job_i));
    }

    constraints
}

fn optimize_upper_level(
    jobs: Vec<Job>,
    input: &Input,
) -> (Vec<Vec<Schedule>>, Vec<Vec<Vec<Option<usize>>>>) {
    let constraints = create_constraints(&jobs, input);
    let mut assigned_jobs: Vec<Vec<Job>> = vec![vec![]; N];
    for job in jobs.iter() {
        assigned_jobs[rnd::gen_index(N)].push(job.clone());
    }

    let mut schedules = jobs_to_schedules(assigned_jobs);
    let mut cur_score = eval_schedules(&schedules, &constraints, &jobs, input);
    eprintln!("[start]  upper-level-score: {}", cur_score);

    for _t in 0..100_000 {
        let p = rnd::nextf();
        let threshold = if cur_score > 1_000_000 { 1_000 } else { 1 };
        if p < 0.2 {
            // 1. 一つのスケジュールの時間を伸ばす・減らす
            let ci = rnd::gen_index(N);
            if schedules[ci].len() == 0 {
                continue;
            }
            let d = if rnd::nextf() < 0.5 { 1 } else { !0 };
            let t = rnd::gen_index(schedules[ci].last().unwrap().end_t);
            let t = if d == 1 { t } else { t.max(1) }; // オーバーフロー対策
            let a = schedules[ci].clone();
            for s in schedules[ci].iter_mut() {
                if s.start_t >= t {
                    s.start_t += d;
                }
                if s.end_t >= t && s.start_t < s.end_t + d {
                    s.end_t += d;
                }
            }
            let new_score = eval_schedules(&schedules, &constraints, &jobs, input);

            if new_score - cur_score < threshold {
                cur_score = new_score;
                // eprintln!("[{_t}] {} -> {}", cur_score, new_score);
            } else {
                schedules[ci] = a;
            }
        } else if p < 0.4 {
            // 一つのコンテナの置く位置を変更する
            let c = rnd::gen_index(N * N);
            let mut prev_p = None;
            let new_p = (rnd::gen_index(N), rnd::gen_range(1, N - 1));
            for i in 0..N {
                for s in schedules[i].iter_mut() {
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
            let Some(prev_p) = prev_p else { continue };

            let new_score = eval_schedules(&schedules, &constraints, &jobs, input);
            if new_score - cur_score < threshold {
                // eprintln!("[{_t}] {} -> {}", cur_score, new_score);
                cur_score = new_score;
            } else {
                for i in 0..N {
                    for s in schedules[i].iter_mut() {
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
        } else if p < 0.6 {
            // 3. 一つのジョブを移動する
            let (ci, cj) = (rnd::gen_index(N), rnd::gen_index(N));
            if ci == cj || schedules[ci].len() == 0 {
                continue;
            }
            let (si, sj) = (
                rnd::gen_index(schedules[ci].len()),
                rnd::gen_index(schedules[cj].len() + 1),
            );
            let s = schedules[ci].remove(si);
            schedules[cj].insert(sj, s);

            let new_score = eval_schedules(&schedules, &constraints, &jobs, input);
            if new_score - cur_score < threshold {
                // eprintln!("[{_t}] {} -> {}", cur_score, new_score);
                cur_score = new_score;
            } else {
                let s = schedules[cj].remove(sj);
                schedules[ci].insert(si, s);
            }
        } else if p < 0.8 {
            // 4. クレーン間でジョブをスワップする
            let (ci, cj) = (rnd::gen_index(N), rnd::gen_index(N));
            if ci == cj || schedules[ci].len() == 0 || schedules[cj].len() == 0 {
                continue;
            }
            let (si, sj) = (
                rnd::gen_index(schedules[ci].len()),
                rnd::gen_index(schedules[cj].len()),
            );
            (schedules[ci][si].job, schedules[cj][sj].job) =
                (schedules[cj][sj].job, schedules[ci][si].job);

            let new_score = eval_schedules(&schedules, &constraints, &jobs, input);
            if new_score - cur_score < threshold {
                cur_score = new_score;
                // eprintln!("[{_t}] {} -> {}", cur_score, new_score);
            } else {
                (schedules[ci][si].job, schedules[cj][sj].job) =
                    (schedules[cj][sj].job, schedules[ci][si].job);
            }
        } else {
            // 5. クレーン内でジョブをスワップする
            let ci = rnd::gen_index(N);
            if schedules[ci].len() < 2 {
                continue;
            }
            let (si, sj) = (
                rnd::gen_index(schedules[ci].len()),
                rnd::gen_index(schedules[ci].len()),
            );
            if si == sj {
                continue;
            }
            (schedules[ci][si].job, schedules[ci][sj].job) =
                (schedules[ci][sj].job, schedules[ci][si].job);

            let new_score = eval_schedules(&schedules, &constraints, &jobs, input);
            if new_score - cur_score < threshold {
                cur_score = new_score;
                // eprintln!("[{_t}] {} -> {}", cur_score, new_score);
            } else {
                (schedules[ci][si].job, schedules[ci][sj].job) =
                    (schedules[ci][sj].job, schedules[ci][si].job);
            }
        }
        // 2. 一時点のスケジュールを全てのクレーンで伸ばす
    }

    eprintln!("[end]    upper-level-score: {}", cur_score);
    assert!(eval_schedules(&schedules, &constraints, &jobs, input) < 1_000);
    let container_occupations = create_container_occupations(&schedules, input);
    (schedules, container_occupations)
}

fn eval_container_occupation(schedules: &Vec<Vec<Schedule>>, input: &Input) -> i64 {
    let mut t_in = vec![vec![0; N]; N];

    let mut container_time_range = vec![(None, None); N * N];
    let mut container_pos = vec![None; N * N];

    for ci in 0..N {
        for s in schedules[ci].iter() {
            if s.job.is_in_job() {
                let (i, j) = input.c_to_a_ij[s.job.c];
                t_in[i][j] = s.start_t;
            } else {
                container_time_range[s.job.c].1 = Some(s.start_t);
            }

            if !s.job.is_out_job() {
                container_time_range[s.job.c].0 = Some(s.end_t);
                container_pos[s.job.c] = Some(s.job.to);
            }
        }
    }

    let mut occupations = vec![vec![vec![]; N]; N];

    for c in 0..N * N {
        let (l, r) = container_time_range[c];
        let Some(l) = l else { continue };
        let r = r.unwrap();
        let p = container_pos[c].unwrap();
        occupations[p.0][p.1].push((l, r));
    }

    let mut penalty = 0;
    for i in 0..N {
        for j in 0..N {
            for k1 in 0..occupations[i][j].len() {
                for k2 in k1 + 1..occupations[i][j].len() {
                    let (l1, r1) = occupations[i][j][k1];
                    let (l2, r2) = occupations[i][j][k2];
                    let (l, r) = (l1.max(l2), r1.min(r2));
                    if r > l {
                        penalty += r - l;
                    }
                }
            }
        }
    }

    penalty as i64
}

fn eval_schedules(
    schedules: &Vec<Vec<Schedule>>,
    constraints: &Vec<Constraint>,
    jobs: &Vec<Job>,
    input: &Input,
) -> i64 {
    let mut mp = vec![(0, 0); jobs.len()];
    for i in 0..N {
        for (j, s) in schedules[i].iter().enumerate() {
            mp[s.job.idx] = (i, j);
        }
    }

    let mut constraints = constraints.clone();
    for i in 0..N {
        for j in 0..schedules[i].len() {
            let to = schedules[i][j].job.from;
            if j == 0 {
                constraints.push(Constraint::FirstJob(schedules[i][j].job.idx));
            } else {
                constraints.push(Constraint::Consecutive(
                    schedules[i][j - 1].job.idx,
                    schedules[i][j].job.idx,
                ));
            }
        }
    }

    let mut penalty = 0;
    for c in constraints {
        match c {
            Constraint::Start(prev_job_i, next_job_i) => {
                let (prev_s, next_s) = (
                    &schedules[mp[prev_job_i].0][mp[prev_job_i].1],
                    &schedules[mp[next_job_i].0][mp[next_job_i].1],
                );
                let interval = 2;
                if prev_s.start_t + interval > next_s.start_t {
                    penalty += prev_s.start_t + interval - next_s.start_t;
                    assert!(
                        penalty < 1_000_000_000_000,
                        "{:?} {:?} {:?}",
                        c,
                        prev_s,
                        next_s
                    );
                }
            }
            Constraint::End(prev_job_i, next_job_i) => {
                let (prev_s, next_s) = (
                    &schedules[mp[prev_job_i].0][mp[prev_job_i].1],
                    &schedules[mp[next_job_i].0][mp[next_job_i].1],
                );
                let interval = 2;
                if prev_s.end_t + interval > next_s.end_t {
                    penalty += prev_s.end_t + interval - next_s.end_t;
                    assert!(
                        penalty < 1_000_000_000_000,
                        "{:?} {:?} {:?} {}",
                        c,
                        prev_s,
                        next_s,
                        prev_s.end_t + interval - next_s.end_t
                    );
                }
            }
            Constraint::Consecutive(prev_job_i, next_job_i) => {
                let (prev_s, next_s) = (
                    &schedules[mp[prev_job_i].0][mp[prev_job_i].1],
                    &schedules[mp[next_job_i].0][mp[next_job_i].1],
                );
                let duration = dist(prev_s.job.to, next_s.job.from) + 2;
                if prev_s.end_t + duration > next_s.start_t {
                    penalty += prev_s.end_t + duration - next_s.start_t;
                    assert!(
                        penalty < 1_000_000_000_000,
                        "{:?} {:?} {:?}",
                        c,
                        prev_s,
                        next_s
                    );
                }
            }
            Constraint::FirstJob(job_i) => {
                let s = &schedules[mp[job_i].0][mp[job_i].1];
                let duration = dist((input.c_to_a_ij[s.job.c].0, 0), s.job.from);
                if s.start_t < duration {
                    penalty += duration - s.start_t;
                    assert!(penalty < 1_000_000_000_000, "{:?} {:?}", c, s);
                }
            }
            Constraint::Job(job_i) => {
                let s = &schedules[mp[job_i].0][mp[job_i].1];
                let duration = dist(s.job.from, s.job.to) + 2;
                assert_eq!(s.job.idx, job_i);
                if s.start_t + duration > s.end_t {
                    penalty += s.start_t + duration - s.end_t;
                    assert!(penalty < 1_000_000_000_000, "{:?}", s);
                }
            }
        }
    }

    let score = (0..N)
        .filter(|&ci| schedules[ci].len() > 0)
        .map(|ci| schedules[ci].last().unwrap().end_t)
        .max()
        .unwrap();

    (score + penalty * 1_000_000) as i64 + eval_container_occupation(schedules, input) * 1_000
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
    over_container: bool,
    crane_log: &Vec<Vec<(usize, usize)>>,
    container_occupations: &Vec<Vec<Vec<Option<usize>>>>,
) -> bool {
    let (ni, nj) = (v.0 + d.0, v.1 + d.1);
    if ni >= N || nj >= N {
        return false;
    }
    if container_occupations[t + 1][ni][nj].is_some() && !over_container {
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
) -> Option<Vec<(usize, usize)>> {
    let mut path = vec![];

    // last_t -> s.start_t の間に start_pos -> s.job.from に移動する
    let Some(path1) = path_finder.find_path(
        ci,
        last_t,
        s.start_t,
        start_pos,
        s.job.from,
        true,
        &crane_log,
        &container_occupations,
    ) else {
        return None;
    };
    path.extend(path1);
    path.push(s.job.from); // P

    // s.start_t + 1 -> s.end_t - 1の間に s.job.from -> s.job.to に移動する
    let Some(path2) = path_finder.find_path(
        ci,
        s.start_t + 1,
        s.end_t,
        s.job.from,
        s.job.to,
        ci == 0,
        &crane_log,
        &container_occupations,
    ) else {
        return None;
    };
    path.extend(path2);
    path.push(s.job.to); // Q

    Some(path)
}

const PATH_NOT_FOUND: (usize, usize) = (N, N);

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
        over_container: bool,
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
                        if !is_moveable(
                            ci,
                            t,
                            (i, j),
                            d,
                            over_container,
                            crane_log,
                            container_occupations,
                        ) {
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
                let last_t = state.crane_log[ci].len() - 1;
                let start_pos = *state.crane_log[ci].last().unwrap();
                let Some(path) = find_path_for_schedule(
                    ci,
                    last_t,
                    start_pos,
                    s,
                    &mut state.path_finder,
                    &state.crane_log,
                    &state.container_occupations,
                ) else {
                    dbg!(ci, last_t, start_pos, &s);
                    for t in last_t - 2..=s.start_t + 2 {
                        state.print_t(t);
                    }
                    state.crane_log[ci].extend(vec![PATH_NOT_FOUND; s.end_t - last_t - 1]);
                    state.crane_log[ci].push(s.job.to);
                    continue;
                };
                state.crane_log[ci].extend(path);
                assert_eq!(
                    s.end_t + 2,
                    state.crane_log[ci].len(),
                    "{:?} {:?}",
                    s,
                    state.crane_log[ci]
                );
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

    fn print_t(&self, t: usize) {
        eprintln!("t={};", t);
        for i in 0..N {
            for j in 0..N {
                if let Some(c) = self.container_occupations[t][i][j] {
                    eprint!("{:2}", c);
                } else {
                    eprint!("..");
                }
            }
            eprintln!();
        }
        eprintln!();
        eprintln!("t={};", t);
        let mut a = vec![vec![N; N]; N];
        for ci in 0..N {
            if t >= self.crane_log[ci].len() {
                continue;
            }
            let v = self.crane_log[ci][t];
            if v != PATH_NOT_FOUND {
                a[v.0][v.1] = ci;
            }
        }
        for i in 0..N {
            for j in 0..N {
                if a[i][j] != N {
                    eprint!("{}", a[i][j]);
                } else {
                    eprint!(".");
                }
            }
            eprintln!();
        }
        eprintln!()
    }
}

fn listup_jobs(input: &Input) -> Vec<Job> {
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
