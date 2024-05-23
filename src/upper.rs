use crate::def::*;
use crate::helper::*;
use crate::util::*;

pub fn optimize_upper_level(
    jobs: Vec<Job>,
    input: &Input,
) -> (Vec<Vec<Schedule>>, Vec<Vec<Vec<Option<usize>>>>) {
    let constraints = create_constraints(&jobs, input);
    let mut assigned_jobs: Vec<Vec<Job>> = vec![vec![]; N];
    for job in jobs.iter() {
        assigned_jobs[rnd::gen_index(N)].push(job.clone());
        // assigned_jobs[0].push(job.clone());
    }

    let mut schedules = jobs_to_schedules(assigned_jobs);
    let mut cur_score = eval_schedules(&schedules, &constraints, &jobs);
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
            let new_score = eval_schedules(&schedules, &constraints, &jobs);

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

            let new_score = eval_schedules(&schedules, &constraints, &jobs);
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

            let new_score = eval_schedules(&schedules, &constraints, &jobs);
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

            let new_score = eval_schedules(&schedules, &constraints, &jobs);
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

            let new_score = eval_schedules(&schedules, &constraints, &jobs);
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
    assert!(eval_schedules(&schedules, &constraints, &jobs) < 1_000);
    let container_occupations = create_container_occupations_tensor(&schedules, input);
    (schedules, container_occupations)
}

fn eval_schedules(
    schedules: &Vec<Vec<Schedule>>,
    constraints: &Vec<Constraint>,
    jobs: &Vec<Job>,
) -> i64 {
    let raw_score = (0..N)
        .filter(|&ci| schedules[ci].len() > 0)
        .map(|ci| schedules[ci].last().unwrap().end_t)
        .max()
        .unwrap() as i64;
    // TODO: 必要なところまで計算する
    let constraint_penalty = eval_constraints(schedules, &constraints, jobs);

    let container_occupations = create_container_occupations(schedules);
    let container_occupation_penalty = eval_container_occupation(&container_occupations);

    let schedule_feasibility_penalty = eval_schedule_feasibility(schedules, &container_occupations);

    raw_score
        + constraint_penalty * 1_000_000
        + container_occupation_penalty * 1_000
        + schedule_feasibility_penalty * 1_000
}

fn eval_container_occupation(occupations: &Vec<Vec<Vec<(usize, usize, usize)>>>) -> i64 {
    let mut penalty = 0;
    for i in 0..N {
        for j in 0..N {
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
    }
    penalty as i64
}

fn eval_constraints(
    schedules: &Vec<Vec<Schedule>>,
    constraints: &Vec<Constraint>,
    jobs: &Vec<Job>,
) -> i64 {
    // TODO: .clone()しない
    let mut constraints = constraints.clone();
    for i in 0..N {
        for j in 0..schedules[i].len() {
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

    let mut mp = vec![(0, 0); jobs.len()];
    for i in 0..N {
        for (j, s) in schedules[i].iter().enumerate() {
            mp[s.job.idx] = (i, j);
        }
    }

    let mut penalty = 0;
    for &c in constraints.iter() {
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
                let interval = dist(prev_s.job.to, next_s.job.from) + 1;
                if prev_s.end_t + interval > next_s.start_t {
                    penalty += prev_s.end_t + interval - next_s.start_t;
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
                let interval = dist((mp[job_i].0, 0), s.job.from);
                if s.start_t < interval {
                    penalty += interval - s.start_t;
                    assert!(penalty < 1_000_000_000_000, "{:?} {:?}", c, s);
                }
            }
            Constraint::Job(job_i) => {
                let s = &schedules[mp[job_i].0][mp[job_i].1];
                let interval = dist(s.job.from, s.job.to) + 2;
                assert_eq!(s.job.idx, job_i);
                if s.start_t + interval > s.end_t {
                    penalty += s.start_t + interval - s.end_t;
                    assert!(penalty < 1_000_000_000_000, "{:?}", s);
                }
            }
        }
    }

    penalty as i64
}

fn eval_schedule_feasibility(
    schedules: &Vec<Vec<Schedule>>,
    container_occupations: &Vec<Vec<Vec<(usize, usize, usize)>>>,
) -> i64 {
    let mut penalty = 0;
    for ci in 0..N {
        for s in schedules[ci].iter() {
            // s.start_t + 1 -> s.end_tにs.job.from -> s.job.toへの経路が存在するかどうか
            // 存在しない場合、衝突したコンテナの個数をペナルティとして加える
        }
    }
    penalty
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
