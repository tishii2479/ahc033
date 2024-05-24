use crate::def::*;
use crate::helper::*;

pub fn find_path_for_crane(
    ci: usize,
    jobs: &Vec<Job>,
    crane_schedules: &Vec<Vec<Schedule>>,
    crane_log: &mut Vec<Vec<(usize, usize)>>,
    container_occupations: &Vec<Vec<Vec<(usize, usize, usize)>>>,
    path_finder: &mut PathFinder,
) -> i64 {
    let mut cost = 0;
    for s in crane_schedules[ci].iter() {
        let last_t = crane_log[ci].len() - 1;
        let start_pos = *crane_log[ci].last().unwrap();

        // last_t -> s.start_t の間に start_pos -> jobs[s.job_idx].from に移動する
        let (path1, cost1) = path_finder.find_path(
            ci,
            last_t,
            s.start_t,
            start_pos,
            jobs[s.job_idx].from,
            true,
            &crane_log,
            &container_occupations,
        );
        crane_log[ci].extend(path1);
        crane_log[ci].push(jobs[s.job_idx].from); // P
        cost += cost1;

        // s.start_t + 1 -> s.end_t - 1の間に jobs[s.job_idx].from -> jobs[s.job_idx].to に移動する
        let (path2, cost2) = path_finder.find_path(
            ci,
            s.start_t + 1,
            s.end_t,
            jobs[s.job_idx].from,
            jobs[s.job_idx].to,
            ci == 0,
            &crane_log,
            &container_occupations,
        );
        crane_log[ci].extend(path2);
        crane_log[ci].push(jobs[s.job_idx].to); // Q
        cost += cost2;

        assert_eq!(s.end_t + 2, crane_log[ci].len());
    }

    cost
}

pub fn optimize_lower_level(
    jobs: &Vec<Job>,
    crane_schedules: &Vec<Vec<Schedule>>,
    container_occupations: &Vec<Vec<Vec<(usize, usize, usize)>>>,
    path_finder: &mut PathFinder,
) -> (Vec<Vec<(usize, usize)>>, i64) {
    let mut crane_log: Vec<Vec<(usize, usize)>> = (0..N).map(|ci| vec![(ci, 0)]).collect();
    let mut score = 0;

    for _t in 0..3 {
        score = 0;
        for ci in (0..N).rev() {
            crane_log[ci].clear();
            crane_log[ci].push((ci, 0));
            score += find_path_for_crane(
                ci,
                jobs,
                crane_schedules,
                &mut crane_log,
                container_occupations,
                path_finder,
            );
        }
        // eprintln!("score = {:?}", score);
    }
    // eprintln!();

    (crane_log, score)
}
