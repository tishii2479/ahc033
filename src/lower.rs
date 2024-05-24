use crate::def::*;
use crate::helper::*;

pub fn find_path_for_crane(
    ci: usize,
    crane_schedules: &Vec<Vec<Schedule>>,
    crane_log: &mut Vec<Vec<(usize, usize)>>,
    container_occupations: &Vec<Vec<Vec<(usize, usize, usize)>>>,
    path_finder: &mut PathFinder,
    debug: bool,
) -> i64 {
    let mut cost = 0;
    for s in crane_schedules[ci].iter() {
        let last_t = crane_log[ci].len() - 1;
        let start_pos = *crane_log[ci].last().unwrap();

        // last_t -> s.start_t の間に start_pos -> s.job.from に移動する
        let (path1, cost1) = path_finder.find_path(
            ci,
            last_t,
            s.start_t,
            start_pos,
            s.job.from,
            true,
            &crane_log,
            &container_occupations,
        );
        if debug && cost1 > 0 {
            dbg!(ci, s, &path1);
        }
        crane_log[ci].extend(path1);
        crane_log[ci].push(s.job.from); // P
        cost += cost1;

        // s.start_t + 1 -> s.end_t - 1の間に s.job.from -> s.job.to に移動する
        let (path2, cost2) = path_finder.find_path(
            ci,
            s.start_t + 1,
            s.end_t,
            s.job.from,
            s.job.to,
            ci == 0,
            &crane_log,
            &container_occupations,
        );
        if debug && cost2 > 0 {
            dbg!(ci, s, &path2);
        }
        crane_log[ci].extend(path2);
        crane_log[ci].push(s.job.to); // Q
        cost += cost2;

        assert_eq!(s.end_t + 2, crane_log[ci].len());
    }

    cost
}

pub fn optimize_lower_level(
    crane_schedules: &Vec<Vec<Schedule>>,
    container_occupations: &Vec<Vec<Vec<(usize, usize, usize)>>>,
) -> (Vec<Vec<(usize, usize)>>, i64) {
    let mut crane_log: Vec<Vec<(usize, usize)>> = (0..N).map(|ci| vec![(ci, 0)]).collect();
    let mut path_finder = PathFinder::new();
    let mut score = 0;

    for _t in 0..3 {
        score = 0;
        for ci in (0..N).rev() {
            crane_log[ci].clear();
            crane_log[ci].push((ci, 0));
            score += find_path_for_crane(
                ci,
                crane_schedules,
                &mut crane_log,
                container_occupations,
                &mut path_finder,
                false,
            );
        }
        // eprintln!("score = {:?}", score);
    }
    // eprintln!();

    (crane_log, score)
}
