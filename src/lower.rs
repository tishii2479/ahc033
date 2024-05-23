use crate::def::*;
use crate::helper::*;
use crate::util::*;

pub fn optimize_lower_level(
    crane_schedules: Vec<Vec<Schedule>>,
    container_occupations: Vec<Vec<Vec<Option<usize>>>>,
) -> Vec<Vec<Move>> {
    let mut state = State::initialize(crane_schedules, container_occupations);
    for _t in 0..1 {
        let ci = rnd::gen_index(N);
        let j = rnd::gen_index(state.path_info[ci].len());
        let ((l, r), prev_cost, over_container) = state.path_info[ci][j];
        let (path, cost) = state.path_finder.find_path(
            ci,
            l,
            r,
            state.crane_log[ci][l],
            state.crane_log[ci][r],
            over_container,
            &state.crane_log,
            &state.container_occupations,
        );
        if cost > 0 {
            for t in l - 3..=r {
                state.print_t(t);
            }
            dbg!(
                l,
                r,
                &path,
                cost,
                ci,
                over_container,
                (l..=r)
                    .map(|i| state.crane_log[ci][i])
                    .collect::<Vec<(usize, usize)>>()
            );
        }
        let new_score = state.score - prev_cost + cost;
        if new_score <= state.score {
            // eprintln!("[{_t}] {} -> {}", state.score, new_score);
            for t in l + 1..=r {
                state.crane_log[ci][t] = path[t - (l + 1)];
                state.path_info[ci][j].1 = cost;
            }
            state.score = new_score;
        }
    }
    state.to_moves()
}

struct State {
    crane_log: Vec<Vec<(usize, usize)>>,
    crane_schedules: Vec<Vec<Schedule>>,
    path_info: Vec<Vec<((usize, usize), i64, bool)>>,
    container_occupations: Vec<Vec<Vec<Option<usize>>>>,
    path_finder: PathFinder,
    score: i64,
}

impl State {
    fn initialize(
        crane_schedules: Vec<Vec<Schedule>>,
        container_occupations: Vec<Vec<Vec<Option<usize>>>>,
    ) -> State {
        let mut state = State {
            crane_log: (0..N).map(|ci| vec![(ci, 0)]).collect(),
            crane_schedules,
            path_info: vec![vec![]; N],
            container_occupations,
            path_finder: PathFinder::new(),
            score: 0,
        };
        for ci in (0..N).rev() {
            for s in state.crane_schedules[ci].iter() {
                let last_t = state.crane_log[ci].len() - 1;
                let start_pos = *state.crane_log[ci].last().unwrap();

                // last_t -> s.start_t の間に start_pos -> s.job.from に移動する
                let (path1, cost1) = state.path_finder.find_path(
                    ci,
                    last_t,
                    s.start_t,
                    start_pos,
                    s.job.from,
                    true,
                    &state.crane_log,
                    &state.container_occupations,
                );
                state.crane_log[ci].extend(path1);
                state.crane_log[ci].push(s.job.from); // P
                state.path_info[ci].push(((last_t, s.start_t), cost1, true));
                state.score += cost1;

                // s.start_t + 1 -> s.end_t - 1の間に s.job.from -> s.job.to に移動する
                let (path2, cost2) = state.path_finder.find_path(
                    ci,
                    s.start_t + 1,
                    s.end_t,
                    s.job.from,
                    s.job.to,
                    ci == 0,
                    &state.crane_log,
                    &state.container_occupations,
                );
                state.crane_log[ci].extend(path2);
                state.crane_log[ci].push(s.job.to); // Q
                state.path_info[ci].push(((s.start_t + 1, s.end_t), cost2, ci == 0));
                state.score += cost2;

                assert_eq!(s.end_t + 2, state.crane_log[ci].len(),);
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
                    eprint!(" {:2}", c);
                } else {
                    eprint!("  .");
                }
            }
            eprintln!();
        }
        eprintln!();
        // eprintln!("t={};", t);
        // let mut a = vec![vec![N; N]; N];
        // for ci in 0..N {
        //     if t >= self.crane_log[ci].len() {
        //         continue;
        //     }
        //     let v = self.crane_log[ci][t];
        //     a[v.0][v.1] = ci;
        // }
        // for i in 0..N {
        //     for j in 0..N {
        //         if a[i][j] != N {
        //             eprint!("{}", a[i][j]);
        //         } else {
        //             eprint!(".");
        //         }
        //     }
        //     eprintln!();
        // }
        // eprintln!()
    }
}
