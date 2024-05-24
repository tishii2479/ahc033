mod def;
mod helper;
mod lower;
mod pretask;
mod solver;
mod util;

use proconio::input;

use crate::def::*;
use crate::helper::*;
use crate::pretask::*;
use crate::solver::*;
use crate::util::*;

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
    let mut solver = Solver::new(jobs, &input);
    let iteration = 100_000;
    let moves = solver.solve(iteration, &input);
    output_ans(&moves);
}
