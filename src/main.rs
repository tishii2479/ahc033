mod def;
mod helper;
mod lower;
mod pretask;
mod upper;
mod util;

use proconio::input;

use crate::def::*;
use crate::helper::*;
use crate::lower::*;
use crate::pretask::*;
use crate::upper::*;
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
    eprintln!("{}", time::elapsed_seconds());
    let (crane_schedules, container_occupations) = optimize_upper_level(jobs, &input);
    eprintln!("{}", time::elapsed_seconds());
    for ci in 0..N {
        eprintln!("ci={ci}:");
        for s in crane_schedules[ci].iter() {
            eprintln!("{:?}", s);
        }
    }
    let moves = optimize_lower_level(crane_schedules, container_occupations);
    output_ans(&moves);
}
