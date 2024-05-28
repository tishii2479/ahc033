use itertools::iproduct;

pub const TIME_LIMIT: f64 = 2.9;
pub const N: usize = 5;

pub const MAX_T: usize = 1000;
pub const D: [(usize, usize); 5] = [(0, 0), (0, 1), (1, 0), (0, !0), (!0, 0)];
pub const D_MOVE: [Move; 5] = [Move::Idle, Move::Right, Move::Down, Move::Left, Move::Up];

#[derive(Clone, Copy, Debug)]
pub enum Violation {
    PickUp(usize),
    Carry(usize),
}

#[derive(Clone, Copy, Debug)]
pub enum Constraint {
    Start(usize, usize),
    End(usize, usize),
    FirstJob(usize),
    Consecutive(usize, usize),
    Job(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    pub start_t: usize,
    pub end_t: usize,
    pub job_idx: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Job {
    pub idx: usize,
    pub c: usize,
    pub from: (usize, usize),
    pub to: (usize, usize),
}

impl Job {
    pub fn is_in_job(&self) -> bool {
        self.from.1 == 0
    }

    pub fn is_out_job(&self) -> bool {
        self.to.1 == N - 1
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Move {
    Pick,  // P
    Drop,  // Q
    Up,    // U
    Down,  // D
    Left,  // L
    Right, // R
    Idle,  // .
    Blow,  // B
}

impl Move {
    pub fn to_str(&self) -> &str {
        match self {
            Move::Pick => "P",
            Move::Drop => "Q",
            Move::Up => "U",
            Move::Down => "D",
            Move::Left => "L",
            Move::Right => "R",
            Move::Idle => ".",
            Move::Blow => "B",
        }
    }

    pub fn from_d(d: (usize, usize)) -> Move {
        for i in 0..D.len() {
            if D[i] == d {
                return D_MOVE[i];
            }
        }
        unreachable!("{:?}", d);
    }
}

pub struct Input {
    pub a: Vec<Vec<usize>>,
    pub c_to_a_ij: Vec<(usize, usize)>,
}

impl Input {
    pub fn new(a: Vec<Vec<usize>>) -> Input {
        let mut c_to_a_ij = vec![(0, 0); N * N];
        for (i, j) in iproduct!(0..N, 0..N) {
            c_to_a_ij[a[i][j]] = (i, j);
        }
        Input { a, c_to_a_ij }
    }
}
