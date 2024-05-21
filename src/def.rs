pub const N: usize = 5;

pub const MAX_T: usize = 1000;
pub const D: [(usize, usize); 5] = [(0, 0), (0, 1), (1, 0), (0, !0), (!0, 0)];
pub const REV_D: [(usize, usize); 5] = [(0, 0), (0, !0), (!0, 0), (0, 1), (1, 0)];
pub const D_MOVE: [Move; 5] = [Move::Idle, Move::Right, Move::Down, Move::Left, Move::Up];

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

    pub fn to_d(&self) -> (usize, usize) {
        match self {
            Move::Up => (!0, 0),
            Move::Down => (1, 0),
            Move::Left => (0, !0),
            Move::Right => (0, 1),
            _ => (0, 0),
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
