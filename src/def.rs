pub const N: usize = 5;

pub const T: usize = 100;
pub const D: [(usize, usize); 5] = [(0, 0), (0, 1), (1, 0), (0, !0), (!0, 0)];
pub const REV_D: [(usize, usize); 5] = [(0, 0), (0, !0), (!0, 0), (0, 1), (1, 0)];
pub const D_MOVE: [Move; 5] = [Move::Idle, Move::Right, Move::Down, Move::Left, Move::Up];

#[derive(Clone, Copy)]
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
}
