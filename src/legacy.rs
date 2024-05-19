fn get_move(from: (usize, usize), to: (usize, usize)) -> Vec<Move> {
    let mut v = vec![];
    if from.0 > to.0 {
        v.extend(vec![Move::Up; from.0 - to.0]);
    } else {
        v.extend(vec![Move::Down; to.0 - from.0]);
    }
    if from.1 > to.1 {
        v.extend(vec![Move::Left; from.1 - to.1]);
    } else {
        v.extend(vec![Move::Right; to.1 - from.1]);
    }
    v
}

fn solve_crane0(
    in_order: &Vec<usize>,
    in_place: &Vec<(usize, usize)>,
    a: &Vec<Vec<usize>>,
) -> Vec<Move> {
    let mut moved = vec![false; N * N]; // moved[c] := コンテナcを搬入したかどうか
    let mut used_count = vec![vec![0; N]; N];
    let mut in_count = vec![0; N]; // in_count[i] := 搬入口iから搬入した個数
    let mut out_count = vec![0; N]; // out_count[i] := 搬出口iから搬出した個数

    let mut moves = vec![];
    let mut cur_pos = (0, 0);

    for &i in in_order.iter() {
        let c = a[i][in_count[i]];
        let (pi, pj) = in_place[c];
        in_count[i] += 1;
        moved[c] = true;

        // (i, 0)から(pi, pj)に運ぶ
        moves.extend(get_move(cur_pos, (i, 0)));
        moves.push(Move::Pick);
        moves.extend(get_move((i, 0), (pi, pj)));
        moves.push(Move::Drop);
        cur_pos = (pi, pj);

        // 搬出できるコンテナを搬出する
        let (out_i, out_j) = (c / N, c % N);
        for j in out_j..N {
            let c = out_i * N + j;
            if !moved[c] || out_count[out_i] != j {
                break;
            }

            // (pi, pj) := in_place[c]
            // (pi, pj)から(out_i, 4)に運ぶ
            let (pi, pj) = in_place[c];
            used_count[pi][pj] -= 1;
            moves.extend(get_move(cur_pos, (pi, pj)));
            moves.push(Move::Pick);
            moves.extend(get_move((pi, pj), (out_i, 4)));
            moves.push(Move::Drop);
            cur_pos = (out_i, 4);

            out_count[out_i] += 1;
        }
    }

    moves
}

fn solve(
    in_order: &Vec<usize>,
    in_place: &Vec<(usize, usize)>,
    a: &Vec<Vec<usize>>,
) -> Vec<Vec<Move>> {
    let mut tasks: VecDeque<Task> = VecDeque::new();
    let mut in_count = vec![0; N];
    let mut out_count = vec![0; N];

    for &i in in_order.iter() {
        let c = a[i][in_count[i]];
        let (pi, pj) = in_place[c];
        in_count[i] += 1;
        tasks.push_back(Task {
            from: (i, 0),
            to: (pi, pj),
        });
    }

    let mut cranes: Vec<Crane> = (0..N)
        .map(|i| Crane {
            status: CraneStatus::Idle,
            task: None,
            pos: (i, 0),
            moves: vec![],
        })
        .collect();
    let mut containers: Vec<Container> = vec![
        Container {
            status: ContainerStatus::Out,
            pos: None
        };
        N * N
    ];
    let mut terminal = vec![vec![vec![TileStatus::Empty; N]; N]; T];

    // TODO: 状態の初期化

    for t in 0..1 {
        // 1. 新たに発生したタスクを列挙する
        // - 搬出できるコンテナが発生したら、タスクの先頭に加える
        for c in 0..N * N {
            if containers[c].status != ContainerStatus::Await {
                continue;
            }
            let (i, j) = (c / N, c % N);
            if out_count[i] != j {
                continue;
            }
            if let Some(cp) = containers[c].pos {
                containers[c].status = ContainerStatus::Assigned;
                tasks.push_front(Task {
                    from: cp,
                    to: (i, N - 1),
                });
            }
        }

        // 2. タスクがないクレーンにタスクを割り当てる
        for i in 0..N {
            if cranes[i].status == CraneStatus::Blown || cranes[i].task.is_some() {
                continue;
            }
            let Some(task) = tasks.pop_front() else {
                continue;
            };

            let cur_pos = cranes[i].pos;
            let moves = get_moves_for_task(&terminal, t, cur_pos, task, i == 0);
            add_moves(&mut terminal, &mut cranes[i], moves, cur_pos, i);
            cranes[i].task = Some(task);
        }

        for i in 0..N {}

        // 3. 各クレーンについての操作
        for i in 0..N {
            /*
            if cranes[i].status == CraneStatus::Blown {
                continue;
            }
            let Some(task) = cranes[i].task else {
                cranes[i].moves.push(Move::Idle);
                continue;
            };

            // 1. 目的地にいれば積み下ろしを行う
            if task.from == cranes[i].pos && cranes[i].status == CraneStatus::Moving {
                cranes[i].moves.push(Move::Pick);
                cranes[i].status = CraneStatus::Hanging;
                continue;
            } else if task.to == cranes[i].pos && cranes[i].status == CraneStatus::Hanging {
                cranes[i].moves.push(Move::Drop);
                cranes[i].status = CraneStatus::Idle;
                continue;
            }

            // 2. 目的地に近づくような操作のうち、実行可能な操作を列挙する
            assert!(
                cranes[i].status == CraneStatus::Moving || cranes[i].status == CraneStatus::Hanging
            );
            let dest = if cranes[i].status == CraneStatus::Moving {
                task.from
            } else {
                task.to
            };
            let over_container = i == 0 || cranes[i].status == CraneStatus::Moving;
            let moves = listup_valid_moves(&terminal, t, cranes[i].pos, dest, over_container);

            // 3. 2.で列挙した操作からランダムに一つ選択し、実行する
            */
        }
    }

    (0..N).map(|i| cranes[i].moves.clone()).collect()
}
