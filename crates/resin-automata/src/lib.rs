//! Cellular automata for procedural pattern generation.
//!
//! Implements both 1D (elementary) and 2D cellular automata.
//!
//! # Example
//!
//! ```
//! use rhizome_resin_automata::{ElementaryCA, GameOfLife};
//!
//! // 1D: Rule 30
//! let mut ca = ElementaryCA::new(100, 30);
//! ca.randomize(12345);
//! ca.step();
//!
//! // 2D: Game of Life
//! let mut life = GameOfLife::life(50, 50);
//! life.randomize(12345, 0.3);
//! life.step();
//! ```

/// 1D Elementary Cellular Automaton.
///
/// Implements Wolfram's elementary cellular automata with 256 possible rules.
#[derive(Debug, Clone)]
pub struct ElementaryCA {
    /// Current cell states (true = alive).
    cells: Vec<bool>,
    /// Rule number (0-255).
    rule: u8,
    /// Wrap around at edges.
    wrap: bool,
}

impl ElementaryCA {
    /// Creates a new 1D cellular automaton.
    pub fn new(width: usize, rule: u8) -> Self {
        Self {
            cells: vec![false; width],
            rule,
            wrap: true,
        }
    }

    /// Sets whether the grid wraps at edges.
    pub fn set_wrap(&mut self, wrap: bool) {
        self.wrap = wrap;
    }

    /// Returns the width of the automaton.
    pub fn width(&self) -> usize {
        self.cells.len()
    }

    /// Returns the rule number.
    pub fn rule(&self) -> u8 {
        self.rule
    }

    /// Sets the rule number.
    pub fn set_rule(&mut self, rule: u8) {
        self.rule = rule;
    }

    /// Gets the state of a cell.
    pub fn get(&self, x: usize) -> bool {
        self.cells.get(x).copied().unwrap_or(false)
    }

    /// Sets the state of a cell.
    pub fn set(&mut self, x: usize, alive: bool) {
        if x < self.cells.len() {
            self.cells[x] = alive;
        }
    }

    /// Clears all cells.
    pub fn clear(&mut self) {
        self.cells.fill(false);
    }

    /// Sets a single cell in the center (common starting condition).
    pub fn set_center(&mut self) {
        self.clear();
        let center = self.cells.len() / 2;
        self.cells[center] = true;
    }

    /// Randomizes the cells.
    pub fn randomize(&mut self, seed: u64) {
        let mut rng = SimpleRng::new(seed);
        for cell in &mut self.cells {
            *cell = rng.next_bool();
        }
    }

    /// Advances the automaton by one step.
    pub fn step(&mut self) {
        let width = self.cells.len();
        let mut next = vec![false; width];

        for i in 0..width {
            let left = if i == 0 {
                if self.wrap {
                    self.cells[width - 1]
                } else {
                    false
                }
            } else {
                self.cells[i - 1]
            };

            let center = self.cells[i];

            let right = if i == width - 1 {
                if self.wrap { self.cells[0] } else { false }
            } else {
                self.cells[i + 1]
            };

            // Convert neighborhood to index (0-7)
            let index = (left as u8) << 2 | (center as u8) << 1 | (right as u8);

            // Look up new state from rule
            next[i] = (self.rule >> index) & 1 == 1;
        }

        self.cells = next;
    }

    /// Advances multiple steps.
    pub fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Returns a reference to the cell states.
    pub fn cells(&self) -> &[bool] {
        &self.cells
    }

    /// Generates a 2D pattern by running the CA for multiple steps.
    ///
    /// Returns a 2D grid where each row is a generation.
    pub fn generate_pattern(&mut self, generations: usize) -> Vec<Vec<bool>> {
        let mut pattern = Vec::with_capacity(generations);

        for _ in 0..generations {
            pattern.push(self.cells.clone());
            self.step();
        }

        pattern
    }
}

/// 2D Cellular Automaton with configurable rules.
#[derive(Debug, Clone)]
pub struct CellularAutomaton2D {
    /// Cell states.
    cells: Vec<Vec<bool>>,
    /// Width.
    width: usize,
    /// Height.
    height: usize,
    /// Birth rule (number of neighbors that cause birth).
    birth: Vec<u8>,
    /// Survival rule (number of neighbors that allow survival).
    survive: Vec<u8>,
    /// Wrap around at edges.
    wrap: bool,
}

impl CellularAutomaton2D {
    /// Creates a new 2D cellular automaton with custom rules.
    ///
    /// Rules are specified as birth/survival counts (e.g., B3/S23 for Game of Life).
    pub fn new(width: usize, height: usize, birth: &[u8], survive: &[u8]) -> Self {
        Self {
            cells: vec![vec![false; width]; height],
            width,
            height,
            birth: birth.to_vec(),
            survive: survive.to_vec(),
            wrap: true,
        }
    }

    /// Sets whether the grid wraps at edges.
    pub fn set_wrap(&mut self, wrap: bool) {
        self.wrap = wrap;
    }

    /// Returns the width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Gets the state of a cell.
    pub fn get(&self, x: usize, y: usize) -> bool {
        self.cells
            .get(y)
            .and_then(|row| row.get(x))
            .copied()
            .unwrap_or(false)
    }

    /// Sets the state of a cell.
    pub fn set(&mut self, x: usize, y: usize, alive: bool) {
        if y < self.height && x < self.width {
            self.cells[y][x] = alive;
        }
    }

    /// Clears all cells.
    pub fn clear(&mut self) {
        for row in &mut self.cells {
            row.fill(false);
        }
    }

    /// Randomizes cells with given density (0.0 to 1.0).
    pub fn randomize(&mut self, seed: u64, density: f32) {
        let mut rng = SimpleRng::new(seed);
        for row in &mut self.cells {
            for cell in row {
                *cell = rng.next_f32() < density;
            }
        }
    }

    /// Counts alive neighbors for a cell.
    fn count_neighbors(&self, x: usize, y: usize) -> u8 {
        let mut count = 0u8;

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let nx = if self.wrap {
                    ((x as i32 + dx).rem_euclid(self.width as i32)) as usize
                } else {
                    let nx = x as i32 + dx;
                    if nx < 0 || nx >= self.width as i32 {
                        continue;
                    }
                    nx as usize
                };

                let ny = if self.wrap {
                    ((y as i32 + dy).rem_euclid(self.height as i32)) as usize
                } else {
                    let ny = y as i32 + dy;
                    if ny < 0 || ny >= self.height as i32 {
                        continue;
                    }
                    ny as usize
                };

                if self.cells[ny][nx] {
                    count += 1;
                }
            }
        }

        count
    }

    /// Advances the automaton by one step.
    pub fn step(&mut self) {
        let mut next = vec![vec![false; self.width]; self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                let neighbors = self.count_neighbors(x, y);
                let alive = self.cells[y][x];

                next[y][x] = if alive {
                    self.survive.contains(&neighbors)
                } else {
                    self.birth.contains(&neighbors)
                };
            }
        }

        self.cells = next;
    }

    /// Advances multiple steps.
    pub fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Returns a reference to the cell grid.
    pub fn cells(&self) -> &Vec<Vec<bool>> {
        &self.cells
    }

    /// Counts total alive cells.
    pub fn population(&self) -> usize {
        self.cells
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&c| c)
            .count()
    }
}

/// Conway's Game of Life (B3/S23).
pub type GameOfLife = CellularAutomaton2D;

impl GameOfLife {
    /// Creates a new Game of Life grid.
    pub fn life(width: usize, height: usize) -> Self {
        Self::new(width, height, &[3], &[2, 3])
    }
}

/// Common 2D CA rule presets.
pub mod rules {
    /// Game of Life (B3/S23) - classic rules.
    pub const LIFE: (&[u8], &[u8]) = (&[3], &[2, 3]);

    /// HighLife (B36/S23) - similar to Life but with more action.
    pub const HIGH_LIFE: (&[u8], &[u8]) = (&[3, 6], &[2, 3]);

    /// Seeds (B2/S) - explosive growth.
    pub const SEEDS: (&[u8], &[u8]) = (&[2], &[]);

    /// Day & Night (B3678/S34678) - symmetric rules.
    pub const DAY_NIGHT: (&[u8], &[u8]) = (&[3, 6, 7, 8], &[3, 4, 6, 7, 8]);

    /// Maze (B3/S12345) - creates maze-like patterns.
    pub const MAZE: (&[u8], &[u8]) = (&[3], &[1, 2, 3, 4, 5]);

    /// Diamoeba (B35678/S5678) - amoeba-like growth.
    pub const DIAMOEBA: (&[u8], &[u8]) = (&[3, 5, 6, 7, 8], &[5, 6, 7, 8]);

    /// Replicator (B1357/S1357) - patterns replicate.
    pub const REPLICATOR: (&[u8], &[u8]) = (&[1, 3, 5, 7], &[1, 3, 5, 7]);
}

/// Common 1D CA rules.
pub mod elementary_rules {
    /// Rule 30 - chaotic, used for random number generation.
    pub const RULE_30: u8 = 30;

    /// Rule 90 - Sierpinski triangle.
    pub const RULE_90: u8 = 90;

    /// Rule 110 - Turing complete.
    pub const RULE_110: u8 = 110;

    /// Rule 184 - traffic flow model.
    pub const RULE_184: u8 = 184;

    /// Rule 250 - simple growth.
    pub const RULE_250: u8 = 250;
}

// ============================================================================
// Maze Generation
// ============================================================================

/// A 2D maze grid.
///
/// Cells are either passages (false) or walls (true).
/// Odd coordinates are cells, even coordinates are walls/borders.
#[derive(Debug, Clone)]
pub struct Maze {
    /// Grid data (true = wall, false = passage).
    grid: Vec<Vec<bool>>,
    /// Width in cells (not including walls).
    width: usize,
    /// Height in cells (not including walls).
    height: usize,
}

impl Maze {
    /// Creates a new maze filled with walls.
    pub fn new(width: usize, height: usize) -> Self {
        // Grid dimensions include walls between cells
        let grid_width = width * 2 + 1;
        let grid_height = height * 2 + 1;

        Self {
            grid: vec![vec![true; grid_width]; grid_height],
            width,
            height,
        }
    }

    /// Returns the maze width in cells.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the maze height in cells.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the grid width (including walls).
    pub fn grid_width(&self) -> usize {
        self.width * 2 + 1
    }

    /// Returns the grid height (including walls).
    pub fn grid_height(&self) -> usize {
        self.height * 2 + 1
    }

    /// Returns true if the given grid position is a wall.
    pub fn is_wall(&self, x: usize, y: usize) -> bool {
        self.grid
            .get(y)
            .and_then(|row| row.get(x))
            .copied()
            .unwrap_or(true)
    }

    /// Returns the grid as a 2D boolean array.
    pub fn grid(&self) -> &Vec<Vec<bool>> {
        &self.grid
    }

    /// Carves a passage at cell coordinates.
    fn carve_cell(&mut self, cx: usize, cy: usize) {
        let gx = cx * 2 + 1;
        let gy = cy * 2 + 1;
        if gy < self.grid.len() && gx < self.grid[0].len() {
            self.grid[gy][gx] = false;
        }
    }

    /// Carves a passage between two adjacent cells.
    fn carve_between(&mut self, cx1: usize, cy1: usize, cx2: usize, cy2: usize) {
        let gx = cx1 + cx2 + 1;
        let gy = cy1 + cy2 + 1;
        if gy < self.grid.len() && gx < self.grid[0].len() {
            self.grid[gy][gx] = false;
        }
    }

    /// Converts the maze to a string representation.
    pub fn to_string_art(&self) -> String {
        let mut result = String::new();
        for row in &self.grid {
            for &cell in row {
                result.push(if cell { '█' } else { ' ' });
            }
            result.push('\n');
        }
        result
    }
}

/// Maze generation algorithm type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MazeAlgorithm {
    /// Recursive backtracker (depth-first search).
    /// Creates long, winding passages with few dead ends.
    #[default]
    RecursiveBacktracker,
    /// Prim's algorithm (minimum spanning tree).
    /// Creates mazes with many short dead ends.
    Prims,
    /// Kruskal's algorithm (minimum spanning tree).
    /// Creates mazes with uniform passage distribution.
    Kruskals,
    /// Eller's algorithm (row-by-row).
    /// Memory-efficient, can generate infinite mazes.
    Ellers,
    /// Binary tree algorithm.
    /// Simple and fast, but has diagonal bias.
    BinaryTree,
    /// Sidewinder algorithm.
    /// Similar to binary tree but with horizontal bias.
    Sidewinder,
}

/// Generates a maze using the specified algorithm.
///
/// # Example
///
/// ```
/// use rhizome_resin_automata::{generate_maze, MazeAlgorithm};
///
/// let maze = generate_maze(10, 10, MazeAlgorithm::RecursiveBacktracker, 12345);
/// assert_eq!(maze.width(), 10);
/// assert_eq!(maze.height(), 10);
/// ```
pub fn generate_maze(width: usize, height: usize, algorithm: MazeAlgorithm, seed: u64) -> Maze {
    match algorithm {
        MazeAlgorithm::RecursiveBacktracker => generate_recursive_backtracker(width, height, seed),
        MazeAlgorithm::Prims => generate_prims(width, height, seed),
        MazeAlgorithm::Kruskals => generate_kruskals(width, height, seed),
        MazeAlgorithm::Ellers => generate_ellers(width, height, seed),
        MazeAlgorithm::BinaryTree => generate_binary_tree(width, height, seed),
        MazeAlgorithm::Sidewinder => generate_sidewinder(width, height, seed),
    }
}

/// Recursive backtracker maze generation (depth-first search).
fn generate_recursive_backtracker(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = SimpleRng::new(seed);
    let mut visited = vec![vec![false; width]; height];
    let mut stack = Vec::new();

    // Start at (0, 0)
    let start_x = 0;
    let start_y = 0;
    visited[start_y][start_x] = true;
    maze.carve_cell(start_x, start_y);
    stack.push((start_x, start_y));

    while let Some(&(cx, cy)) = stack.last() {
        // Get unvisited neighbors
        let mut neighbors = Vec::new();

        if cx > 0 && !visited[cy][cx - 1] {
            neighbors.push((cx - 1, cy));
        }
        if cx < width - 1 && !visited[cy][cx + 1] {
            neighbors.push((cx + 1, cy));
        }
        if cy > 0 && !visited[cy - 1][cx] {
            neighbors.push((cx, cy - 1));
        }
        if cy < height - 1 && !visited[cy + 1][cx] {
            neighbors.push((cx, cy + 1));
        }

        if neighbors.is_empty() {
            stack.pop();
        } else {
            // Choose random neighbor
            let idx = (rng.next_u64() as usize) % neighbors.len();
            let (nx, ny) = neighbors[idx];

            visited[ny][nx] = true;
            maze.carve_cell(nx, ny);
            maze.carve_between(cx, cy, nx, ny);
            stack.push((nx, ny));
        }
    }

    maze
}

/// Prim's algorithm maze generation.
fn generate_prims(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = SimpleRng::new(seed);
    let mut in_maze = vec![vec![false; width]; height];
    let mut frontier = Vec::new();

    // Start at (0, 0)
    in_maze[0][0] = true;
    maze.carve_cell(0, 0);

    // Add neighbors to frontier
    if width > 1 {
        frontier.push((1, 0, 0, 0));
    }
    if height > 1 {
        frontier.push((0, 1, 0, 0));
    }

    while !frontier.is_empty() {
        // Pick random frontier cell
        let idx = (rng.next_u64() as usize) % frontier.len();
        let (fx, fy, px, py) = frontier.swap_remove(idx);

        if in_maze[fy][fx] {
            continue;
        }

        // Add to maze
        in_maze[fy][fx] = true;
        maze.carve_cell(fx, fy);
        maze.carve_between(px, py, fx, fy);

        // Add new frontier cells
        let neighbors = [
            (fx.wrapping_sub(1), fy),
            (fx + 1, fy),
            (fx, fy.wrapping_sub(1)),
            (fx, fy + 1),
        ];

        for (nx, ny) in neighbors {
            if nx < width && ny < height && !in_maze[ny][nx] {
                frontier.push((nx, ny, fx, fy));
            }
        }
    }

    maze
}

/// Kruskal's algorithm maze generation.
fn generate_kruskals(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = SimpleRng::new(seed);

    // Initialize cells
    for y in 0..height {
        for x in 0..width {
            maze.carve_cell(x, y);
        }
    }

    // Create list of all walls between cells
    let mut walls = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if x < width - 1 {
                walls.push((x, y, x + 1, y));
            }
            if y < height - 1 {
                walls.push((x, y, x, y + 1));
            }
        }
    }

    // Shuffle walls
    for i in (1..walls.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        walls.swap(i, j);
    }

    // Union-find structure
    let mut parent: Vec<usize> = (0..width * height).collect();

    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    fn union(parent: &mut [usize], i: usize, j: usize) {
        let pi = find(parent, i);
        let pj = find(parent, j);
        if pi != pj {
            parent[pi] = pj;
        }
    }

    // Process walls
    for (x1, y1, x2, y2) in walls {
        let i1 = y1 * width + x1;
        let i2 = y2 * width + x2;

        if find(&mut parent, i1) != find(&mut parent, i2) {
            maze.carve_between(x1, y1, x2, y2);
            union(&mut parent, i1, i2);
        }
    }

    maze
}

/// Eller's algorithm maze generation (row-by-row).
fn generate_ellers(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = SimpleRng::new(seed);

    // Set IDs for each cell in current row
    let mut set_id: Vec<usize> = (0..width).collect();
    let mut next_set = width;

    for y in 0..height {
        // Carve all cells in this row
        for x in 0..width {
            maze.carve_cell(x, y);
        }

        // Randomly join adjacent cells in different sets
        for x in 0..width - 1 {
            if set_id[x] != set_id[x + 1] && (y == height - 1 || rng.next_bool()) {
                let old_set = set_id[x + 1];
                let new_set = set_id[x];
                for i in 0..width {
                    if set_id[i] == old_set {
                        set_id[i] = new_set;
                    }
                }
                maze.carve_between(x, y, x + 1, y);
            }
        }

        // If not last row, create vertical connections
        if y < height - 1 {
            // Group cells by set
            let mut sets: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();
            for (x, &s) in set_id.iter().enumerate() {
                sets.entry(s).or_default().push(x);
            }

            // Each set must have at least one vertical connection
            let mut has_down = vec![false; width];
            for cells in sets.values() {
                // Randomly select cells for vertical connections (at least one)
                let mut made_connection = false;
                for &x in cells {
                    if rng.next_bool() || (!made_connection && x == *cells.last().unwrap()) {
                        has_down[x] = true;
                        maze.carve_between(x, y, x, y + 1);
                        made_connection = true;
                    }
                }
            }

            // Create new set IDs for next row
            for x in 0..width {
                if !has_down[x] {
                    set_id[x] = next_set;
                    next_set += 1;
                }
            }
        }
    }

    maze
}

/// Binary tree maze generation.
fn generate_binary_tree(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = SimpleRng::new(seed);

    for y in 0..height {
        for x in 0..width {
            maze.carve_cell(x, y);

            // Choose to carve north or west (if possible)
            let can_north = y > 0;
            let can_west = x > 0;

            if can_north && can_west {
                if rng.next_bool() {
                    maze.carve_between(x, y, x, y - 1);
                } else {
                    maze.carve_between(x, y, x - 1, y);
                }
            } else if can_north {
                maze.carve_between(x, y, x, y - 1);
            } else if can_west {
                maze.carve_between(x, y, x - 1, y);
            }
        }
    }

    maze
}

/// Sidewinder maze generation.
fn generate_sidewinder(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = SimpleRng::new(seed);

    for y in 0..height {
        let mut run_start = 0;

        for x in 0..width {
            maze.carve_cell(x, y);

            let at_east_boundary = x == width - 1;
            let at_north_boundary = y == 0;

            let should_close = at_east_boundary || (!at_north_boundary && rng.next_bool());

            if should_close {
                if !at_north_boundary {
                    // Carve north from a random cell in the run
                    let carve_x = run_start + (rng.next_u64() as usize) % (x - run_start + 1);
                    maze.carve_between(carve_x, y, carve_x, y - 1);
                }
                run_start = x + 1;
            } else {
                // Carve east
                maze.carve_between(x, y, x + 1, y);
            }
        }
    }

    maze
}

/// Simple RNG for cellular automata and maze generation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementary_ca_creation() {
        let ca = ElementaryCA::new(100, 30);
        assert_eq!(ca.width(), 100);
        assert_eq!(ca.rule(), 30);
    }

    #[test]
    fn test_elementary_ca_center() {
        let mut ca = ElementaryCA::new(10, 30);
        ca.set_center();

        assert!(!ca.get(0));
        assert!(ca.get(5));
        assert!(!ca.get(9));
    }

    #[test]
    fn test_elementary_ca_step() {
        let mut ca = ElementaryCA::new(10, 30);
        ca.set_center();

        let initial = ca.cells().to_vec();
        ca.step();
        let after = ca.cells().to_vec();

        // State should change
        assert_ne!(initial, after);
    }

    #[test]
    fn test_elementary_ca_rule_90() {
        // Rule 90 produces Sierpinski triangle
        let mut ca = ElementaryCA::new(11, 90);
        ca.set_center();

        // After one step, should have two cells
        ca.step();

        let alive_count: usize = ca.cells().iter().filter(|&&c| c).count();
        assert_eq!(alive_count, 2);
    }

    #[test]
    fn test_elementary_ca_generate_pattern() {
        let mut ca = ElementaryCA::new(20, 30);
        ca.set_center();

        let pattern = ca.generate_pattern(10);

        assert_eq!(pattern.len(), 10);
        assert_eq!(pattern[0].len(), 20);
    }

    #[test]
    fn test_2d_ca_creation() {
        let ca = CellularAutomaton2D::new(10, 10, &[3], &[2, 3]);
        assert_eq!(ca.width(), 10);
        assert_eq!(ca.height(), 10);
    }

    #[test]
    fn test_2d_ca_set_get() {
        let mut ca = CellularAutomaton2D::new(10, 10, &[3], &[2, 3]);

        assert!(!ca.get(5, 5));
        ca.set(5, 5, true);
        assert!(ca.get(5, 5));
    }

    #[test]
    fn test_2d_ca_randomize() {
        let mut ca = CellularAutomaton2D::new(20, 20, &[3], &[2, 3]);
        ca.randomize(12345, 0.5);

        let pop = ca.population();
        // Should have roughly 50% alive (with some variance)
        assert!(pop > 100 && pop < 300);
    }

    #[test]
    fn test_game_of_life_blinker() {
        // Blinker oscillator
        let mut life = GameOfLife::life(5, 5);

        // Horizontal blinker
        life.set(1, 2, true);
        life.set(2, 2, true);
        life.set(3, 2, true);

        let initial_pop = life.population();
        assert_eq!(initial_pop, 3);

        // After one step, should become vertical
        life.step();

        assert!(!life.get(1, 2));
        assert!(life.get(2, 1));
        assert!(life.get(2, 2));
        assert!(life.get(2, 3));
        assert!(!life.get(3, 2));
    }

    #[test]
    fn test_game_of_life_block() {
        // Block still life
        let mut life = GameOfLife::life(5, 5);

        life.set(1, 1, true);
        life.set(2, 1, true);
        life.set(1, 2, true);
        life.set(2, 2, true);

        let initial = life.cells().clone();
        life.step();

        // Block should not change
        assert_eq!(life.cells(), &initial);
    }

    #[test]
    fn test_count_neighbors() {
        let mut ca = CellularAutomaton2D::new(5, 5, &[3], &[2, 3]);

        // Set a cross pattern around (2,2)
        ca.set(1, 2, true);
        ca.set(3, 2, true);
        ca.set(2, 1, true);
        ca.set(2, 3, true);

        let count = ca.count_neighbors(2, 2);
        assert_eq!(count, 4);
    }

    #[test]
    fn test_rules_presets() {
        let (birth, survive) = rules::LIFE;
        let life = CellularAutomaton2D::new(10, 10, birth, survive);
        assert_eq!(life.width(), 10);

        let (birth, survive) = rules::HIGH_LIFE;
        let _high = CellularAutomaton2D::new(10, 10, birth, survive);

        let (birth, survive) = rules::SEEDS;
        let _seeds = CellularAutomaton2D::new(10, 10, birth, survive);
    }

    #[test]
    fn test_wrap_behavior() {
        let mut ca = CellularAutomaton2D::new(5, 5, &[3], &[2, 3]);
        ca.set_wrap(true);

        // Set cells at edges
        ca.set(0, 0, true);
        ca.set(4, 0, true);
        ca.set(0, 4, true);

        // Cell at (0,0) should count (4,4) as neighbor when wrapping
        let count = ca.count_neighbors(0, 0);
        assert!(count >= 2);
    }

    // Maze generation tests

    #[test]
    fn test_maze_dimensions() {
        let maze = Maze::new(10, 8);
        assert_eq!(maze.width(), 10);
        assert_eq!(maze.height(), 8);
        assert_eq!(maze.grid_width(), 21);
        assert_eq!(maze.grid_height(), 17);
    }

    #[test]
    fn test_maze_recursive_backtracker() {
        let maze = generate_maze(5, 5, MazeAlgorithm::RecursiveBacktracker, 12345);
        assert_eq!(maze.width(), 5);
        assert_eq!(maze.height(), 5);

        // All cells should be carved (not walls)
        for y in 0..5 {
            for x in 0..5 {
                let gx = x * 2 + 1;
                let gy = y * 2 + 1;
                assert!(
                    !maze.is_wall(gx, gy),
                    "Cell ({}, {}) should be carved",
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn test_maze_prims() {
        let maze = generate_maze(5, 5, MazeAlgorithm::Prims, 12345);
        assert_eq!(maze.width(), 5);

        // Check that cells are connected
        let mut passage_count = 0;
        for row in maze.grid() {
            for &cell in row {
                if !cell {
                    passage_count += 1;
                }
            }
        }
        assert!(passage_count > 0);
    }

    #[test]
    fn test_maze_kruskals() {
        let maze = generate_maze(5, 5, MazeAlgorithm::Kruskals, 12345);
        assert_eq!(maze.width(), 5);

        // All cells should be carved
        for y in 0..5 {
            for x in 0..5 {
                let gx = x * 2 + 1;
                let gy = y * 2 + 1;
                assert!(!maze.is_wall(gx, gy));
            }
        }
    }

    #[test]
    fn test_maze_ellers() {
        let maze = generate_maze(5, 5, MazeAlgorithm::Ellers, 12345);
        assert_eq!(maze.width(), 5);
        assert_eq!(maze.height(), 5);
    }

    #[test]
    fn test_maze_binary_tree() {
        let maze = generate_maze(5, 5, MazeAlgorithm::BinaryTree, 12345);
        assert_eq!(maze.width(), 5);

        // First row should have only westward passages (except first cell)
        // This is a characteristic of binary tree mazes
    }

    #[test]
    fn test_maze_sidewinder() {
        let maze = generate_maze(5, 5, MazeAlgorithm::Sidewinder, 12345);
        assert_eq!(maze.width(), 5);
    }

    #[test]
    fn test_maze_deterministic() {
        // Same seed should produce same maze
        let maze1 = generate_maze(10, 10, MazeAlgorithm::RecursiveBacktracker, 42);
        let maze2 = generate_maze(10, 10, MazeAlgorithm::RecursiveBacktracker, 42);

        assert_eq!(maze1.grid(), maze2.grid());
    }

    #[test]
    fn test_maze_different_seeds() {
        // Different seeds should produce different mazes
        let maze1 = generate_maze(10, 10, MazeAlgorithm::RecursiveBacktracker, 1);
        let maze2 = generate_maze(10, 10, MazeAlgorithm::RecursiveBacktracker, 2);

        assert_ne!(maze1.grid(), maze2.grid());
    }

    #[test]
    fn test_maze_to_string_art() {
        let maze = generate_maze(3, 3, MazeAlgorithm::BinaryTree, 12345);
        let art = maze.to_string_art();

        // Should have 7 lines (3*2+1 rows)
        let lines: Vec<_> = art.lines().collect();
        assert_eq!(lines.len(), 7);

        // Each line should have 7 characters
        for line in &lines {
            assert_eq!(line.chars().count(), 7);
        }
    }

    #[test]
    fn test_maze_all_algorithms_complete() {
        // All algorithms should produce valid mazes
        let algorithms = [
            MazeAlgorithm::RecursiveBacktracker,
            MazeAlgorithm::Prims,
            MazeAlgorithm::Kruskals,
            MazeAlgorithm::Ellers,
            MazeAlgorithm::BinaryTree,
            MazeAlgorithm::Sidewinder,
        ];

        for algo in algorithms {
            let maze = generate_maze(8, 8, algo, 12345);
            assert_eq!(maze.width(), 8);
            assert_eq!(maze.height(), 8);

            // All cell positions should be passages
            for y in 0..8 {
                for x in 0..8 {
                    let gx = x * 2 + 1;
                    let gy = y * 2 + 1;
                    assert!(
                        !maze.is_wall(gx, gy),
                        "Algorithm {:?} failed at ({}, {})",
                        algo,
                        x,
                        y
                    );
                }
            }
        }
    }
}
