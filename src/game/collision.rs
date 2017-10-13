use na;
use na::{Dot, Absolute};
use ::fixvec::{CmpNorm, Norm, vec_from_angle, vec_angle};
use ::fixmath::{Fixed, PI, min_angle_diff, normalized_angle};
pub type GameReal = Fixed;
pub type Vec2 = na::Vec2<GameReal>;
const EPSILON: GameReal = ::fixmath::ONE_HUNDRENTH; // 0.01

#[derive(Debug)]
pub struct Circle {
    pub center: Vec2,
    pub radius: GameReal,
}
#[derive(Debug)]
pub struct Collider {
    pub circle: Circle,
    pub velocity: Vec2,
}
#[derive(Debug, Clone, RustcDecodable)]
pub struct Line {
    pub point_a: Vec2,
    pub point_b: Vec2,
}

impl Circle {
    fn bounding_box(&self) -> BoundingBox {
        BoundingBox {
            min: Vec2::new(self.center.x - self.radius, self.center.y - self.radius),
            max: Vec2::new(self.center.x + self.radius, self.center.y + self.radius),
        }
    }
}

impl Line {
    fn bounding_box(&self) -> BoundingBox {
        BoundingBox {
            min: Vec2::new(self.point_a.x.min_(self.point_b.x), self.point_a.y.min_(self.point_b.y)),
            max: Vec2::new(self.point_a.x.max_(self.point_b.x), self.point_a.y.max_(self.point_b.y)),
        }
    }
}

fn bounding_boxes_overlap(c: &Circle, l: &Line) -> bool {
    let cl = c.center.x - c.radius;
    let cb = c.center.y - c.radius;
    let cr = c.center.x + c.radius;
    let ct = c.center.y + c.radius;
    let ll = l.point_a.x.min_(l.point_b.x);
    let lb = l.point_a.y.min_(l.point_b.y);
    let lr = l.point_a.x.max_(l.point_b.x);
    let lt = l.point_a.y.max_(l.point_b.y);
    !(lr < cl || cr < ll || ct < lb || lt < cb)
}

pub fn overlapping_point_line(c: &Circle, l: &Line) -> Option<Vec2> {
    if bounding_boxes_overlap(c, l) {
        let closest_point = get_closest_point_line(c.center, l);
        let d = closest_point - c.center;
        if CmpNorm(d) <= c.radius {
            Some(closest_point)
        } else {
            None
        }
    } else {
        None
    }
}

fn get_closest_point_line(p: Vec2, l: &Line) -> Vec2 {
    let ldir = (l.point_b - l.point_a).normalize();
    let lrel = p - l.point_a;

    let lp = ldir.dot(&lrel);
    if lp > fixed!(0) && lp < ldir.dot(&(l.point_b - l.point_a)) { // ldir.dot(&l.point_a) < lp && lp < ldir.dot(&l.point_b) {
        l.point_a + ldir * lp
    } else {
        let da = CmpNorm(p - l.point_a);
        let db = CmpNorm(p - l.point_b);

        if da < db {
            l.point_a
        } else {
            l.point_b
        }
    }
}

fn get_contact_circle_line(a: &Collider, l: &Line) -> Option<GameReal> {
    if !bounding_boxes_overlap(&a.circle, l) { return None; }
    let closest_point = get_closest_point_line(a.circle.center, l);
    /* ::dbg_lines.lock().unwrap().push(Line {
        point_a: a.circle.center,
        point_b: closest_point,
    }); */
    let r = a.circle.radius + EPSILON;
    let d = closest_point - a.circle.center;
    if CmpNorm(d) <= r {
        Some(vec_angle(d))
    } else {
        None
    }
}

fn intersection_time_collider_line(collider: &Collider, line: &Line, max_time: GameReal) -> Option<GameReal> {
    if !bounding_boxes_overlap(&Circle {
        center: collider.circle.center,
        radius: collider.circle.radius + collider.velocity.x.abs() + collider.velocity.y.abs()
    }, line) { return None; }
    fn pen_vector(p: Vec2, line: &Line) -> Vec2 {
        get_closest_point_line(p, line) - p
    }
    if CmpNorm(pen_vector(collider.circle.center, line)) < collider.circle.radius {
        return Some(fixed!(0));
    }
    if CmpNorm(pen_vector(collider.circle.center + collider.velocity * max_time, line)) > collider.circle.radius + EPSILON {
        return None;
    }
    let mut low = fixed!(0);
    let mut high = max_time;
    loop {
        if low >= high {
            return Some(high);
        }
        let mid = (low + high) / fixed!(2);
        let r = CmpNorm(pen_vector(collider.circle.center + collider.velocity * mid, line));
        if r < collider.circle.radius {
            high = mid - EPSILON;
        } else if r > collider.circle.radius + EPSILON {
            low = mid + EPSILON;
        } else {
            return Some(mid);
        }
    }
}
fn intersection_time_collider_collider(a: &Collider, b: &Collider, max_time: GameReal) -> Option<GameReal> {
    let overlap_distance = a.circle.radius + b.circle.radius;
    let vel_rel = a.velocity - b.velocity;
    if CmpNorm(a.circle.center - b.circle.center) < overlap_distance - EPSILON {
        return Some(fixed!(0));
    } else if CmpNorm(a.circle.center - b.circle.center + vel_rel * max_time) > overlap_distance + EPSILON {
        return None;
    }
    let mut low = fixed!(0);
    let mut high = max_time;
    loop {
        if low >= high {
            return Some(high);
        }
        let mid = (low + high) / fixed!(2);
        let r = CmpNorm(a.circle.center - b.circle.center + vel_rel * mid);
        if r < overlap_distance {
            high = mid - EPSILON;
        } else if r > overlap_distance + EPSILON {
            low = mid + EPSILON;
        } else {
            return Some(mid);
        }
    }
}

fn find_line_contacts(colliders: &Vec<Collider>, lines_quadtree: &QuadTree<(usize, Line)>) -> Vec<(usize, usize, GameReal)> {
    let mut contacts = vec![];
    for (i, collider) in colliders.iter().enumerate() {
        let bbox = collider.circle.bounding_box().expand_by(Vec2::new(EPSILON, EPSILON)).expand_by(Vec2::new(-EPSILON, -EPSILON));
        let mut outputs = vec![];
        lines_quadtree.get(&bbox, &mut outputs);
        for &&(j, ref line) in outputs.iter() {
            match get_contact_circle_line(collider, line) {
                Some(angle) => { contacts.push((i, j, angle)); },
                None => { },
            }
        }
    }
    contacts
}

fn find_collider_contacts(colliders: &Vec<Collider>) -> Vec<(usize, usize, GameReal)> {
    let mut contacts = vec![];
    for (i, a) in colliders.iter().enumerate() {
        for (j, b) in colliders[i + 1..].iter().enumerate() {
            let d = b.circle.center - a.circle.center;
            let r = b.circle.radius + a.circle.radius + EPSILON;
            if CmpNorm(d) <= r {
                contacts.push((i, i + 1 + j, vec_angle(d)));
            }
        }
    }
    contacts
}

/// ContactCircle accrues contact angles and keeps track of the range restricted by them.
/// `Open` means no restricted angles, `Partial` means only the angles between `low` and `high` are
/// unrestricted, and `Closed` means all angles are restricted. Because every contact angle added
/// restricts an arc of pi, it is impossible have an unrestricted range that is not continuous.
#[derive(Debug)]
enum ContactCircle {
    Open,
    Partial { low: GameReal, high: GameReal },
    Closed,
}
impl ContactCircle {
    fn add_contact(&mut self, contact_angle: GameReal) {
        *self = match *self {
            ContactCircle::Open => ContactCircle::Partial {
                low: normalized_angle(contact_angle + PI / fixed!(2)),
                high: normalized_angle(contact_angle - PI / fixed!(2)),
            },
            ContactCircle::Partial { low, high } => {
                let from_low = min_angle_diff(contact_angle, low);
                let from_high = min_angle_diff(contact_angle, high);
                let (new_low, new_high) = if from_low.abs() < PI / fixed!(2) {
                    (normalized_angle(contact_angle + PI / fixed!(2)), high)
                } else if from_high.abs() < PI / fixed!(2) {
                    (low, normalized_angle(contact_angle - PI / fixed!(2)))
                } else {
                    (low, high)
                };
                let d = min_angle_diff(new_high, new_low);
                if d <= fixed!(0) && d > -PI + EPSILON {
                    ContactCircle::Closed
                } else {
                    ContactCircle::Partial { low: new_low, high: new_high }
                }
            },
            ContactCircle::Closed => ContactCircle::Closed,
        }
    }
    fn best_edge(&self, angle: GameReal) -> Option<GameReal> {
        match *self {
            ContactCircle::Open => Some(angle),
            ContactCircle::Partial { low, high } => {
                let from_low = min_angle_diff(angle, low);
                let from_high = min_angle_diff(angle, high);
                if from_low > fixed!(0) && from_high < fixed!(0) {
                    Some(angle)
                } else if from_low.abs() < from_high.abs() {
                    Some(low)
                } else {
                    Some(high)
                }
            },
            ContactCircle::Closed => None,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::ContactCircle;
    use ::fixmath::{Fixed, PI, min_angle_diff, normalized_angle};
    pub type GameReal = Fixed;
    #[test]
    fn open_circle() {
        let mut contacts = ContactCircle::Open;
        contacts.add_contact(fixed!(0));
        println!("{:?}", contacts);
        assert_eq!(Some(PI), contacts.best_edge(PI));
    }
    #[test]
    fn partial_circle() {
        let mut contacts = ContactCircle::Open;
        contacts.add_contact(fixed!(0));
        println!("{:?}", contacts);
        contacts.add_contact(PI / fixed!(2));
        println!("{:?}", contacts);
        assert_eq!(Some(PI * fixed!(5) / fixed!(4)), contacts.best_edge(PI * fixed!(5) / fixed!(4)));
    }
    #[test]
    fn partial_circle_fail() {
        let mut contacts = ContactCircle::Open;
        contacts.add_contact(fixed!(0));
        println!("{:?}", contacts);
        contacts.add_contact(PI / fixed!(2));
        println!("{:?}", contacts);
        let r = contacts.best_edge(PI * fixed!(3) / fixed!(4));
        assert!(r != None);
        assert!(r != Some(PI * fixed!(3) / fixed!(4)));
    }
    #[test]
    fn closed_circle() {
        let mut contacts = ContactCircle::Open;
        contacts.add_contact(fixed!(0));
        println!("{:?}", contacts);
        contacts.add_contact(PI / fixed!(2));
        println!("{:?}", contacts);
        contacts.add_contact(-PI * fixed!(3) / fixed!(4));
        println!("{:?}", contacts);
        assert!(match contacts { ContactCircle::Closed => true, _ => false });
    }
    #[test]
    fn partial_lots() {
        use rand::random;
        for _ in 0..100 {
            let a = random::<GameReal>() * PI * fixed!(2);
            let mut contacts = ContactCircle::Open;
            // println!("\n\nRun:{}", a);
            for _ in 0..1000 {
                let c = a + PI / fixed!(2) + fixed!(1) / fixed!(10) + random::<GameReal>() * (PI - fixed!(2) / fixed!(10));
                contacts.add_contact(c);
                // println!("{:?}", c);
                // println!("{:?}", contacts);
                assert_eq!(Some(a), contacts.best_edge(a));
            }
        }
    }
}

fn limit_velocities(colliders: &mut Vec<Collider>, velocities: &Vec<Vec2>, contacts: &Vec<ContactCircle>) {
    for i in 0..colliders.len() {
        if CmpNorm(colliders[i].velocity) == fixed!(0) {
            continue;
        }
        let vel = velocities[i];
        let vel_dir = vec_angle(vel);
        colliders[i].velocity = match contacts[i].best_edge(vel_dir) {
            Some(a) => {
                let edge_vec = vec_from_angle(a);
                let mag = vel.dot(&edge_vec);
                if mag < fixed!(0) {
                    Vec2::new(fixed!(0), fixed!(0))
                } else {
                    edge_vec * mag
                }
            },
            None => Vec2::new(fixed!(0), fixed!(0))
        };
    }
}

struct BoundingBox {
    min: Vec2,
    max: Vec2,
}
impl BoundingBox {
    fn overlaps(&self, other: &BoundingBox) -> bool {
        !(
            self.max.x < other.min.x ||
            self.max.y < other.min.y ||
            other.max.x < self.min.x ||
            other.max.y < self.min.y
        )
    }
    fn expand_by(&self, amount: Vec2) -> BoundingBox {
        BoundingBox {
            min: Vec2::new(self.min.x + amount.x.min_(fixed!(0)), self.min.y + amount.y.min_(fixed!(0))),
            max: Vec2::new(self.max.x + amount.x.max_(fixed!(0)), self.max.y + amount.y.max_(fixed!(0))),
        }
    }
}
struct QuadTree<T> {
    bounds: BoundingBox,
    depth: u8,
    children: Option<Box<[QuadTree<T>; 4]>>,
    entries: Vec<(BoundingBox, T)>,
}
const MAX_QUAD_TREE_DEPTH: u8 = 10;
impl<T> QuadTree<T> {
    fn insert(&mut self, bbox: BoundingBox, t: T) {
        if self.depth < MAX_QUAD_TREE_DEPTH && self.children.is_none() {
            let x0 = bbox.min.x;
            let x1 = (bbox.min.x + bbox.max.x) / fixed!(2);
            let x2 = bbox.max.x;
            let y0 = bbox.min.y;
            let y1 = (bbox.min.y + bbox.max.y) / fixed!(2);
            let y2 = bbox.max.y;
            fn vec2(x: Fixed, y: Fixed) -> Vec2 {
                Vec2::new(x, y)
            }
            self.children = Some(Box::new([
                QuadTree { bounds: BoundingBox { min: vec2(x0, y0), max: vec2(x1, y1) }, depth: self.depth + 1, children: None, entries: vec![] },
                QuadTree { bounds: BoundingBox { min: vec2(x1, y0), max: vec2(x2, y1) }, depth: self.depth + 1, children: None, entries: vec![] },
                QuadTree { bounds: BoundingBox { min: vec2(x0, y1), max: vec2(x1, y2) }, depth: self.depth + 1, children: None, entries: vec![] },
                QuadTree { bounds: BoundingBox { min: vec2(x1, y1), max: vec2(x2, y2) }, depth: self.depth + 1, children: None, entries: vec![] },
            ]));
        }
        if let Some(ref mut children) = self.children {
            let overlapper = {
                let mut overlapping_children = children.iter_mut().filter(|child| child.bounds.overlaps(&bbox));
                let first_overlapper = overlapping_children.next();
                if overlapping_children.next().is_none() {
                    first_overlapper
                } else {
                    None
                }
            };
            if let Some(overlapper) = overlapper {
                overlapper.insert(bbox, t);
            } else {
                self.entries.push((bbox, t));
            }
        } else {
            self.entries.push((bbox, t));
        }
    }
    fn get<'a>(&'a self, query_bbox: &BoundingBox, output: &mut Vec<&'a T>) {
        for entry in self.entries.iter() {
            if entry.0.overlaps(query_bbox) {
                output.push(&entry.1);
            }
        }
        if let Some(ref children) = self.children {
            for child in children.iter() {
                if child.bounds.overlaps(query_bbox) {
                    child.get(query_bbox, output);
                }
            }
        }
    }
}

pub fn move_colliders(colliders: &mut Vec<Collider>, lines: &Vec<Line>) {
    let mut t = fixed!(1);
    let mut contacts = colliders.iter().map(|_| ContactCircle::Open).collect::<Vec<_>>();
    let velocities = colliders.iter().map(|c| c.velocity).collect::<Vec<_>>();
    let mut lines_quadtree: QuadTree<(usize, Line)> = QuadTree { bounds: BoundingBox { min: Vec2::new(fixed!(-10_000), fixed!(-10_000)), max: Vec2::new(fixed!(10_000), fixed!(10_000)) }, depth: 0, children: None, entries: vec![] };

    for (index, line) in lines.iter().enumerate() {
        let bbox = line.bounding_box();
        lines_quadtree.insert(bbox, (index, line.clone()));
    }

    while t > fixed!(0) {
        let collider_contacts = find_collider_contacts(colliders);
        let line_contacts = find_line_contacts(colliders, &lines_quadtree);

        for (index, angle) in collider_contacts.iter()
            .flat_map(|&(i, j, angle)| vec![(i, angle), (j, normalized_angle(angle + PI))])
            .chain(line_contacts.iter().map(|&(i, _, angle)| (i, angle)))
        {
            contacts[index].add_contact(angle);
        }

        limit_velocities(colliders, &velocities, &contacts);

        let mut dt = t;
        for (i, a) in colliders.iter().enumerate() {
            let collider_potential_bbox = a.circle.bounding_box().expand_by(a.velocity);
            for (j, b) in colliders[i + 1..].iter().enumerate() {
                if collider_contacts.iter().find(|t| (t.0, t.1) == (i, i + 1 + j)).is_some() {
                    continue;
                }
                if let Some(tcol) = intersection_time_collider_collider(a, b, dt) {
                    dt = tcol;
                }
            }
            let mut outputs = vec![];
            lines_quadtree.get(&collider_potential_bbox, &mut outputs);
            for &&(j, ref l) in outputs.iter() {
                if line_contacts.iter().find(|t| (t.0, t.1) == (i, j)).is_some() {
                    continue;
                }
                if let Some(tcol) = intersection_time_collider_line(a, l, dt) {
                    dt = tcol;
                }
            }
        }

        dt = dt.max_(fixed!(1) / fixed!(100)).min_(t);
        // let dt = fixed!(1) / fixed!(10);
        for collider in colliders.iter_mut() {
            collider.circle.center = collider.circle.center + collider.velocity * dt;
        }
        t = t - dt;
    }
}
