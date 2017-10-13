extern crate xml;

use self::xml::reader::{EventReader};
use fixmath::Fixed;
use fixvec::{Norm, vec_cross_z};
use na::Dot;
use na::Vec2;
use na::Mat4;
use na::Eye;

// #[derive(RustcDecodable, RustcEncodable)]
pub struct LevelData {
	pub grass_polygons: Vec<Vec<LevelLine>>,
	pub water_polygons: Vec<Vec<LevelLine>>,
	pub sand_polygons: Vec<Vec<LevelLine>>,
	pub collision_lines: Vec<LevelLine>,
}

// #[derive(RustcDecodable, RustcEncodable)]
pub struct LevelLine {
    pub a: Vec2<Fixed>,
    pub b: Vec2<Fixed>,
}

pub struct Polygon {
	pub lines: Vec<LevelLine>,
}

fn polygon_from_points(points: &[Vec2<Fixed>]) -> Option<Polygon> {
	if points.len() < 2 { return None; }
	let mut lines = vec![];
	for i in 0..points.len() {
		lines.push(LevelLine {
			a: points[i],
			b: points[(i + 1) % points.len()],
		});
	}
	Some(Polygon { lines: lines })
}

type Curve = Fn(Fixed) -> Vec2<Fixed>;
fn cubic_bezier(p0: Vec2<Fixed>, p1: Vec2<Fixed>, p2: Vec2<Fixed>, p3: Vec2<Fixed>) -> Box<Curve> {
	let q1 = quadratic_bezier(p0, p1, p2);
	let q2 = quadratic_bezier(p1, p2, p3);
	Box::new(move |t| mix(q1(t), q2(t), t))
}
fn quadratic_bezier(p0: Vec2<Fixed>, p1: Vec2<Fixed>, p2: Vec2<Fixed>) -> Box<Curve> {
	Box::new(move |t| mix(mix(p0, p1, t), mix(p1, p2, t), t))
}
fn mix(p0: Vec2<Fixed>, p1: Vec2<Fixed>, t: Fixed) -> Vec2<Fixed> {
	p0 + (p1 - p0) * t
}

fn subdivide_sample_curve(curve: &Box<Curve>, t0: Fixed, t1: Fixed, slack: Fixed, mut result: Vec<Vec2<Fixed>>) -> Vec<Vec2<Fixed>> {
	let start = curve(t0);
	let end = curve(t1);
	let tmid = (t1 + t0) / fixed!(2);
	let mid = curve(tmid);
	let norm = vec_cross_z((end - start).normalize().normalize());
	let dm = norm.x * mid.x + norm.y * mid.y; // norm.dot(&mid);
	let ds = norm.x * start.x + norm.y * start.y; // norm.dot(&start);
	let d = (dm - ds).abs();
	if d > slack * slack {
		let mut first_half = subdivide_sample_curve(curve, t0, tmid, slack, result);
		first_half.pop();
		subdivide_sample_curve(curve, tmid, t1, slack, first_half)
	} else {
		result.push(start);
		result.push(end);
		result
	}
}

fn parse_vector<'a, 'b, I: Iterator<Item=&'a str>>(iter: &'b mut I) -> Option<Vec2<Fixed>> {
	match (iter.next().and_then(Fixed::parse), iter.next().and_then(Fixed::parse)) {
		(Some(a), Some(b)) => Some(Vec2::new(a, -b)),
		_ => None
	}
}

fn parse_path(svg_path_d: &str, transform: Mat4<Fixed>) -> Result<Vec<Polygon>, &'static str> {
	macro_rules! tryo {
		($e:expr) => { match $e {
			Some(v) => v,
			None => return Err("Failed to parse vector"),
		}}
	}

	let mut polygons = vec![];
	let mut current_points = vec![];

	let mut last_point = Vec2::new(fixed!(0), fixed!(0));
	let mut last_tangent: Option<Vec2<Fixed>> = None;
	let mut tokens = &mut svg_path_d.split(|c| c == ' ' || c == ',').peekable();
	let mut command = 'L';
	loop {
		let next_token = match tokens.peek().cloned() {
			Some(t) => t,
			None => break,
		};
		if next_token.len() == 1 {
			let next_char = next_token.chars().nth(0).unwrap();
			if "mMlLhHvVzZcCsSqQtTaA".contains(next_char) {
				command = next_char;
				tokens.next();
			}
		}

		let next_vector = move |iter: &mut _| {
			if command.is_lowercase() {
				parse_vector(iter).map(|v| last_point + v)
			} else {
				parse_vector(iter)
			}
		};

		let mut next_last_tangent = None;

		match command {
			'm' | 'M' => {
				match polygon_from_points(&current_points) {
					Some(poly) => polygons.push(poly),
					_ => (),
				}
				current_points.clear();
				let p = tryo!(next_vector(tokens));
				last_point = p;
				current_points.push(p);
				command = if command == 'm' { 'l' } else { 'L' };
			},
			'z' | 'Z' => {
				match polygon_from_points(&current_points) {
					Some(poly) => polygons.push(poly),
					_ => (),
				}
				last_point = current_points.get(0).cloned().unwrap_or(Vec2::new(fixed!(0), fixed!(0)));
				current_points.clear();
				command = 'L';
			}
			'l' | 'L' => {
				let p = tryo!(next_vector(tokens));
				last_point = p;
				current_points.push(p);
			},
			'h' | 'H' => {
				let x = tryo!(tokens.next().and_then(Fixed::parse));
				let p = if command == 'h' {
					last_point + Vec2::new(x, fixed!(0))
				} else {
					Vec2::new(x, last_point.y)
				};
				last_point = p;
				current_points.push(p);
			},
			'v' | 'V' => {
				let y = tryo!(tokens.next().and_then(Fixed::parse));
				let p = if command == 'v' {
					last_point + Vec2::new(fixed!(0), y)
				} else {
					Vec2::new(last_point.x, y)
				};
				last_point = p;
				current_points.push(p);
			},
			'c' | 'C' => {
				let first_point = last_point;
				let control_point1 = tryo!(next_vector(tokens));
				let control_point2 = tryo!(next_vector(tokens));
				let end_point = tryo!(next_vector(tokens));
				last_point = end_point;
				next_last_tangent = Some(end_point - control_point2);
				let curve = cubic_bezier(first_point, control_point1, control_point2, end_point);
				let curve_points = subdivide_sample_curve(&curve, fixed!(0), fixed!(1), fixed!(1) / fixed!(2), vec![]);
				current_points.extend(curve_points.iter().skip(1));
			},
			's' | 'S' => {
				let first_point = last_point;
				let control_point2 = tryo!(next_vector(tokens));
				let end_point = tryo!(next_vector(tokens));
				let control_point1 = match last_tangent {
					Some(t) => first_point + t,
					None => control_point2,
				};
				last_point = end_point;
				next_last_tangent = Some(end_point - control_point2);
				let curve = cubic_bezier(first_point, control_point1, control_point2, end_point);
				current_points.extend(subdivide_sample_curve(&curve, fixed!(0), fixed!(1), fixed!(1) / fixed!(2), vec![]).iter().skip(1));
			},
			'q' | 'Q' => {
				let first_point = last_point;
				let control_point1 = tryo!(next_vector(tokens));
				let end_point = tryo!(next_vector(tokens));
				last_point = end_point;
				next_last_tangent = Some(end_point - control_point1);
				let curve = quadratic_bezier(first_point, control_point1, end_point);
				current_points.extend(subdivide_sample_curve(&curve, fixed!(0), fixed!(1), fixed!(1) / fixed!(2), vec![]).iter().skip(1));
			},
			't' | 'T' => {
				let first_point = last_point;
				let end_point = tryo!(next_vector(tokens));
				let control_point1 = match last_tangent {
					Some(t) => first_point + t,
					None => first_point,
				};
				last_point = end_point;
				next_last_tangent = Some(end_point - control_point1);
				let curve = quadratic_bezier(first_point, control_point1, end_point);
				current_points.extend(subdivide_sample_curve(&curve, fixed!(0), fixed!(1), fixed!(1) / fixed!(2), vec![]).iter().skip(1));
			},
			'a' | 'A' => {
				tokens.next(); // rx
				tokens.next(); // ry
				tokens.next(); // x-axis-rotation
                tokens.next(); // large-arc-flag
                tokens.next(); // sweep-flag
				let p = tryo!(next_vector(tokens));
				last_point = p;
				current_points.push(p);
			},
			_ => unreachable!(),
		}
	}
	Ok(polygons)
}

pub fn load_level_from_file(path: &str) -> LevelData {
    use ::std::fs::File;
    use ::std::io::Read;
    let mut file = File::open(path).unwrap();
    let mut buf = vec![];
    file.read_to_end(&mut buf).unwrap();
    load_level(buf)
}

pub fn load_level(bytes: Vec<u8>) -> LevelData {
    use self::xml::reader::XmlEvent::*;
    let mut level_data = LevelData { grass_polygons: vec![], water_polygons: vec![], sand_polygons: vec![], collision_lines: vec![] };
    enum Layer { Collision, Grass, Water, Sand }
    let mut layer = None;
	let identity = Mat4::new_identity(4);
    for event in EventReader::new(&bytes[..]) {
        match event {
            Ok(StartElement { name, attributes, .. }) => {
                match name.local_name.as_ref() {
                    "g" => {
                        match attributes.iter().filter(|attr| attr.name.local_name == "label").next() {
                            Some(attr) => {
                                layer = match attr.value.as_ref() {
                                    "Collision" => Some(Layer::Collision),
                                    "Grass" => Some(Layer::Grass),
                                    "Water" => Some(Layer::Water),
                                    "Sand" => Some(Layer::Sand),
                                    _ => layer,
                                }
                            },
                            None => (),
                        };
                    },
                    "path" => {
                        match attributes.iter().filter(|attr| attr.name.local_name == "d").next() {
                            Some(path) => match parse_path(&path.value, identity) {
								Ok(polygons) => {
									for poly in polygons {
		                                match layer {
		                                    Some(Layer::Collision) => level_data.collision_lines.extend(poly.lines),
		                                    Some(Layer::Grass) => level_data.grass_polygons.push(poly.lines),
		                                    Some(Layer::Water) => level_data.water_polygons.push(poly.lines),
		                                    Some(Layer::Sand) => level_data.sand_polygons.push(poly.lines),
		                                    _ => (),
		                                }
									}
								},
								Err(msg) => println!("Error loading level: {}", msg),
							},
                            None => (),
                        }
                    },
                    _ => (),
                }
            },
            _ => {

            },
        }
    }
    level_data
}