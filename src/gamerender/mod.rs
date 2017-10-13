pub mod effect;
mod interpolate;
pub mod presentation;

use na::{self, Vec2, Vec4, Mat4, Norm, Dot};
use std::f32::consts::PI as PI32;
use std::rc::Rc;
use glium::backend::Context;
use glium::{self, Surface};
use self::effect::{Effect, dust_effect, charge_effect, parried_effect, directed_effect};
use game::{self, BLOCK_SPREAD, GameState, Entity, Projectile, EntityState, Hitcircle, HitcircleState, CommitEffect};
use load_level::LevelLine;
use drawing::{self, color, translate, scale, rotate, trail, polygon_to_triangles};
use ::{dbg_lines, DbgTimestops, vec_from_angle, vec_cross_z, vec_angle};
use rand::random;
use fixvec::FromGame;
use self::interpolate::{Interpolation, Interpolate, Angle};

fn fold_items<T: Iterator<Item=U>, U, F>(mut i: T, f: F) -> U
	where F: FnMut(U, U) -> U
{
	let start = i.next().unwrap();
	i.fold(start, f)
}
fn min_float<T: Iterator<Item=f32>>(i: T) -> f32 {
	fold_items(i, |a, x| if x < a { x } else { a })
}
fn max_float<T: Iterator<Item=f32>>(i: T) -> f32 {
	fold_items(i, |a, x| if x > a { x } else { a })
}
fn get_intersections(y: f32, polygon: &Vec<LevelLine>) -> Vec<f32> {
	let mut nodes = vec![];
	for line in polygon.iter() {
		let ax = line.a.x.from_game();
		let ay = line.a.y.from_game();
		let bx = line.b.x.from_game();
		let by = line.b.y.from_game();
		if (y < ay) != (y < by) {
			let dy = by - ay;
			let dx = bx - ax;
			let t = (y - ay) / dy;
			nodes.push(ax + dx * t);
		}
	}
	nodes
}
const GRASS_DENSITY: f32 = 5.0;
fn build_grass_from_polygon(builder: &mut drawing::grass::GrassBuilder, polygon: &Vec<LevelLine>) {
	// https://www.reddit.com/r/rust/comments/29kia3/no_ord_for_f32/
	let min_x = min_float(polygon.iter().map(|line| line.a.x.from_game()));
	let max_x = max_float(polygon.iter().map(|line| line.a.x.from_game()));
	let min_y = min_float(polygon.iter().map(|line| line.a.y.from_game()));
	let max_y = max_float(polygon.iter().map(|line| line.a.y.from_game()));

	let mut y = min_y;
	loop {
		let mut x = min_x;
		if y > max_y { break; }
		// http://alienryderflex.com/polygon_fill/
		let nodes = get_intersections(y, polygon);
		loop {
			if nodes.iter().filter(|&&node_x| node_x < x).count() % 2 == 1 {
				builder.add_blade(x, y);
			}
			if x > max_x { break; }
			x += GRASS_DENSITY;
		}
		y += GRASS_DENSITY;
	}
}
fn build_grass_from_polygons(polygons: &Vec<Vec<LevelLine>>) -> drawing::grass::GrassBuilder {
	let mut builder = drawing::grass::GrassBuilder::new();
	for polygon in polygons.iter() {
		build_grass_from_polygon(&mut builder, polygon);
		builder.add_floor_polygon(&polygon.iter().map(|line| line.a.from_game()).collect::<Vec<_>>());
	}
	builder
}

fn build_geometry_from_polygons(polygons: &Vec<Vec<LevelLine>>) -> drawing::geometry::GeometryBuilder<drawing::geometry::SimpleVertex> {
	let mut builder = drawing::geometry::GeometryBuilder::new();
	for polygon in polygons.iter() {
		let vertices: Vec<_> = polygon.iter().map(|line| line.a.from_game()).collect();
		let indices: Vec<_> = polygon_to_triangles(&vertices).iter().flat_map(|is| is.iter().map(|&i| i as u32)).collect();
		builder.add_triangles(&vertices, &indices);
	}
	builder
}

pub struct Renderers {
    pub textures: drawing::texture::TextureCache,
    pub sprite_renderer: drawing::sprite::SpriteRenderer,
	pub trail_renderer: drawing::trail::TrailRenderer,
	pub triangle_renderer: drawing::dynamic_geometry::DynamicRenderer,
	pub line_renderer: drawing::gl_lines::LineRenderer,
	pub model_renderer: drawing::model::ModelRenderer,
    pub text: drawing::text::Text,
}
pub struct GameRenderer {
    // interpolations: Interpolations,
	effects: Vec<Effect>,
    trails: WeaponTrails,

    grass: drawing::grass::Grass,
	water: drawing::water::Water,
	pub sand: drawing::sand::Sand,
	disturber: drawing::simulation::Disturber,
    pub renderers: Renderers,
}
impl GameRenderer {
    pub fn new(level: &::load_level::LevelData, ctx: &Rc<Context>) -> GameRenderer {
        GameRenderer {
            effects: vec![],
            trails: WeaponTrails { trails: vec![] },

            grass: build_grass_from_polygons(&level.grass_polygons).build(&ctx),
			water: drawing::water::Water::new(&ctx, build_geometry_from_polygons(&level.water_polygons).build(&ctx)),
			sand: drawing::sand::Sand::new(&ctx, build_geometry_from_polygons(&level.sand_polygons).build(&ctx)),
			disturber: drawing::simulation::Disturber::new(&ctx),

            renderers: Renderers {
                textures: drawing::texture::TextureCache::new(ctx),
    			sprite_renderer: drawing::sprite::SpriteRenderer::new(ctx),
    			trail_renderer: drawing::trail::TrailRenderer::new(ctx),
				triangle_renderer: drawing::dynamic_geometry::DynamicRenderer::new(ctx),
				line_renderer: drawing::gl_lines::LineRenderer::new(ctx),
    			model_renderer: drawing::model::ModelRenderer::new(ctx),
                text: drawing::text::Text::new(ctx),
            }
        }
    }
}

impl<'a> Interpolation<'a, Entity> {
    fn position(&self) -> Vec2<f32> { self.field(|e| e.position.from_game()) }
    fn facing(&self) -> f32 { self.field(|e| Angle(e.facing.from_game())).0 }
	fn radius(&self) -> f32 { self.field(|e| e.radius.from_game()) }
}

#[derive(Debug)]
struct TimeCone {
	center: Vec2<f32>,
	present: f32,
	slope: f32,
}

#[derive(Copy, Clone)]
struct TimePos {
	time: f32,
	position: Vec2<f32>,
}
impl Interpolate for TimePos {
	fn interpolate(&self, other: &TimePos, t: f32) -> TimePos {
        TimePos {
			time: self.time.interpolate(&other.time, t),
			position: self.position.interpolate(&other.position, t),
		}
    }
}

fn time_at_point(p: Vec2<f32>, timecone: &TimeCone) -> f32 {
	timecone.present + (p - timecone.center).norm() * timecone.slope
}

fn vert_rect(m: Vec2<f32>, w: f32, h: f32) -> Vec<Vec2<f32>> {
	let hx = Vec2::new(w / 2.0, 0.0);
	let hy = Vec2::new(0.0, h / 2.0);
	let a = m - hx - hy;
	let b = m - hx + hy;
	let c = m + hx + hy;
	let d = m + hx - hy;
	vec![a, b, c, a, c, d]
}

fn vert_square(m: Vec2<f32>, s: f32) -> Vec<Vec2<f32>> {
	vert_rect(m, s, s)
}

fn vert_circle(center: Vec2<f32>, radius: f32, n: i32) -> Vec<Vec2<f32>> {
	let mut outer_vertices = vec![];
	for i in 0..n+1 {
		let a = i as f32 * PI32 * 2.0 / n as f32;
		outer_vertices.push(Vec2::new(center.x + radius * a.cos(), center.y + radius * a.sin()));
	}
	let mut result = vec![];
	for (a, b) in outer_vertices.iter().zip(outer_vertices.iter().cycle().skip(1)) {
		result.push(center);
		result.push(*a);
		result.push(*b);
	}
	result
}

mod warp_rendering {
	use super::interpolate::{Interpolation, Interpolate, Angle};
	use fixvec::FromGame;
	use super::{TimeCone, TimePos, time_at_point};
	use na::{self, Vec2, Vec4, Mat4, Norm, Dot};

	pub trait Positionable {
		fn get_position(&self) -> Vec2<f32>;
	}
	pub struct SimpleVertex {
		pub position: Vec2<f32>,
	}
	impl Positionable for SimpleVertex {
		fn get_position(&self) -> Vec2<f32> { self.position }
	}
	impl Interpolate for SimpleVertex {
		fn interpolate(&self, other: &SimpleVertex, t: f32) -> SimpleVertex {
			SimpleVertex { position: self.position.interpolate(&other.position, t) }
		}
	}

	pub struct FrameShapeId {
		pub id: u64,
		pub shape: FrameShape,
	}

	pub struct GenericFrameShape<V, P> {
		pub vertices: Vec<V>,
		pub persistent: P,
	}
	struct GenericTimeShape<V, P> {
		vertices: Vec<Vec<TimedVertex<V>>>,
		persistent: P,
	}

	pub fn solid_shape(vertices: Vec<SimpleVertex>, indices: Option<Vec<u16>>, color: Vec4<f32>) -> FrameShape {
		FrameShape::Solid(GenericFrameShape {
			vertices: vertices,
			persistent: ColorAndIndices {
				indices: indices,
				color: color,
			}	
		})
	}

	pub struct ColorAndIndices {
		pub indices: Option<Vec<u16>>,
		pub color: Vec4<f32>,
	}
	pub enum FrameShape {
		Solid(GenericFrameShape<SimpleVertex, ColorAndIndices>)
	}
	enum TimeShape {
		Empty,
		Solid(GenericTimeShape<SimpleVertex, ColorAndIndices>)
	}

	struct TimedVertex<T> {
		time: f32,
		vertex: T,
	}
	impl<T: Positionable + Interpolate> TimedVertex<T> {
		fn timepos(&self) -> TimePos {
			TimePos {
				position: self.vertex.get_position(),
				time: self.time,
			}
		}
	}
	impl<T: Positionable + Interpolate> Interpolate for TimedVertex<T> {
		fn interpolate(&self, other: &TimedVertex<T>, t: f32) -> TimedVertex<T> {
			TimedVertex {
				vertex: self.vertex.interpolate(&other.vertex, t),
				time: self.time.interpolate(&other.time, t),
			}
		}
	}

	fn merge_generic<V, P>(frameshape: GenericFrameShape<V, P>, t: f32, timeshape: &mut GenericTimeShape<V, P>) {
		let wrapped = frameshape.vertices.into_iter().map(|v| TimedVertex { time: t, vertex: v });
		for (v, a) in timeshape.vertices.iter_mut().zip(wrapped) {
			v.push(a);
		}
	}
	fn init_generic<V, P>(frameshape: GenericFrameShape<V, P>, t: f32) -> GenericTimeShape<V, P> {
		let wrapped = frameshape.vertices.into_iter().map(|v| TimedVertex { time: t, vertex: v });
		GenericTimeShape {
			vertices: wrapped.map(|v| vec![v]).collect(),
			persistent: frameshape.persistent
		}
	}

	fn merge(frameshape: FrameShape, t: f32, timeshape: &mut TimeShape) {
		let replacement = match frameshape {
			FrameShape::Solid(frameshape) => match *timeshape {
				TimeShape::Solid(ref mut timeshape) => { merge_generic(frameshape, t, timeshape); None },
				_ => Some(TimeShape::Solid(init_generic(frameshape, t)))
			}
		};
		if let Some(ts) = replacement {
			*timeshape = ts;
		}
	}

	use std::collections::BTreeMap;
	fn frame_shapes_to_time_shapes(frameshapes: Vec<(f32, Vec<FrameShapeId>)>) -> BTreeMap<u64, TimeShape> {
		let mut map: BTreeMap<u64, TimeShape> = BTreeMap::new();
		for (t, shapes) in frameshapes {
			for shape in shapes {
				let ts = map.entry(shape.id).or_insert(TimeShape::Empty);
				merge(shape.shape, t, ts);
			}
		}
		map
	}

	fn quadratic(a: f32, b: f32, c: f32) -> Option<(f32, f32)> {
		let d = b * b - 4.0 * a * c;
		if d < 0.0 || a == 0.0 {
			None
		} else {
			let dr = d.sqrt();
			let r1 = (-b - dr) / (2.0 * a);
			let r2 = (-b + dr) / (2.0 * a);
			Some(if r1 < r2 { (r1, r2) } else { (r2, r1) })
		}
	}

	fn find_cone_intersect<T: Interpolate + Positionable>(s0: &TimedVertex<T>, s1: &TimedVertex<T>, timecone: &TimeCone) -> Option<TimedVertex<T>> {
		let TimePos { time: t0, position: p0 } = s0.timepos();
		let TimePos { time: t1, position: p1 } = s1.timepos();
		assert!(t0 < t1);
		// TimeCone C defines a time field
		//     T(p) = C.present + C.slope * || p - C.center ||
		//     C.present : time
		//     C.slope : time / distance
		//     (C.slope will be negative)
		// Want to find a TimePos interpolated between s0 and s1 by a where
		//     T(s0.position ~a~ s1.position) = s0.time ~a~ s1.time
		let k0 = (t0 - timecone.present) / timecone.slope; // distance
		let k1 = (t1 - t0) / timecone.slope; // distance
		let v0 = p0 - timecone.center; // vector
		let v1 = p1 - p0; // vector

		let a = k1 * k1 - v1.dot(&v1); // distance^2
		let b = 2.0 * k0 * k1 - 2.0 * v0.dot(&v1); // distance^2
		let c = k0 * k0 - v0.dot(&v0); // distance^2

		let r = quadratic(a, b, c);
		r.and_then(|(a1, a2)| {
			if a2 <= 1.0 && a2 >= 0.0 {
				let i2 = s0.interpolate(&s1, a2);
				if i2.time <= timecone.present {
					return Some(i2);
				}
			}
			if a1 <= 1.0 && a1 >= 0.0 {
				let i1 = s0.interpolate(&s1, a1);
				if i1.time <= timecone.present {
					return Some(i1);
				}
			}
			None
		}).or_else(|| {
			// TODO: doublecheck this part, taken from the old line_cone_intersect but I
			//       can't remember exactly what it's supposed to be doing.  I think it
			//       exists to prevent the player at the focus from flickering out?
			// Check if the minimum of the parabola is near enough to 0
			let amin = -b / (2.0 * a); // Division by 0 is handled by Infinity
			if amin >= 0.0 && amin <= 1.0 {
				let r = (v0 + v1 * amin).norm() - (k0 + k1 * amin);
				if r.abs() <= 0.01 {
					return Some(s0.interpolate(&s1, amin));
				}
			}
			None
		})
	}

	fn find_point_timepos<T: Interpolate + Positionable>(timevertices: Vec<TimedVertex<T>>, timecone: &TimeCone) -> Option<TimedVertex<T>> {
		for (s0, s1) in timevertices.iter().rev().skip(1).zip(timevertices.iter().rev()) {
			match find_cone_intersect(s0, s1, timecone) {
				Some(r) => return Some(r),
				None => (),
			}
		}
		// Check if too far away to have intersected the cone
		if let Some(old) = timevertices.into_iter().next() {
			// assumes that the oldest provided frame is always at time 0.0
			if old.time == 0.0 && time_at_point(old.vertex.get_position(), timecone) < 0.0 {
				return Some(old);
			}
		}
		None
	}

	fn time_shape_to_real(timeshape: TimeShape, timecone: &TimeCone) -> Option<FrameShape> {
		macro_rules! warp_vertices {
			($vertices:expr, $persistent:expr) => {{
				let vertices = $vertices;
				let persistent = $persistent;
				let mut r = Vec::with_capacity(vertices.len());
				for v in vertices {
					match find_point_timepos(v, &timecone) {
						Some(vnew) => r.push(vnew.vertex),
						None => return None,
					}
				}
				GenericFrameShape {
					vertices: r,
					persistent: persistent,
				}
			}}
		}
		Some(match timeshape {
			TimeShape::Solid(GenericTimeShape { vertices, persistent }) => FrameShape::Solid(warp_vertices!(vertices, persistent)),
			TimeShape::Empty => return None,
		})
	}

	pub(in super) fn warp_shapes(frameshapes: Vec<(f32, Vec<FrameShapeId>)>, timecone: &TimeCone) -> Vec<FrameShape> {
		let timeshapes = frame_shapes_to_time_shapes(frameshapes);
		timeshapes.into_iter().map(|(_, v)| v).flat_map(|ts| time_shape_to_real(ts, timecone)).collect()
	}
}

fn vert_circle_indexed(center: Vec2<f32>, radius: f32, n: i32) -> (Vec<Vec2<f32>>, Vec<u16>) {
	let mut vertices = Vec::with_capacity(n as usize + 1);
	let mut indices = Vec::with_capacity(3 * n as usize);
	vertices.push(center);
	for i in 0..n+1 {
		let a = i as f32 * PI32 * 2.0 / n as f32;
		vertices.push(Vec2::new(center.x + radius * a.cos(), center.y + radius * a.sin()));
	}
	let outer_indices = 1..vertices.len();
	for (a, b) in outer_indices.clone().zip(outer_indices.cycle().skip(1)) {
		indices.push(0);
		indices.push(a as u16);
		indices.push(b as u16);
	}
	(vertices, indices)
}

fn vert_arc_indexed(c: Vec2<f32>, r1: f32, r2: f32, a1: f32, a2: f32, num_segments: i32) -> (Vec<Vec2<f32>>, Vec<u16>) {
	let mut vertices = Vec::with_capacity((num_segments as usize + 1) * 2);
	let mut indices = Vec::with_capacity(num_segments as usize * 6);

	assert!(a2 >= a1);

	fn p(c: Vec2<f32>, r: f32, a: f32) -> Vec2<f32> {
		Vec2::new(c.x + r * a.cos(), c.y + r * a.sin())
	}

	for i in 0..num_segments + 1 {
		let a = a1 + (a2 - a1) * (i as f32) / (num_segments as f32);
		vertices.push(p(c, r1, a)); // inner
		vertices.push(p(c, r2, a)); // outer
	}

	let mut vi = 0;
	for _ in 0..num_segments {
		indices.push(vi + 0);
		indices.push(vi + 3);
		indices.push(vi + 1);
		indices.push(vi + 0);
		indices.push(vi + 2);
		indices.push(vi + 3);
		vi += 2;
	}
	
	(vertices, indices)
}

fn simple_circle(position: Vec2<f32>, radius: f32, n: i32, color: Vec4<f32>) -> warp_rendering::FrameShape {
	let (circle_verts, circle_indices) = vert_circle_indexed(position, radius, n);
	let circle_verts = circle_verts.into_iter().map(|v| warp_rendering::SimpleVertex { position: v } ).collect();
	warp_rendering::solid_shape(circle_verts, Some(circle_indices), color)
}

fn simple_arc(position: Vec2<f32>, r1: f32, r2: f32, a1: f32, a2: f32, n: i32, color: Vec4<f32>) -> warp_rendering::FrameShape {
	let (arc_verts, arc_indices) = vert_arc_indexed(position, r1, r2, a1, a2, n);
	let arc_verts = arc_verts.into_iter().map(|v| warp_rendering::SimpleVertex { position: v } ).collect();
	warp_rendering::solid_shape(arc_verts, Some(arc_indices), color)
}

fn get_hitcircle_shape(id_base: u64, hitcircle: &game::Hitcircle) -> warp_rendering::FrameShapeId {
	let (filled, color) = match hitcircle.state {
		HitcircleState::Attack => (true, color(0xff0000ff)),
		HitcircleState::Parry => (false, color(0x0000ffff)),
		_ => (true, color(0x77777777)),
	};

	let pos = hitcircle.position.from_game();
	let radius = hitcircle.radius.from_game();

	if filled {
		warp_rendering::FrameShapeId {
			id: id_base + 0,
			shape: simple_circle(pos, radius, 32, color)
		}
	} else {
		warp_rendering::FrameShapeId {
			id: id_base + 1,
			shape: simple_arc(pos, radius - 5.0, radius, 0.0, PI32 * 2.0, 32, color)
		}
	}
}

fn vert_line_indexed(points: &[Vec2<f32>], width: f32) -> (Vec<Vec2<f32>>, Vec<u16>) {
	let num_points = points.len();
	let mut vertices = Vec::with_capacity(num_points * 2);
	let mut indices = Vec::with_capacity((num_points - 1) * 6);

	if points.len() < 2 {
		return (vertices, indices);
	}

	let mut prev_point = points[0] * 2.0 - points[1];
	for &p in points {
		let perp = vec_cross_z((p - prev_point).normalize()) * width / 2.0;
		vertices.push(p + perp);
		vertices.push(p - perp);
		prev_point = p;
	}

	let mut vi = 0;
	for _ in 0..points.len() - 1 {
		indices.push(vi + 0);
		indices.push(vi + 3);
		indices.push(vi + 1);
		indices.push(vi + 0);
		indices.push(vi + 2);
		indices.push(vi + 3);
		vi += 2;
	}

	(vertices, indices)
}
fn simple_line(points: &[Vec2<f32>], width: f32, color: Vec4<f32>) -> warp_rendering::FrameShape {
	let (line_verts, line_indices) = vert_line_indexed(&points, width);
	let line_verts = line_verts.into_iter().map(|v| warp_rendering::SimpleVertex { position: v } ).collect();
	warp_rendering::solid_shape(line_verts, Some(line_indices), color)
}

fn cubic_bezier_points(p0: Vec2<f32>, p1: Vec2<f32>, p2: Vec2<f32>, n: i32) -> Vec<Vec2<f32>> {
	(0..n).map(move |i| {
		let t = i as f32 / (n - 1) as f32;
		let q0 = Interpolate::interpolate(&p0, &p1, t);
		let q1 = Interpolate::interpolate(&p1, &p2, t);
		Interpolate::interpolate(&q0, &q1, t)
	}).collect()
}
fn get_arm_shape(entity: &game::Entity, fist: &game::Hitcircle, attach_angle: f32) -> warp_rendering::FrameShape {
	let shoulder_vec = vec_from_angle(entity.facing.from_game() + attach_angle) * entity.radius.from_game();
	let fist_pos = fist.position.from_game();
    let arm_pos = entity.position.from_game() + shoulder_vec;
    let elbow_pos = arm_pos + shoulder_vec;
    let hand_pos = {
        let v = fist_pos - elbow_pos;
        if v.sqnorm() < 60.0 * 60.0 {
            fist_pos
        } else {
            elbow_pos + v.normalize() * 60.0
        }
    };
    let points = cubic_bezier_points(arm_pos, elbow_pos, hand_pos, 10);
	simple_line(&points, 2.0, color(0x000000ff))
}

fn get_entity_shapes(entity: &game::Entity) -> Vec<warp_rendering::FrameShapeId> {
	const ENTITY_SHAPES: u64 = 256;
	let base_id = entity.id as u64;
	let mut shapes = vec![];
	let pos = entity.position.from_game();
	let radius = entity.radius.from_game();
	let stats = &entity.stats;

	{
		let mut add = |offset: u8, shape: warp_rendering::FrameShape| {
			shapes.push(warp_rendering::FrameShapeId {
				id: base_id * ENTITY_SHAPES + offset as u64,
				shape: shape
			});
		};
		// Background
		add(0, simple_circle(pos, radius, 32, color(0x000000FF)));

		// Health ring (5px width)
		{
			let mid = (stats.health.from_game() / stats.health_max.from_game() * PI32 * 2.0).max(0.0).min(PI32 * 2.0);
			add(1, simple_arc(pos, radius - 7.0, radius - 2.0, -PI32 / 2.0, mid - PI32 / 2.0, 32, color(0xFF3C3CFF)));
			if let Some(v) = entity.stats.health_display.query() {
				let end = (v.from_game() / stats.health_max.from_game() * PI32 * 2.0).max(0.0).min(PI32 * 2.0);
				add(2, simple_arc(pos, radius - 7.0, radius - 2.0,  mid - PI32 / 2.0, end - PI32 / 2.0, 32, color(0xCD0021FF)));
			}
		}

		// Stamina arc (5px width, 2px margins)
		{
			let angle = (stats.stamina.from_game() / stats.stamina_max.from_game() * PI32 * 2.0).max(-PI32 * 2.0).min(PI32 * 2.0);
			let (start, end, c) = if angle >= 0.0 {
				(0.0, angle, color(0xffffffff))
			} else {
				(angle, 0.0, color(0x333333ff))
			};

			if let Some(v) = entity.stats.stamina_display.query() {
				let high = (v.from_game() / stats.stamina_max.from_game() * PI32 * 2.0).max(0.0).min(PI32 * 2.0);
				add(3, simple_arc(pos, radius - 14.0, radius - 9.0,  end - PI32 / 2.0, high - PI32 / 2.0, 32, color(0x333333FF)));
			}
			add(4, simple_arc(pos, radius - 14.0, radius - 9.0,  start - PI32 / 2.0, end - PI32 / 2.0, 32, c));

			let cap_width = 3.0;
			let cap_arc = cap_width / (radius - 15.0);

			add(5, simple_arc(pos, radius - 14.0, radius - 9.0,  0.0 - PI32 / 2.0 + angle - cap_arc / 2.0, cap_arc - PI32 / 2.0 + angle - cap_arc / 2.0, 32, color(0xffffffff)));

			/*
			// If staggered, draw rotating dashed segments over the meter
			if let EntityState::Staggered { ttl } = entity.state {
				fn offset_f(ttl: u32) -> f32 {
					let t = ttl as f32 / 60.0;
					let a = if t >= 1.0 {
						1.0 + 2.0 * (t - 1.0)
					} else {
						t.powf(2.0)
					};
					a * PI32 * 2.0 / 2.0
				}
				let offset = offset_f(ttl);
				let dash_length = 15.0;
				let num_segments = ((entity.radius.from_game() - 10.0) * PI32 * 2.0 / dash_length) as u32;
				let full_segment_arc = PI32 * 2.0 / num_segments as f32;
				let segment_arc = full_segment_arc / 2.0;
				for i in 0..num_segments {
					let a = offset + full_segment_arc * i as f32;
					dynamic_shapes::draw_arc(&mut batch, pos, radius - 13.0, radius - 10.0,  a - PI32 / 2.0 + angle - cap_arc / 2.0 - segment_arc / 2.0, a - PI32 / 2.0 + angle - cap_arc / 2.0 + segment_arc / 2.0, color(0x666666ff));
				}
			}
			*/
		}

		let facing = entity.facing.from_game();
		if let EntityState::Idle { blocking: true, .. } = entity.state {
			let angle = BLOCK_SPREAD.from_game();
			add(6, simple_arc(pos, radius - 9.0, radius,  0.0 + facing - angle / 2.0, angle.abs() + facing - angle / 2.0, 16, color(0x00ff0099)));
		}

		let eye_center = pos + vec_from_angle(facing) * (radius - 10.0);
		let eye_offset = vec_cross_z(vec_from_angle(facing)) * 10.0;
		let left_eye = eye_center + eye_offset; // + left_eye_offset;
		let right_eye = eye_center - eye_offset; // + right_eye_offset;
		add(7, simple_circle(left_eye, 5.0, 10, color(0x000000FF)));
		add(8, simple_circle(right_eye, 5.0, 10, color(0x000000FF)));
		add(9, simple_circle(left_eye, 3.0, 10, color(0xFFFFFFFF)));
		add(10, simple_circle(right_eye, 3.0, 10, color(0xFFFFFFFF)));
	}

	for (i, fist) in entity.fists.iter().enumerate() {
		let fist_base_id = i as u64 * 3 + (entity.id as u64) * 1024 + (1 << 32) * 2;
		shapes.push(get_hitcircle_shape(fist_base_id, fist));
		let attach_angle = entity.weapon.fists.get(i).and_then(|f| f.attach_angle);
		if let Some(attach_angle) = attach_angle {
			shapes.push(warp_rendering::FrameShapeId {
				id: fist_base_id + 2,
				shape: get_arm_shape(entity, fist, attach_angle.from_game())
			});
		}
	}

	shapes
}

fn get_frame_shapes(state: &game::GameState) -> Vec<warp_rendering::FrameShapeId> {
	let mut shapes = vec![];
	for entity in state.entities.iter() {
		shapes.extend(get_entity_shapes(entity).into_iter());
	}
	for projectile in state.projectiles.iter() {
		shapes.push(get_hitcircle_shape(projectile.id as u64 * 2 + (1 << 32), &projectile.hitcircle));
	}
	shapes
}
fn render_real_shapes(frame: &mut glium::Frame, view: Mat4<f32>, renderers: &mut Renderers, shapes: Vec<warp_rendering::FrameShape>) {
	use drawing::dynamic_geometry::ColoredVertex;
	let mut batch = renderers.triangle_renderer.batch(frame, &view);
	for shape in shapes { match shape {
		warp_rendering::FrameShape::Solid(warp_rendering::GenericFrameShape { vertices, persistent: warp_rendering::ColorAndIndices { color, indices }}) => {
			let color: [f32; 4] = [color.x, color.y, color.z, color.w];
			if let Some(indices) = indices {
				for is in indices.chunks(3) {
					let vs: Vec<_> = is.iter().map(|&i| {
						let v = &vertices[i as usize];
						ColoredVertex { a_position: [v.position.x, v.position.y], a_color: color }
					}).collect();
					batch.draw_triangle(vs[0], vs[1], vs[2]);
				}
			} else {
				for vs in vertices.chunks(3) {
					let vs: Vec<_> = vs.iter().map(|v| ColoredVertex { a_position: [v.position.x, v.position.y], a_color: color }).collect();
					batch.draw_triangle(vs[0], vs[1], vs[2]);
				}
			}
		}	
	}}
	batch.finish();
}

fn render_time_vertices(frame: &mut glium::Frame, view: Mat4<f32>, renderers: &mut Renderers, timecone: TimeCone, states: &[(f32, &game::GameState)]) {
	let frameshapes = states.iter().map(|&(t, state)| (t, get_frame_shapes(state))).collect();
	let realshapes = warp_rendering::warp_shapes(frameshapes, &timecone);
	render_real_shapes(frame, view, renderers, realshapes);
}

fn add_state_effects(v: &Vec<Entity>) -> Vec<Effect> {
	let mut es = vec![];
	for entity in v {
		match entity.state {
			EntityState::Roll { ttl, dir } if ttl > 10 => {
				for _ in 0..2 { es.push(dust_effect(entity.position.from_game(), dir.from_game())); }
			},
			EntityState::Action { .. } => {
				for fist in entity.fists.iter() {
					match fist.state {
						HitcircleState::Charge => {
							for _ in 0..2 { es.push(charge_effect(fist.position.from_game())); }
							/* fn f(t: f32) -> Vec4<f32> { Vec4::new(0.0, 0.0, 0.0, 0.25 * t) }
							es.push(fading_circle_effect(fist.position.from_game(), fist.radius.from_game(), 10, f)); */
						},
						HitcircleState::Attack { .. } => {
							//for _ in 0..2 { es.push(dust_effect(fist.position.from_game(), ::random_direction())); }
							/* fn f(t: f32) -> Vec4<f32> { Vec4::new(1.0, 0.0, 0.0, if t > 0.5 { 0.5 } else { t }) }
							es.push(fading_circle_effect(fist.position.from_game(), fist.radius.from_game(), 10, f)); */
						},
						HitcircleState::Parry => {
							//for _ in 0..2 { es.push(dust_effect(fist.position.from_game(), ::random_direction())); }
							/* fn f(t: f32) -> Vec4<f32> { Vec4::new(0.0, 0.0, 1.0, 0.25 * t) }
							es.push(fading_circle_effect(fist.position.from_game(), fist.radius.from_game(), 10, f)); */
						},
						_ => { },
					}
				}
			},
			_ => { }
		}
	}
	es
}

#[derive(PartialEq, Eq, Copy, Clone)]
enum WeaponTrailKey {
	Fist { entity_id: u32, fist_id: u32 },
	Roll { entity_id: u32 },
	Entity { entity_id: u32 },
}
struct WeaponTrail {
	trail: trail::Trail,
	active: bool,
	flag: bool,
	key: WeaponTrailKey,
}
struct WeaponTrails {
	trails: Vec<WeaponTrail>,
}
impl WeaponTrails {
	fn get_by_key<F>(&mut self, key: WeaponTrailKey, make: F) -> &mut WeaponTrail
	where F: Fn() -> trail::Trail {
		let has = self.trails.iter_mut().find(|c| c.active && c.key == key).is_some();
		if !has {
			let c = WeaponTrail {
				trail: make(),
				key: key,
				active: true,
				flag: true,
			};
			self.trails.push(c);
		}
		self.trails.iter_mut().find(|c| c.active && c.key == key).unwrap()
	}

	fn update(&mut self, frame: &mut glium::Frame, view: &Mat4<f32>, game: &GameState, trail_renderer: &mut trail::TrailRenderer) {
		for c in self.trails.iter_mut() {
			c.trail.fade_all();
			if !c.active {
				c.trail.decay();
			} else {
				c.flag = false;
			}
		}
		self.trails.retain(|c| c.active || !c.trail.done());
		for entity in game.entities.iter() {
			match entity.state {
				EntityState::Action { .. } => {
					for (fi, fist) in entity.fists.iter().enumerate() {
						let trail_color = match fist.state {
							HitcircleState::Attack { .. } => color(0xff000000),
							HitcircleState::Charge => color(0x00000000),
							HitcircleState::Followup => color(0x00000000),
							HitcircleState::Parry => color(0x0000ff00),
						};
						let key = WeaponTrailKey::Fist { entity_id: entity.id, fist_id: fi as u32 };
						let c = self.get_by_key(key, || trail::Trail::new(fist.radius.from_game(), 20, 5, 0.5));
						c.flag = true;
						c.trail.add_point(fist.position.from_game(), trail_color);
					}
				},
				EntityState::Roll { .. } => {
					let key = WeaponTrailKey::Roll { entity_id: entity.id };
					let c = self.get_by_key(key, || trail::Trail::new(entity.radius.from_game(), 20, 5, 0.5));
					c.flag = true;
					c.trail.add_point(entity.position.from_game(), color(0x0000ff00));
				},
				_ => { }
			}
			let key = WeaponTrailKey::Entity { entity_id: entity.id };
			let c = self.get_by_key(key, || trail::Trail::new(entity.radius.from_game(), 20, 5, 0.25));
			c.flag = true;
			c.trail.add_point(entity.position.from_game(), color(0x00000000));
		}
		trail_renderer.start(frame);
		for c in self.trails.iter_mut() {
			if c.active && !c.flag {
				c.active = false;
			}
			trail_renderer.draw_trail(view, frame, &c.trail);
		}
	}
}

impl GameRenderer {
	fn update_simulations(&mut self, game: &game::GameState, focus: Vec2<f32>) {
		self.water.simulate(focus);
		for entity in game.entities.iter() {
			match entity.state {
				EntityState::Idle { vel, .. } => {
					self.water.disturb(&mut self.disturber, entity.position.from_game(), entity.radius.from_game(), -vel.from_game().norm() / 2.0);
				},
				EntityState::Roll { dir, .. } => {
					self.water.disturb(&mut self.disturber, entity.position.from_game() + dir.from_game() * entity.radius.from_game(), 5.0, -3.0);
				},
				EntityState::Action { .. } => {
					for fist in entity.fists.iter() {
						if fist.state == HitcircleState::Attack {
							self.water.disturb(&mut self.disturber, fist.position.from_game(), fist.radius.from_game(), -2.0);
						}
					}
				},
				_ => { }
			}
		}
		self.water.swap();

		self.sand.simulate(focus); // <-- FOCUS CHANGES HERE, FOCUS IS NOW ONLY VALID FOR THE CURRENTLY-BEING-WRITTEN-TO BUFFER
        for entity in game.entities.iter() {
            match entity.state {
                EntityState::Idle { vel, .. } => {
					self.sand.disturb(&mut self.disturber, entity.position.from_game(), entity.radius.from_game(), -vel.from_game().norm() / 5.0);
                },
                EntityState::Roll { dir, .. } => {
					self.sand.disturb(&mut self.disturber, entity.position.from_game(), entity.radius.from_game(), -3.0);
                },
                EntityState::Action { .. } => {
                    for fist in entity.fists.iter() {
                        if fist.state == HitcircleState::Attack {
							self.sand.disturb(&mut self.disturber, fist.position.from_game(), fist.radius.from_game(), -2.0);
                        }
                    }
                },
                _ => { }
            }
        }
		self.sand.swap(); // <-- SWAP HERE BECAUSE DRAW NEEDS FOCUS TO LINE UP WITH THE CURRENTLY-BEING-READ-FROM BUFFER

		self.grass.simulate(focus);
        self.grass.straighten();
        self.grass.wind();
        for entity in game.entities.iter() {
            // grass.crush_area(entity.position, Vec2::new(0.0, 1.0), 60.0);
            match entity.state {
                EntityState::Idle { vel, .. } => {
                    self.grass.crush_area(entity.position.from_game() + vel.from_game().normalize(), vel.from_game(), 0.0, entity.radius.from_game());
                },
                EntityState::Roll { dir, .. } => {
                    // grass.dbg_set_wind(dir);
                    self.grass.crush_area(entity.position.from_game() + dir.from_game(), dir.from_game() * 30.0, 0.0, entity.radius.from_game());
                },
                EntityState::Action { .. } => {
                    for fist in entity.fists.iter() {
                        if fist.state == HitcircleState::Attack {
							// TODO: This should work for all hitcircles (fist and projectile), construct a list of disturbances while drawing entities/projectiles then apply them after
                            self.grass.crush_area(fist.position.from_game(), Vec2::new(0.0, 0.0), 3.0, fist.radius.from_game() + 20.0);
                        }
                    }
                },
                _ => { }
            }
        }
		self.grass.swap();
	}
    pub fn render_state(&mut self, focus_player: Option<u64>, frame: &mut glium::Frame, states: &[&game::GameState], interp_t: f32, dbg_timestops: &mut DbgTimestops) -> (Vec2<f32>, na::Mat4<f32>) {
		let game = states[states.len() - 1];
		let game_prev = states[states.len() - 2];
		let focus_player = focus_player.unwrap_or(0);
		let focus = if let Some((entity, _)) = game.entities.iter().zip(game.controllers.iter()).find(|&(_, controller)| match controller.get_owner() { Some(n) if n == focus_player as usize => true, _ => false }) {
			let entity_prev = game_prev.entities.iter().find(|x| x.id == entity.id).unwrap_or(entity);
			Interpolation::new(entity_prev, entity, interp_t).position()
        } else {
            Vec2::new(0.0, 0.0)
        };
        let (w, h) = { let (w, h) = frame.get_dimensions(); (w as f32, h as f32) };
        let view = na::OrthoMat3::<f32>::new(w, h, -50.0, 50.0).to_mat() * translate(-focus.x, -focus.y, 0.0);
        dbg_timestops.timestop("Got Frame");

		self.water.draw(frame, &view);
		self.sand.draw(frame, &view);
		self.grass.draw_floor(frame, &(view * translate(0.0, 0.0, -50.0)));
		self.grass.draw(&(view * translate(0.0, 0.0, -50.0)), frame);
        dbg_timestops.timestop("Drew Simulations");

		self.update_simulations(&game, focus);
        dbg_timestops.timestop("Updated Simulations");

        self.renderers.text.draw_string(&(view * translate(0.0, -200.0, 50.0) * scale(32.0, 32.0, 1.0)), frame, color(0xff0000ff), "Hello, world?");
        dbg_timestops.timestop("Drew Text");

        // trails.update(&game, &mut trail_renderer, &mut render_frame);
        //for entity in game.entities.iter() {
        //    draw_entity(frame, view, entity, &mut self.renderers, &interpolations, interp_t);
        //}
		self.render_warped_state(frame, view.clone(), states, /* Vec2::new(0.0, 0.0) */ focus, interp_t);
        dbg_timestops.timestop("Drew Entities");
        {
            let mut batch = self.renderers.line_renderer.batch(frame, &view);
            for line in game.lines.iter() {
                batch.draw_line(line.point_a.from_game(), line.point_b.from_game(), color(0x000000ff));
            }
            for line in dbg_lines.lock().unwrap().iter() {
                batch.draw_line(Vec2::new(line[0], line[1]), Vec2::new(line[2], line[3]), color(0xff0000ff));
            }
            batch.finish();
            dbg_lines.lock().unwrap().clear();
        }

        {
            let mut j = 0;
            for i in 0..self.effects.len() {
                if {
                    let ref mut tick = self.effects[i].tick;
                    !tick(frame, &view, &mut self.renderers)
                } {
                    self.effects.swap(j, i);
                    j += 1;
                }
            }
            self.effects.truncate(j);
            // TODO: Fix flushing for all primitives so that it auto flushes the moment something else is drawn.
            self.renderers.sprite_renderer.flush_sprites(&view, &mut self.renderers.textures, frame);
        }

        // dbg_trail.colorize(|t| Vec4::new(0.0, 1.0, 0.0, t));
        (focus, view)
    }

	pub fn render_warped_state(&mut self, frame: &mut glium::Frame, view: Mat4<f32>, states: &[&game::GameState], focus: Vec2<f32>, t: f32) {
		assert!(t >= 0.0);
		assert!(t <= 1.0);
		let debug_time = false;

		// self.renderers.model_renderer.draw_model_at(frame, &view, &self.renderers.model_renderer.circle, Vec2::new(0.0, 0.0), 10.0, 10.0, color(0xFF00FFFF));
		let n = states.len();
		let t_present = n as f32 - 2.0 + t;
		self.renderers.text.draw_string(&(view * scale(6.0, 6.0, 1.0)), frame, color(0xFF00FFFF), &format!("{:.2}", t_present));
		if debug_time {
			{
				/* let mut batch = self.renderers.line_renderer.batch(frame, &view);
				for i in 0..states.len() {
					let r = time_curve(i as f32);
					for j in 0..32 {
						batch.draw_line(focus + ::vec_from_angle(j as f32 * 2.0 * PI32 / 32.0) * r, focus + ::vec_from_angle((j + 1) as f32 * 2.0 * PI32 / 32.0) * r, color(0xFF00FFFF));
					}
				}
				batch.finish(); */
			}
		}
		// TODO: REMOVE THIS once sprite flushing is done, or whatever. This makes the X eyes appear for these tests.
		self.renderers.sprite_renderer.flush_sprites(&view, &mut self.renderers.textures, frame);

		let timecone = TimeCone {
			center: focus,
			present: t_present,
			slope: -1.0 / 100.0, // pretty subtle, 1 frame per 100 units, the warping with something like -1.0 / 40.0 is much more pronounced and looks cool when rolling
		};
		let timestates: Vec<_> = states.iter().enumerate().map(|(i, s)| (i as f32, *s)).collect();
		render_time_vertices(frame, view, &mut self.renderers, timecone, &timestates);
	}

	pub fn add_state_effects(&mut self, game: &game::GameState) {
		self.effects.extend(add_state_effects(&game.entities));
	}
	pub fn add_effect(&mut self, effect: Effect) {
		self.effects.push(effect);
	}
}

/* TODO move this to warped_rendering
#[cfg(test)]
mod tests {
	use super::{line_cone_intersect, IntersectResult};
	use na::Vec2;

	#[test]
	fn test_near_focus() {
		let dr = 100.0;
		let steps = 1000;
		for i in -2000..2001 {
			let ra = i as f32 / steps as f32 * dr;
			let rb = ra + dr;
			let should_not_intersect = ra < 0.0 && rb < 0.0 || ra > 0.0 && rb > 0.0;
			match line_cone_intersect(Vec2::new(0.0, 0.0), Vec2::new(0.0, 0.0), ra, rb) {
				IntersectResult::None if !should_not_intersect => panic!("No intersection for {}, {}", ra, rb),
				IntersectResult::One(..) | IntersectResult::Two(..) if should_not_intersect => panic!("Intersection for {}, {}", ra, rb),
				_ => ()
			}
		}
	}
}
*/