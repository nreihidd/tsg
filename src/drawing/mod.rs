use image;
use glium::{self};
use na::{self, Vec2, Vec4, Mat4, ToHomogeneous, Diag};

pub mod glium_sdl2;
pub mod geometry;
pub mod simulation;
pub mod grass;
pub mod water;
pub mod sand;
pub mod text;
pub mod trail;
pub mod model;
pub mod texture;
pub mod sprite;
pub mod dynamic_geometry;
pub mod gl_lines;

pub fn load_image_srgb(path: &str) -> glium::texture::RawImage2d<u8> {
	let mut i = image::open(path).unwrap().to_rgba();
	let d = i.dimensions();
	for pixel in i.pixels_mut() {
		// http://ssp.impulsetrain.com/gamma-premult.html
		let alpha = (pixel[3] as f32 / 255.0).powf(1.0/2.2);
		pixel[0] = (pixel[0] as f32 * alpha) as u8;
		pixel[1] = (pixel[1] as f32 * alpha) as u8;
		pixel[2] = (pixel[2] as f32 * alpha) as u8;
	}
	glium::texture::RawImage2d::from_raw_rgba_reversed(i.into_raw(), d)
}

pub fn load_image(path: &str) -> glium::texture::RawImage2d<u8> {
	let mut i = image::open(path).unwrap().to_rgba();
	let d = i.dimensions();
	for pixel in i.pixels_mut() {
		pixel[0] = (pixel[0] as u16 * pixel[3] as u16 / 255) as u8;
		pixel[1] = (pixel[1] as u16 * pixel[3] as u16 / 255) as u8;
		pixel[2] = (pixel[2] as u16 * pixel[3] as u16 / 255) as u8;
	}
	glium::texture::RawImage2d::from_raw_rgba_reversed(i.into_raw(), d)
}

// http://www.realtimerendering.com/blog/gpus-prefer-premultiplication/
pub fn premultiplied_blend() -> glium::Blend {
	let equation = glium::BlendingFunction::Addition {
		source: glium::LinearBlendingFactor::One,
		destination: glium::LinearBlendingFactor::OneMinusSourceAlpha,
	};
	glium::Blend {
		color: equation,
		alpha: equation,
		.. Default::default()
	}
}


pub fn compute_program<T: glium::backend::Facade>(facade: &T, vertex: &'static str, fragment: &'static str) -> Result<glium::program::Program, glium::program::ProgramCreationError> {
    glium::Program::new(facade, glium::program::ProgramCreationInput::SourceCode {
        vertex_shader: vertex,
        fragment_shader: fragment,
        tessellation_control_shader: None,
        tessellation_evaluation_shader: None,
        geometry_shader: None,
        transform_feedback_varyings: None,
        outputs_srgb: true,
        uses_point_size: false,
    })
}

/// RGBA
pub fn color(c: u32) -> Vec4<f32> {
	fn f(i: u32) -> f32 {
		(i as u8) as f32 / 255.0
	}
	Vec4::new(f(c >> 24), f(c >> 16), f(c >> 8), f(c))
}

pub fn scale(sx: f32, sy: f32, sz: f32) -> Mat4<f32> {
	Mat4::from_diag(&Vec4::new(sx, sy, sz, 1.0))
}

pub fn translate(x: f32, y: f32, z: f32) -> Mat4<f32> {
	na::Iso3::new(na::Vec3::new(x, y, z), na::Vec3::new(0.0, 0.0, 0.0f32)).to_homogeneous()
}

pub fn rotate(a: f32) -> Mat4<f32> {
	na::Iso3::new(na::Vec3::new(0.0, 0.0, 0.0f32), na::Vec3::new(0.0, 0.0f32, a)).to_homogeneous()
}

pub fn quad_indices(num_quads: u16) -> Vec<u16> {
	let i = [0, 1, 2, 0, 2, 3u16];
	(0..num_quads).flat_map(|n| i.iter().map(move |x| x + n * 4)).collect()
}

fn is_polygon_clockwise(polygon: &[Vec2<f32>]) -> bool {
	// https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
	polygon.iter().zip(polygon.iter().cycle().skip(1)).map(|(&a, &b)| (b.x - a.x) * (b.y + a.y)).fold(0.0, |acc, x| acc + x) > 0.0
}

pub fn polygon_to_triangles(polygon: &[Vec2<f32>]) -> Vec<[usize; 3]> {
	use std::iter::FromIterator;
	// Straightforward O(n^3) implementation of http://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
	struct Vertex {
		v: Vec2<f32>,
		i: usize,
	}
	let clockwise = is_polygon_clockwise(polygon);
	let mut polygon = Vec::from_iter(polygon.iter().enumerate().map(|(i, &v)| Vertex { v: v, i: i }));
	let mut triangles = vec![];
	for _ in 0..polygon.len() - 2 {
		let mut index_to_remove = 0;
		// Find ear
		for (i, b) in polygon.iter().enumerate() {
			let ref a = polygon[(polygon.len() + i - 1) % polygon.len()];
			let ref c = polygon[(i + 1) % polygon.len()];
			let triangle = [a.v, b.v, c.v];
			let triangle_indices = [a.i, b.i, c.i];
			let is_convex = clockwise == (na::cross(&(a.v - b.v), &(c.v - b.v)).x > 0.0);
			if !is_convex { continue; }
			let is_ear = polygon.iter()
				.filter(|v| triangle_indices.iter().all(|&t_i| t_i != v.i)) // Exclude checking the triangle's vertices
				.all(|v| !point_in_triangle(v.v, &triangle));
			if is_ear {
				triangles.push([a.i, b.i, c.i]);
				index_to_remove = i;
				break;
			}
		}
		polygon.remove(index_to_remove);
	}
	triangles
}

fn point_in_triangle(p: Vec2<f32>, triangle: &[Vec2<f32>; 3]) -> bool {
	let num_positive = triangle.iter()
		.zip(triangle.iter().cycle().skip(1))
		.map(|(&a, &b)| na::cross(&(b - a), &(p - a)).x > 0.0)
		.fold(0, |acc, s| acc + if s { 1 } else { 0 });
	num_positive == 3 || num_positive == 0
}