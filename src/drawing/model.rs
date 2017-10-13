use glium::{self, Surface};
use std::rc::Rc;
use glium::backend::Context;
use na::{self, Vec4, Vec2};
use std::f32::consts::PI as PI32;
use super::{premultiplied_blend, translate, rotate, scale, quad_indices};

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 2],
}
implement_vertex!(Vertex, position);
pub struct Model<T: glium::vertex::Vertex> {
	pub vertex_buffer: glium::VertexBuffer<T>,
	pub index_buffer: glium::IndexBuffer<u16>,
}

const NUM_ARC_VERTICES: usize = 64;

pub struct ModelRenderer {
    pub circle: Model<Vertex>,
    pub rect: Model<Vertex>,
    arc_buffer: glium::VertexBuffer<Vertex>,
	program: glium::Program,
}
impl ModelRenderer {
	pub fn new(ctx: &Rc<Context>) -> ModelRenderer {
		ModelRenderer {
			program: glium::Program::from_source(ctx, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, None).unwrap(),
            arc_buffer: glium::VertexBuffer::<Vertex>::empty_dynamic(ctx, NUM_ARC_VERTICES).unwrap(),
            circle: make_circle_model(ctx, 32),
            rect: make_rect_model(ctx),
		}
	}
    fn params<'a>() -> glium::DrawParameters<'a> {
        glium::DrawParameters {
            blend: premultiplied_blend(),
            depth: glium::Depth {
                test: glium::DepthTest::IfLessOrEqual,
                write: false,
                .. Default::default()
            },
            .. Default::default()
        }
    }
    pub fn draw_model_no_indices(&self, frame: &mut glium::Frame, view: &na::Mat4<f32>, vertex_buffer: &glium::VertexBuffer<Vertex>, color: Vec4<f32>) {
		let uniforms = uniform! { view: view.as_ref().clone(), uColor: color.as_ref().clone() };
		frame.draw(vertex_buffer, glium::index::IndicesSource::NoIndices { primitives: glium::index::PrimitiveType::TrianglesList }, &self.program, &uniforms, &ModelRenderer::params()).unwrap();
    }
	pub fn draw_model(&self, frame: &mut glium::Frame, view: &na::Mat4<f32>, model: &Model<Vertex>, color: Vec4<f32>) {
		let uniforms = uniform! { view: view.as_ref().clone(), uColor: color.as_ref().clone() };
		frame.draw(&model.vertex_buffer, &model.index_buffer, &self.program, &uniforms, &ModelRenderer::params()).unwrap();
	}
	pub fn draw_model_at(&self, frame: &mut glium::Frame, view: &na::Mat4<f32>, model: &Model<Vertex>, v: Vec2<f32>, w: f32, h: f32, color: Vec4<f32>) {
		let view = *view * translate(v.x, v.y, 0.0) * scale(w, h, 1.0);
		self.draw_model(frame, &view, model, color);
	}
    pub fn draw_arc(&mut self, frame: &mut glium::Frame, view: &na::Mat4<f32>, color: na::Vec4<f32>, inner_radius: f32, outer_radius: f32, theta: f32) {
    	let uniforms = uniform! {
    		view: view.as_ref().clone(),
    		uColor: color.as_ref().clone(),
    	};
    	let verts = arc_vertices(inner_radius, outer_radius, theta, NUM_ARC_VERTICES);
    	let buffer_slice = self.arc_buffer.slice(0..verts.len()).unwrap();
    	buffer_slice.write(&verts);
    	frame.draw(buffer_slice, glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip), &self.program, &uniforms, &ModelRenderer::params()).unwrap();
    }
    pub fn draw_arc_angles(&mut self, frame: &mut glium::Frame, view: &na::Mat4<f32>, color: na::Vec4<f32>, inner_radius: f32, outer_radius: f32, start: f32, end: f32) {
    	let view = *view * rotate(start);
    	self.draw_arc(frame, &view, color, inner_radius, outer_radius, end - start);
    }
}

fn make_circle_model(facade: &Rc<Context>, n: u16) -> Model<Vertex> {
	use std::f32::consts::PI;
	let mut vertices = vec![Vertex { position: [0.0, 0.0f32] }];
	let mut indices = vec![0u16];
	for i in 0..n+1 {
		let a = i as f32 * PI * 2.0 / n as f32;
		vertices.push(Vertex { position: [a.cos(), a.sin()] });
		indices.push(i + 1);
	}

	Model {
		vertex_buffer: glium::VertexBuffer::new(facade, &vertices).unwrap(),
		index_buffer: glium::index::IndexBuffer::new(facade, glium::index::PrimitiveType::TriangleFan, &indices).unwrap(),
	}
}

fn make_rect_model(facade: &Rc<Context>) -> Model<Vertex> {
	let vertices = vec![
		Vertex { position: [ -1.0, -1.0] },
		Vertex { position: [  1.0, -1.0] },
		Vertex { position: [  1.0,  1.0] },
		Vertex { position: [ -1.0,  1.0] },
	];

	let indices = quad_indices(1);

	Model {
		vertex_buffer: glium::VertexBuffer::new(facade, &vertices).unwrap(),
		index_buffer: glium::index::IndexBuffer::new(facade, glium::index::PrimitiveType::TrianglesList, &indices).unwrap(),
	}
}

fn arc_vertices(inner_radius: f32, outer_radius: f32, theta: f32, max_vertices: usize) -> Vec<Vertex> {
	let mut v = vec![];

	// Starting vertex
	v.push(Vertex { position: [inner_radius, 0.0] });

	// Intermediate vertices
	let mut outside = true;
	let n = max_vertices - 3;
	for i in 0..n {
		let a = i as f32 * PI32 * 2.0 / n as f32;
		if a >= theta { break; }
		let r = if outside { outer_radius } else { inner_radius };
		v.push(Vertex { position: [r * a.cos(), r * a.sin()]});
		outside = !outside;
	}

	// Cap vertices
	let outer_cap = Vertex { position: [outer_radius * theta.cos(), outer_radius * theta.sin()]};
	let inner_cap = Vertex { position: [inner_radius * theta.cos(), inner_radius * theta.sin()]};
	if outside {
		v.push(outer_cap); v.push(inner_cap);
	} else {
		v.push(inner_cap); v.push(outer_cap);
	}
	v
}

const VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
	uniform mat4 view;
    in vec2 position;
    void main() {
        gl_Position = view * vec4(position, 0.0, 1.0);
    }
"#;

const FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
	uniform vec4 uColor;
    out vec4 color;
    void main() {
		vec4 c = uColor;
		c.rgb *= c.a;
        color = c;
    }
"#;