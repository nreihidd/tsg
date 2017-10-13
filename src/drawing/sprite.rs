use std::rc::Rc;
use glium::backend::Context;
use glium::{self, Surface};
use na::{Vec4, Vec2, Mat4};
use drawing::{quad_indices, premultiplied_blend};
use drawing::texture::TextureCache;

const VERTICES_PER_SPRITE: usize = 4;
const INDICES_PER_SPRITE: usize = 6;

#[derive(Copy, Clone)]
pub struct TexturedVertex {
    position: [f32; 2],
	uv: [f32; 2],
    a_color: [f32; 4],
}
implement_vertex!(TexturedVertex, position, uv, a_color);

pub struct SpriteRenderer {
	vertices: Vec<TexturedVertex>,
	current_path: &'static str,
	pending_sprites: usize,
	batch_size: usize,
	vertex_buffer: glium::VertexBuffer<TexturedVertex>,
	index_buffer: glium::IndexBuffer<u16>,
	program: glium::Program,
}
impl SpriteRenderer {
	pub fn new(facade: &Rc<Context>) -> SpriteRenderer {
        let num_sprites = 128;
        let num_vertices = num_sprites * VERTICES_PER_SPRITE;
		SpriteRenderer {
			current_path: "",
			batch_size: num_sprites,
			pending_sprites: 0,
			vertices: vec![TexturedVertex { position: [0.0, 0.0], uv: [0.0, 0.0], a_color: [0.0, 0.0, 0.0, 0.0] }; num_vertices],
			vertex_buffer: glium::VertexBuffer::empty_dynamic(facade, num_vertices).unwrap(),
			index_buffer: glium::IndexBuffer::new(facade, glium::index::PrimitiveType::TrianglesList, &quad_indices(num_sprites as u16)).unwrap(),
			program: glium::Program::from_source(facade, TEXTURE_VERTEX_SHADER_SRC, TEXTURE_FRAGMENT_SHADER_SRC, None).unwrap(),
		}
	}
	pub fn draw_sprite(&mut self, frame: &mut glium::Frame, view: &Mat4<f32>, textures: &mut TextureCache, path: &'static str, v: Vec2<f32>, w: f32, h: f32, rotation: f32, color: Vec4<f32>) {
		if path != self.current_path {
			if self.pending_sprites != 0 {
				self.flush_sprites(view, textures, frame);
			}
			self.current_path = path;
		}
		// translate(v.x, v.y, 0.0) * scale(w, h, 1.0) * rotate(rotation);
        let cos = rotation.cos();
        let sin = rotation.sin();
        let transform = |p: [f32; 2]| {
            [
                (p[0] * cos - p[1] * sin) * w + v.x,
                (p[0] * sin + p[1] * cos) * h + v.y,
            ]
        };

        let mut verts = [
			TexturedVertex { position: [ -1.0, -1.0], uv: [  0.0,  0.0 ], a_color: color.as_ref().clone() },
			TexturedVertex { position: [  1.0, -1.0], uv: [  1.0,  0.0 ], a_color: color.as_ref().clone() },
			TexturedVertex { position: [  1.0,  1.0], uv: [  1.0,  1.0 ], a_color: color.as_ref().clone() },
			TexturedVertex { position: [ -1.0,  1.0], uv: [  0.0,  1.0 ], a_color: color.as_ref().clone() },
		];
        let i = self.pending_sprites * VERTICES_PER_SPRITE;
        for (o, vert) in verts.iter_mut().enumerate() {
            vert.position = transform(vert.position);
            self.vertices[i + o] = *vert;
        }

		self.pending_sprites += 1;
		if self.pending_sprites == self.batch_size {
			self.flush_sprites(view, textures, frame);
		}
	}
	pub fn flush_sprites(&mut self, view: &Mat4<f32>, textures: &mut TextureCache, frame: &mut glium::Frame) {
		if self.pending_sprites == 0 { return; }
		let texture = textures.get(self.current_path);
		let uniforms = uniform! { view: view.as_ref().clone(), uTexture: texture.sampled() };
		let num_indices = self.pending_sprites * INDICES_PER_SPRITE;
		let num_vertices = self.pending_sprites * VERTICES_PER_SPRITE;
		self.vertex_buffer.slice(0..num_vertices).unwrap().write(&self.vertices[0..num_vertices]);
		let params = glium::DrawParameters {
			blend: premultiplied_blend(),
			depth: glium::Depth {
				test: glium::DepthTest::IfLessOrEqual,
				write: false,
				.. Default::default()
			},
			.. Default::default()
		};
		frame.draw(&self.vertex_buffer, self.index_buffer.slice(0..num_indices).unwrap(), &self.program, &uniforms, &params).unwrap();
		self.pending_sprites = 0;
	}
}

const TEXTURE_VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
	uniform mat4 view;
    in vec2 position;
	in vec2 uv;
    in vec4 a_color;
	out vec2 vUV;
    out vec4 v_color;
    void main() {
        gl_Position = view * vec4(position, 0.0, 1.0);
		vUV = uv;
        v_color = a_color;
    }
"#;

const TEXTURE_FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
	uniform sampler2D uTexture;
	in vec2 vUV;
    in vec4 v_color;
    out vec4 color;
    void main() {
        vec4 sampled = texture(uTexture, vUV);
        // because v_color is not premultiplied!
        sampled.rgb *= v_color.rgb;
        color = sampled * v_color.a;
        // otherwise, if v_color were premultiplied, it would be: color = sampled * v_color;
    }
"#;