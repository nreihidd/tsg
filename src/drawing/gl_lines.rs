use na::{Vec2, Vec4, Mat4};
use glium::{self, Surface};
use glium::backend::Context;
use std::rc::Rc;
use drawing::premultiplied_blend;
use std::ops::Range;

#[derive(Copy, Clone)]
pub struct LineVertex {
    pub a_position: [f32; 2],
    pub a_color: [f32; 4],
}
implement_vertex!(LineVertex, a_position, a_color);

const NUM_BUFFER_LINES: usize = 4096;
const VERTICES_PER_LINE: usize = 2;
const NUM_BUFFER_VERTICES: usize = NUM_BUFFER_LINES * VERTICES_PER_LINE;

pub struct LineRenderer {
	vertex_buffer: glium::VertexBuffer<LineVertex>,
	program: glium::Program,
    vertices: Vec<LineVertex>, // Used by batch, prevents unnecessary reallocations
}
impl LineRenderer {
    pub fn new(facade: &Rc<Context>) -> LineRenderer {
        LineRenderer {
            vertex_buffer: glium::VertexBuffer::empty_dynamic(facade, NUM_BUFFER_VERTICES).unwrap(),
            program: glium::Program::from_source(facade, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, None).unwrap(),
            vertices: vec![LineVertex { a_position: [0.0, 0.0], a_color: [0.0, 0.0, 0.0, 0.0] }; NUM_BUFFER_VERTICES],
        }
    }
    pub fn batch<'a>(&'a mut self, frame: &'a mut glium::Frame, view: &Mat4<f32>,) -> LineBatch<'a> {
        LineBatch::new(frame, *view, self)
    }
    fn draw_line_vertices(&mut self, frame: &mut glium::Frame, view: &Mat4<f32>, vertex_range: Range<usize>) {
        let vertices = &self.vertices[vertex_range];
        if vertices.len() == 0 { return; }
        let uniforms = uniform! { view: view.as_ref().clone() };
        let params = glium::DrawParameters {
            blend: premultiplied_blend(),
            line_width: Some(2.0),
            depth: glium::Depth {
                test: glium::DepthTest::IfLessOrEqual,
                write: false,
                .. Default::default()
            },
            .. Default::default()
        };
        let slice = self.vertex_buffer.slice(0..vertices.len()).unwrap();
        slice.write(vertices);
        frame.draw(slice, glium::index::IndicesSource::NoIndices { primitives: glium::index::PrimitiveType::LinesList }, &self.program, &uniforms, &params).unwrap();
    }
}

pub struct LineBatch<'a> {
    frame: &'a mut glium::Frame,
    view: Mat4<f32>,
    renderer: &'a mut LineRenderer,
    pending_lines: usize,
}
impl<'a> LineBatch<'a> {
    fn new(frame: &'a mut glium::Frame, view: Mat4<f32>, renderer: &'a mut LineRenderer) -> LineBatch<'a> {
        LineBatch {
            frame: frame,
            view: view,
            renderer: renderer,
            pending_lines: 0,
        }
    }
    pub fn draw_line(&mut self, a: Vec2<f32>, b: Vec2<f32>, color: Vec4<f32>) {
        if self.pending_lines >= NUM_BUFFER_LINES {
            self.flush();
        }
        let f = |v: Vec2<f32>| LineVertex { a_position: [v.x, v.y], a_color: color.as_ref().clone() };
        let (a, b) = (f(a), f(b));
        let i = self.pending_lines * VERTICES_PER_LINE;
        self.renderer.vertices[i + 0] = a;
        self.renderer.vertices[i + 1] = b;
        self.pending_lines += 1;
    }
    fn flush(&mut self) {
        self.renderer.draw_line_vertices(self.frame, &self.view, 0..self.pending_lines * VERTICES_PER_LINE);
        self.pending_lines = 0;
    }
    pub fn finish(self) { }
}
impl<'a> Drop for LineBatch<'a> {
    fn drop(&mut self) {
        self.flush();
    }
}

const VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
	uniform mat4 view;
    in vec2 a_position;
	in vec4 a_color;
	out vec4 v_color;
    void main() {
        gl_Position = view * vec4(a_position, 0.0, 1.0);
		v_color = vec4(a_color.rgb * a_color.a, a_color.a); // Because blending mode is premultiplied
    }
"#;

const FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
	in vec4 v_color;
    out vec4 color;
    void main() {
        color = v_color;
    }
"#;
