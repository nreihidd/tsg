use na::{Vec2, Vec4, Mat4, Norm};
use glium::{self, Surface};
use glium::backend::Context;
use std::rc::Rc;
use drawing::premultiplied_blend;
use std::ops::Range;
use {vec_cross_z};

#[derive(Copy, Clone)]
pub struct ColoredVertex {
    pub a_position: [f32; 2],
    pub a_color: [f32; 4],
}
implement_vertex!(ColoredVertex, a_position, a_color);

const NUM_BUFFER_TRIANGLES: usize = 4096;
const VERTICES_PER_TRIANGLE: usize = 3;
const NUM_BUFFER_VERTICES: usize = NUM_BUFFER_TRIANGLES * VERTICES_PER_TRIANGLE;

pub struct DynamicRenderer {
	vertex_buffer: glium::VertexBuffer<ColoredVertex>,
	program: glium::Program,
    vertices: Vec<ColoredVertex>, // Used by batch, prevents unnecessary reallocations
}
impl DynamicRenderer {
    pub fn new(facade: &Rc<Context>) -> DynamicRenderer {
        DynamicRenderer {
            vertex_buffer: glium::VertexBuffer::empty_dynamic(facade, NUM_BUFFER_VERTICES).unwrap(),
            program: glium::Program::from_source(facade, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, None).unwrap(),
            vertices: vec![ColoredVertex { a_position: [0.0, 0.0], a_color: [0.0, 0.0, 0.0, 0.0] }; NUM_BUFFER_VERTICES],
        }
    }
    pub fn batch<'a>(&'a mut self, frame: &'a mut glium::Frame, view: &Mat4<f32>,) -> DynamicBatch<'a> {
        DynamicBatch::new(frame, *view, self)
    }
    pub fn draw_triangles<I>(&mut self, frame: &mut glium::Frame, view: &Mat4<f32>, triangles: I)
        where I: Iterator<Item=(ColoredVertex, ColoredVertex, ColoredVertex)> {
        let mut batch = self.batch(frame, view);
        for (a, b, c) in triangles {
            batch.draw_triangle(a, b, c);
        }
        batch.finish()
    }
    pub fn draw_lines<I>(&mut self, frame: &mut glium::Frame, view: &Mat4<f32>, lines: I)
        where I: Iterator<Item=(Vec2<f32>, Vec2<f32>, Vec4<f32>)> {
        let mut batch = self.batch(frame, view);
        for (a, b, c) in lines {
            batch.draw_line(a, b, c);
        }
        batch.finish()
    }
    fn draw_triangle_vertices(&mut self, frame: &mut glium::Frame, view: &Mat4<f32>, vertex_range: Range<usize>) {
        let vertices = &self.vertices[vertex_range];
        if vertices.len() == 0 { return; }
        let uniforms = uniform! { view: view.as_ref().clone() };
        let params = glium::DrawParameters {
            blend: premultiplied_blend(),
            depth: glium::Depth {
                test: glium::DepthTest::IfLessOrEqual,
                write: false,
                .. Default::default()
            },
            .. Default::default()
        };
        let slice = self.vertex_buffer.slice(0..vertices.len()).unwrap();
        slice.write(vertices);
        frame.draw(slice, glium::index::IndicesSource::NoIndices { primitives: glium::index::PrimitiveType::TrianglesList }, &self.program, &uniforms, &params).unwrap();
    }
}

pub struct DynamicBatch<'a> {
    frame: &'a mut glium::Frame,
    view: Mat4<f32>,
    renderer: &'a mut DynamicRenderer,
    pending_triangles: usize,
}
impl<'a> DynamicBatch<'a> {
    fn new(frame: &'a mut glium::Frame, view: Mat4<f32>, renderer: &'a mut DynamicRenderer) -> DynamicBatch<'a> {
        DynamicBatch {
            frame: frame,
            view: view,
            renderer: renderer,
            pending_triangles: 0,
        }
    }
    pub fn draw_triangle(&mut self, a: ColoredVertex, b: ColoredVertex, c: ColoredVertex) {
        if self.pending_triangles >= NUM_BUFFER_TRIANGLES {
            self.flush();
        }
        let i = self.pending_triangles * VERTICES_PER_TRIANGLE;
        self.renderer.vertices[i + 0] = a;
        self.renderer.vertices[i + 1] = b;
        self.renderer.vertices[i + 2] = c;
        self.pending_triangles += 1;
    }
    pub fn draw_line(&mut self, a: Vec2<f32>, b: Vec2<f32>, color: Vec4<f32>) {
        let dir = (b - a).normalize();
        let dirc = vec_cross_z(dir);
        let f = |v: Vec2<f32>| ColoredVertex { a_position: [v.x, v.y], a_color: color.as_ref().clone() };
        let v1 = f(a - dirc);
        let v2 = f(a + dirc);
        let v3 = f(b + dirc);
        let v4 = f(b - dirc);
        self.draw_triangle(v1, v2, v3);
        self.draw_triangle(v1, v3, v4);
    }
    fn flush(&mut self) {
        self.renderer.draw_triangle_vertices(self.frame, &self.view, 0..self.pending_triangles * VERTICES_PER_TRIANGLE);
        self.pending_triangles = 0;
    }
    pub fn finish(self) { }
}
impl<'a> Drop for DynamicBatch<'a> {
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
