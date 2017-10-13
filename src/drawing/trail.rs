use std::collections::VecDeque;
use std::rc::Rc;
use std::default::Default;
use na::{self, Norm};
use drawing::{premultiplied_blend};
use glium::{self, Surface};
use glium::backend::Context;
pub type Vec2 = na::Vec2<f32>;
pub type Vec4 = na::Vec4<f32>;
pub type Mat4 = na::Mat4<f32>;

const MAX_SEGMENTS: usize = 1024;
const MAX_VERTICES: usize = (MAX_SEGMENTS - 1) * 4;
const MAX_INDICES: usize = MAX_SEGMENTS * 6;

#[derive(Copy, Clone)]
struct TrailVertex {
    a_position: [f32; 2],
    a_color: [f32; 4],
}
implement_vertex!(TrailVertex, a_position, a_color);

const TRAIL_VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
	uniform mat4 u_view;
    in vec2 a_position;
    in vec4 a_color;
    out vec4 v_color;
    void main() {
        gl_Position = u_view * vec4(a_position, 0.0, 1.0);
        vec4 c = a_color;
        c.rgb *= c.a;
        v_color = c;
    }
"#;

const TRAIL_FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
	in vec4 v_color;
    out vec4 color;
    void main() {
        color = v_color;
    }
"#;

#[derive(Copy, Clone)]
struct PointWithTangent {
    point: Vec2,
    tangent: Vec2,
}

pub struct Trail {
    points: VecDeque<PointWithTangent>,
    radius: f32,
    length: usize,
    resolution: usize,
    alpha: f32,
    vertices: VecDeque<TrailVertex>,
}

fn compute_tangent(prev: Option<Vec2>, p: Vec2, next: Option<Vec2>) -> Vec2 {
    match (prev, next) {
        (Some(prev), Some(next)) => (next - prev) / 2.0,
        (None, Some(next)) => next - p,
        (Some(prev), None) => p - prev,
        (None, None) => Vec2::new(0.0, 0.0),
    }
}

fn get_catmull_rom_point(t: f32, p0: Vec2, t0: Vec2, p1: Vec2, t1: Vec2) -> Vec2 {
    let t2 = t  * t;
    let t3 = t2 * t;
    let f1 =  2.0 * t3 - 3.0 * t2 + 1.0;
    let f2 = -2.0 * t3 + 3.0 * t2;
    let f3 =        t3 - 2.0 * t2 + t;
    let f4 =        t3 -       t2;
    (p0 * f1 + p1 * f2 + t0 * f3 + t1 * f4)
}

fn generate_vertices(resolution: usize, color: Vec4, alpha_rate: f32, radius: f32, p0: Vec2, t0: Vec2, p1: Vec2, t1: Vec2) -> Vec<TrailVertex> {
    let mut ps = (0..resolution).map(|i| {
        let t = i as f32 / resolution as f32;
        (t, get_catmull_rom_point(t, p0, t0, p1, t1))
    }).chain(Some((1.0, p1)).into_iter());
    let mut v = vec![];
    let mut a = ps.next().unwrap();
    let mut c = color.as_ref().clone();
    loop {
        match ps.next() {
            Some(b) => {
                let tangent = ::vec_cross_z((b.1 - a.1).normalize()) * radius;
                let t = a.0;
                c[3] = color.w - (1.0 - t) * alpha_rate / resolution as f32;
                v.push(TrailVertex { a_position: (a.1 + tangent).as_ref().clone(), a_color: c });
                v.push(TrailVertex { a_position: (a.1 - tangent).as_ref().clone(), a_color: c });
                a = b;
            },
            None => {
                return v;
            }
        }
    }
}

// TODO: Allow radius to be per-point, also allow radius to change every frame
impl Trail {
    pub fn new(radius: f32, length: usize, resolution: usize, alpha: f32) -> Trail {
        Trail {
            points: VecDeque::with_capacity(length),
            radius: radius,
            length: length,
            resolution: resolution,
            alpha: alpha,
            vertices: VecDeque::with_capacity((length - 1) * resolution),
        }
    }
    fn fade_rate(&self) -> f32 {
        self.alpha / self.length as f32
    }
    pub fn decay(&mut self) {
        if self.points.len() > 0 {
            self.points.pop_front();
            for _ in 0..2*self.resolution { self.vertices.pop_front(); }
        }
    }
    pub fn done(&self) -> bool {
        self.points.len() == 0
    }
    pub fn add_point(&mut self, point: Vec2, color: na::Vec4<f32>) {
        let color = Vec4::new(color.x, color.y, color.z, self.alpha);
        let fade_rate = self.fade_rate();
        // Get rid of oldest points and segments
        if self.points.len() == self.length {
            self.points.pop_front();
            for _ in 0..2*self.resolution { self.vertices.pop_front(); }
        }
        let new_point = PointWithTangent {
            point: point,
            tangent: compute_tangent(self.points.back().map(|x| x.point), point, None),
        };
        // Adjust previous point's tangent, regenerate segments connecting to that previous point, and generate new segments connecting to the new point
        if let Some(mut to_modify) = self.points.pop_back() {
            to_modify.tangent = compute_tangent(self.points.back().map(|x| x.point), to_modify.point, Some(point));
            // Regenerate segments
            if let Some(earlier_point) = self.points.back() {
                for _ in 0..2*self.resolution { self.vertices.pop_back(); }
                self.vertices.extend(generate_vertices(self.resolution, color, fade_rate, self.radius, earlier_point.point, earlier_point.tangent, to_modify.point, to_modify.tangent));
            }
            self.vertices.extend(generate_vertices(self.resolution, color, fade_rate, self.radius, to_modify.point, to_modify.tangent, new_point.point, new_point.tangent));
            self.points.push_back(to_modify);
        }
        self.points.push_back(new_point);
    }
    pub fn fade_all(&mut self) {
        let by_alpha = self.fade_rate();
        for vertex in self.vertices.iter_mut() {
            vertex.a_color[3] -= by_alpha;
        }
    }
}

pub struct TrailRenderer {
	vertex_buffer: glium::VertexBuffer<TrailVertex>,
	// index_buffer: glium::IndexBuffer<u16>,
	program: glium::Program,
    stencil_id: i32,
}

impl TrailRenderer {
    pub fn new(facade: &Rc<Context>) -> TrailRenderer {
        TrailRenderer {
            vertex_buffer: glium::VertexBuffer::empty_dynamic(facade, MAX_VERTICES).unwrap(),
            // index_buffer: glium::IndexBuffer::empty_dynamic(facade, glium::index::PrimitiveType::TrianglesList, INDICES_PER_BATCH).unwrap(),
            program: glium::Program::from_source(facade, TRAIL_VERTEX_SHADER_SRC, TRAIL_FRAGMENT_SHADER_SRC, None).unwrap(),
            stencil_id: 0,
        }
    }
    pub fn start(&mut self, frame: &mut glium::Frame) {
        frame.clear_stencil(0);
        self.stencil_id = 0;
    }
    pub fn draw_trail(&mut self, view: &Mat4, frame: &mut glium::Frame, trail: &Trail) {
        self.stencil_id += 1;
        let uniforms = uniform! { u_view: view.as_ref().clone() };
        let params = glium::DrawParameters {
            blend: premultiplied_blend(),
            depth: glium::Depth {
                test: glium::DepthTest::IfLessOrEqual,
                write: false,
                .. Default::default()
            },
            stencil: glium::draw_parameters::Stencil {
                test_clockwise: glium::StencilTest::IfMore { mask: 0xffffffff },
                test_counter_clockwise: glium::StencilTest::IfMore { mask: 0xffffffff },
                reference_value_clockwise: self.stencil_id,
                reference_value_counter_clockwise: self.stencil_id,
                depth_pass_operation_clockwise: glium::StencilOperation::Replace,
                depth_pass_operation_counter_clockwise: glium::StencilOperation::Replace,
                .. Default::default()
            },
            .. Default::default()
        };

        let (a, b) = trail.vertices.as_slices();

        if a.len() > 0 { self.vertex_buffer.slice(0..a.len()).unwrap().write(a); }
        if b.len() > 0 { self.vertex_buffer.slice(a.len()..a.len() + b.len()).unwrap().write(b); }

		frame.draw(self.vertex_buffer.slice(0..a.len() + b.len()).unwrap(), glium::index::IndicesSource::NoIndices { primitives: glium::index::PrimitiveType::TriangleStrip }, &self.program, &uniforms, &params).unwrap();
    }
}
