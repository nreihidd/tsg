use glium::backend::Context;
use std::rc::Rc;
use glium::{self, Texture2d, VertexBuffer, Program, Surface};
use glium::texture::{UncompressedFloatFormat, MipmapsOption};
use glium::framebuffer::SimpleFrameBuffer;
use glium::uniforms::Sampler;
use na::{self, Vec2, Mat4, Inv};
use super::{translate, scale, compute_program};
use std;

#[derive(Copy, Clone, Debug)]
struct SimulationVertex {
    a_position: [f32; 2],
}
implement_vertex!(SimulationVertex, a_position);

fn make_basic_quad(facade: &Rc<Context>) -> VertexBuffer<SimulationVertex> {
    let f = |x, y| SimulationVertex { a_position: [x, y] };
    let vertices = vec![
        f(-1.0, -1.0), f(-1.0, 1.0),
        f( 1.0, -1.0), f( 1.0, 1.0),
    ];
    VertexBuffer::new(facade, &vertices).unwrap()
}

pub struct Simulation {
    world_size: Vec2<f32>,
    simulation_size: Vec2<u32>,
    sim_map_write: Texture2d,
    sim_map_read: Texture2d,
    basic_quad: VertexBuffer<SimulationVertex>,
    // `focus` is in simulation space because all scrolling should be in pixel amounts on the texture,
    // otherwise the height field will rapidly blur when sampled between values.  Simply changing the
    // sampler when simulating to use NEAREST for both minify and magnify does not work.
    focus: Vec2<i32>,
}

impl Simulation {
    pub fn new(facade: &Rc<Context>, sim_format: UncompressedFloatFormat, world_size: Vec2<f32>, simulation_size: Vec2<u32>) -> Simulation {
        let make_texture = || {
            let tex = Texture2d::empty_with_format(facade, sim_format, MipmapsOption::NoMipmap, simulation_size.x, simulation_size.y).unwrap();
            tex.as_surface().clear_color(0.0, 0.0, 0.0, 0.0);
            tex
        };
        Simulation {
            sim_map_write: make_texture(),
            sim_map_read: make_texture(),
            world_size: world_size,
            simulation_size: simulation_size,
            focus: Vec2::new(0, 0),
            basic_quad: make_basic_quad(facade),
        }
    }
}

fn map_vec2<T, U, F>(v: Vec2<T>, f: F) -> Vec2<U>
    where F: Fn(T) -> U
{
    Vec2::new(f(v.x), f(v.y))
}

impl Simulation {
    fn world_to_sim(&self, v: Vec2<f32>) -> Vec2<i32> {
        let si = map_vec2(self.simulation_size, |x| x as f32);
        let v = v * si / self.world_size;
        map_vec2(v, |x| x as i32)
    }

    fn sim_to_world(&self, v: Vec2<i32>) -> Vec2<f32> {
        map_vec2(v, |x| x as f32) / map_vec2(self.simulation_size, |x| x as f32) * self.world_size
    }

    pub fn draw(&self) -> SimpleFrameBuffer {
        self.sim_map_write.as_surface()
    }
    pub fn sampled(&self) -> Sampler<Texture2d> {
        self.sim_map_read.sampled()
    }

    pub fn world_to_simuv(&self) -> Mat4<f32> {
        scale(0.5, 0.5, 1.0) * translate(1.0, 1.0, 0.0) * self.world_to_simgl()
    }
    pub fn world_to_simgl(&self) -> Mat4<f32> {
        let focus = self.sim_to_world(self.focus);
        na::OrthoMat3::<f32>::new(self.world_size.x, self.world_size.y, -1.0, 1.0).to_mat() * translate(-focus.x, -focus.y, 0.0)
    }

    pub fn simulate(&mut self, scroll_to: Vec2<f32>, program: &Program) {
        let scroll_to_on_texture = self.world_to_sim(scroll_to);
        let scroll_amount = scroll_to_on_texture - self.focus;
        let mut draw = self.sim_map_write.as_surface();
        let uniforms = uniform! {
            u_height_map: self.sim_map_read.sampled(),
            u_resolution: map_vec2(self.simulation_size, |x| x as f32).as_ref().clone(),
            u_world_size: self.world_size.as_ref().clone(),
            u_scroll: scroll_amount.as_ref().clone(),
        };
        let params = glium::DrawParameters {
            .. Default::default()
        };
        draw.draw(&self.basic_quad, glium::index::IndicesSource::NoIndices { primitives: glium::index::PrimitiveType::TriangleStrip }, program, &uniforms, &params).unwrap();
        self.focus = scroll_to_on_texture;
    }

    pub fn swap(&mut self) {
        std::mem::swap(&mut self.sim_map_write, &mut self.sim_map_read);
    }

    pub fn get_world_size(&self) -> Vec2<f32> {
        self.world_size
    }
    pub fn get_simulation_size(&self) -> Vec2<f32> {
        map_vec2(self.simulation_size, |x| x as f32)
    }
    pub fn get_focus(&self) -> Vec2<f32> {
        self.sim_to_world(self.focus)
    }
}

pub struct Disturber {
    basic_quad: VertexBuffer<SimulationVertex>,
    disturb_program: Program,
}
impl Disturber {
    pub fn new(facade: &Rc<Context>) -> Disturber {
        Disturber {
            basic_quad: make_basic_quad(facade),
            disturb_program: compute_program(facade, DISTURB_VERTEX_SHADER_SRC, DISTURB_FRAGMENT_SHADER_SRC).unwrap(),
        }
    }
    pub fn disturb(&mut self, simulation: &Simulation, p: Vec2<f32>, radius: f32, amount: f32) {
        let mut draw = simulation.draw();
        let params = glium::DrawParameters {
            blend: glium::Blend {
                color: glium::BlendingFunction::Addition {
                    source: glium::LinearBlendingFactor::SourceAlpha,
                    destination: glium::LinearBlendingFactor::One,
                },
                .. Default::default()
            },
            .. Default::default()
        };
        let view = simulation.world_to_simgl().inv().unwrap();
        let uniforms = uniform! {
            u_center: p.as_ref().clone(),
            u_radius: radius,
            u_amount: amount,
            u_inv_view: view.as_ref().clone(),
        };
        draw.draw(&self.basic_quad, glium::index::IndicesSource::NoIndices { primitives: glium::index::PrimitiveType::TriangleStrip }, &self.disturb_program, &uniforms, &params).unwrap();
    }
}

const DISTURB_VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
    uniform mat4 u_inv_view;
    in vec2 a_position;
    out vec2 v_position;
    void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_position = (u_inv_view * vec4(a_position, 0.0, 1.0)).xy;
    }
"#;
const DISTURB_FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
    uniform vec2 u_center;
    uniform float u_radius;
    uniform float u_amount;
    in vec2 v_position;
    out vec4 value;
    void main() {
        value = vec4(u_amount * smoothstep(u_radius, 0.0, length(v_position - u_center)), 0.0, 0.0, 1.0);
    }
"#;