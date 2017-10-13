use glium::backend::Context;
use std::rc::Rc;
use glium::{self, Program, Surface};
use glium::texture::{UncompressedFloatFormat};
use na::{Vec2, Mat4};
use super::{premultiplied_blend, load_image, compute_program};
use drawing::simulation::{Simulation, Disturber};
use drawing::geometry::{Geometry, SimpleVertex};

const SAND_AREA: f32 = 2048.0;
const SAND_SIMULATION_SIZE: u32 = 512;

pub struct Sand {
    simulation: Simulation,
    sand_terrain: glium::texture::SrgbTexture2d,
    sand_program: Program,
    simulation_program: Program,
    geometry: Geometry<u32, SimpleVertex>,
}

impl Sand {
    pub fn new(facade: &Rc<Context>, geometry: Geometry<u32, SimpleVertex>) -> Sand {
        Sand {
            simulation: Simulation::new(facade, UncompressedFloatFormat::F32F32, Vec2::new(SAND_AREA, SAND_AREA), Vec2::new(SAND_SIMULATION_SIZE, SAND_SIMULATION_SIZE)),
            sand_terrain: glium::texture::SrgbTexture2d::new(facade, load_image("assets/sand-terrain.png")).unwrap(),
            sand_program: Program::from_source(facade, RENDER_VERTEX_SHADER_SRC, RENDER_FRAGMENT_SHADER_SRC, None).unwrap(),
            simulation_program: compute_program(facade, SIMULATION_VERTEX_SHADER_SRC, SIMULATION_FRAGMENT_SHADER_SRC).unwrap(),
            geometry: geometry,
        }
    }

    pub fn simulate(&mut self, scroll_to: Vec2<f32>) {
        self.simulation.simulate(scroll_to, &self.simulation_program);
    }

    pub fn swap(&mut self) {
        self.simulation.swap();
    }

    pub fn disturb(&self, disturber: &mut Disturber, p: Vec2<f32>, radius: f32, amount: f32) {
        disturber.disturb(&self.simulation, p, radius, amount);
    }

    pub fn draw(&self, frame: &mut glium::Frame, view: &Mat4<f32>) {
        let uniforms = uniform! {
            u_view: view.as_ref().clone(),
            u_height_map: self.simulation.sampled(),
            u_terrain: self.sand_terrain.sampled().wrap_function(glium::uniforms::SamplerWrapFunction::Repeat),
            u_world_to_uv: self.simulation.world_to_simuv().as_ref().clone(),
            u_resolution: self.simulation.get_simulation_size().as_ref().clone(),
            u_world_size: self.simulation.get_world_size().as_ref().clone(),
        };
        let params = glium::DrawParameters {
            blend: premultiplied_blend(),
            .. Default::default()
        };
        frame.draw(&self.geometry.vertices, &self.geometry.indices, &self.sand_program, &uniforms, &params).unwrap();
    }
}

const SIMULATION_VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
    in vec2 a_position;
    void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
"#;

const SIMULATION_FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
	uniform sampler2D u_height_map;
    uniform vec2 u_resolution;
    uniform vec2 u_world_size;
    uniform ivec2 u_scroll;
    out vec4 color;

    // Max height difference between two adjacent pixels: sin(angle_of_repose)
    #define MAX_SLOPE 0.56

    float adjustment(float self, float other, float slope) {
        // Divide by 8.0 because both should take half, but also there are 4 adjustments total?
        // Okay I don't know why dividing by 8 works but dividing by 2 blows up.
        // Dividing by 2 will lead to NaNs, but why?
        if (self + slope < other) {
            return (other - self - slope) / 8.0;
        } else if (self > other + slope) {
         	return -(self - other - slope) / 8.0;
        } else {
         	return 0.0;
        }
    }

    void main() {
        vec2 e = vec2(1.0) / u_resolution;
        vec2 uv = (gl_FragCoord.xy + u_scroll) * e;

        vec2 c = texture(u_height_map, uv).xy;
        float hc = c.x;
        float hcp = c.y;
        float hl = texture(u_height_map, vec2(uv.x - e.x, uv.y)).x;
        float hr = texture(u_height_map, vec2(uv.x + e.x, uv.y)).x;
        float hb = texture(u_height_map, vec2(uv.x, uv.y - e.y)).x;
        float ht = texture(u_height_map, vec2(uv.x, uv.y + e.y)).x;

        vec2 scale = u_world_size / u_resolution;

        float r = hc;
        r += adjustment(hc, hl, MAX_SLOPE * scale.x);
        r += adjustment(hc, hr, MAX_SLOPE * scale.x);
        r += adjustment(hc, hb, MAX_SLOPE * scale.y);
        r += adjustment(hc, ht, MAX_SLOPE * scale.y);
        r *= 0.9995;

        color = vec4(r, hc, 0.0, 1.0);
    }
"#;

const RENDER_VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
    uniform mat4 u_view;
    uniform mat4 u_world_to_uv;
    in vec2 a_position;
    out vec2 v_uv;
    out vec2 v_position;
    void main() {
        gl_Position = u_view * vec4(a_position, 0.0, 1.0);
        v_position = a_position;
        v_uv = (u_world_to_uv * vec4(a_position, 0.0, 1.0)).xy;
    }
"#;

const RENDER_FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
	uniform sampler2D u_height_map;
    uniform sampler2D u_terrain;
    uniform vec2 u_resolution;
    uniform vec2 u_world_size;
    in vec2 v_position;
    in vec2 v_uv;
    out vec4 color;

    float sample(vec2 uv) {
        if (uv.x > 1.0 || uv.y > 1.0 || uv.x < 0.0 || uv.y < 0.0) {
            return 0.0;
        }
        return texture(u_height_map, uv).x;
    }

    void main() {
        vec2 e = vec2(1.0) / u_resolution;
        vec2 uv = v_uv; // * e;

        // float hc = sample(uv);
        float hl = sample(vec2(uv.x - e.x, uv.y));
        float hr = sample(vec2(uv.x + e.x, uv.y));
        float hb = sample(vec2(uv.x, uv.y - e.y));
        float ht = sample(vec2(uv.x, uv.y + e.y));
        vec2 scale = u_world_size / u_resolution;
        vec2 gradient = vec2((hr - hl) / (2.0 * scale.x), (ht - hb) / (2.0 * scale.y));
        vec3 norm = normalize(cross(vec3(1.0, 0.0, gradient.x), vec3(0.0, 1.0, gradient.y)));

        vec3 terrain_color = texture(u_terrain, v_position / 256.0).rgb;
        vec3 light = normalize(vec3(-0.2, 0.5, 0.7));
        float diffuse = dot(norm, light);
        vec3 ambientLight = vec3(0.9, 0.95, 1.0) * 0.05; // these values are different than the shadertoy because glium by default uses GL_FRAMEBUFFER_SRGB
        vec3 base = mix(vec3(0.86, 0.54, 0.14), vec3(0.95, 0.6, 0.25), terrain_color.rgb);
        color = vec4(base * diffuse + base * ambientLight, 1.0);
        // color = vec4(gradient, 1.0);
    }
"#;