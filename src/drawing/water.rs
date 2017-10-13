use glium::backend::Context;
use std::rc::Rc;
use glium::{self, Program, Surface};
use glium::texture::{UncompressedFloatFormat};
use na::{Vec2, Mat4};
use super::{premultiplied_blend, load_image_srgb, compute_program};
use drawing::simulation::{Simulation, Disturber};
use drawing::geometry::{Geometry, SimpleVertex};

#[derive(Copy, Clone, Debug)]
struct WaterQuadVertex {
    a_position: [f32; 2],
}
implement_vertex!(WaterQuadVertex, a_position);

const WATER_AREA: f32 = 2048.0;
const WATER_SIMULATION_SIZE: u32 = 512;

pub struct Water {
    simulation: Simulation,
    water_terrain: glium::texture::SrgbTexture2d,
    water_program: Program,
    simulation_program: Program,
    geometry: Geometry<u32, SimpleVertex>,
}

impl Water {
    pub fn new(facade: &Rc<Context>, geometry: Geometry<u32, SimpleVertex>) -> Water {
        Water {
            simulation: Simulation::new(facade, UncompressedFloatFormat::F32F32, Vec2::new(WATER_AREA, WATER_AREA), Vec2::new(WATER_SIMULATION_SIZE, WATER_SIMULATION_SIZE)),
            water_terrain: glium::texture::SrgbTexture2d::new(facade, load_image_srgb("assets/water-terrain.png")).unwrap(),
            water_program: Program::from_source(facade, RENDER_VERTEX_SHADER_SRC, RENDER_FRAGMENT_SHADER_SRC, None).unwrap(),
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
            u_terrain: self.water_terrain.sampled().wrap_function(glium::uniforms::SamplerWrapFunction::Repeat),
            u_world_to_uv: self.simulation.world_to_simuv().as_ref().clone(),
            u_resolution: self.simulation.get_simulation_size().as_ref().clone(),
            u_world_size: self.simulation.get_world_size().as_ref().clone(),
        };
        let params = glium::DrawParameters {
            blend: premultiplied_blend(),
            .. Default::default()
        };
        frame.draw(&self.geometry.vertices, &self.geometry.indices, &self.water_program, &uniforms, &params).unwrap();
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
    uniform ivec2 u_scroll;
    out vec4 color;
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

        // TODO: Actually use the constants (h is scale.x or scale.y), c is wave speed, t is delta time,
        //       The equation changes too, see http://www.roxlu.com/downloads/scholar/001.fluid.height_field_simulation.pdf
        //       Would need to figure out where the h^2 comes from and how to use both scale.x and scale.y
        float r = 0.0;
        if (uv.x <= e.x || uv.x >= 1.0 - e.x || uv.y <= e.y || uv.y >= 1.0 - e.y) {
            float h = 1.0;
            float ct = 1.0;
            float inside = 0.0;
            if (uv.x <= e.x) {
                inside = hr;
            } else if (uv.x >= 1.0 - e.x) {
                inside = hl;
            } else if (uv.y <= e.y) {
                inside = ht;
            } else {
                inside = hb;
            }
            r = (hc*ct + inside*h) / (h + ct);
        } else {
            // r = -hcp + (hl + hr + hb + ht) / 2.0;
            r = hc - hcp + (hl + hr + hb + ht) / 4.0;
        }
        r *= 0.999;

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

    const float RATIO_OF_REFRACTIVE_INDICES = 1.001 / 1.333;
    const float FLOOR_Z = -50.0;

    float sample(vec2 uv) {
        if (uv.x > 1.0 || uv.y > 1.0 || uv.x < 0.0 || uv.y < 0.0) {
            return 0.0;
        }
        return texture(u_height_map, uv).x;
    }

    void main() {
        vec2 e = vec2(1.0) / u_resolution;
        vec2 uv = v_uv; // * e;

        float hc = sample(uv);
        float hl = sample(vec2(uv.x - e.x, uv.y));
        float hr = sample(vec2(uv.x + e.x, uv.y));
        float hb = sample(vec2(uv.x, uv.y - e.y));
        float ht = sample(vec2(uv.x, uv.y + e.y));
        vec2 scale = u_world_size / u_resolution;
        vec2 gradient = vec2((hr - hl) / (2.0 * scale.x), (ht - hb) / (2.0 * scale.y));
        vec3 norm = normalize(cross(vec3(1.0, 0.0, gradient.x), vec3(0.0, 1.0, gradient.y))); // TODO: Isn't this the same as normalize(vec3(-gradient.x, -gradient.y, 1.0)) ??

        // No way for the refracted_vec.z to be >= 0.0, right?
        vec3 refracted_vec = refract(vec3(0.0, 0.0, -1.0), norm, RATIO_OF_REFRACTIVE_INDICES);
        float d = (hc - FLOOR_Z) / -refracted_vec.z;
        vec2 refracted_floor_pos = v_position + refracted_vec.xy * d;
        vec3 terrain_color = texture(u_terrain, refracted_floor_pos / 256.0).rgb; // TODO: Maybe don't hardcode the size of the floor texture?

        vec3 light = normalize(vec3(-0.2, 0.5, 0.7));
        float diffuse = 1.0; // dot(light, norm);

        // TODO: Come back to this and find a better specular shading method?

        // Phong specular
        // Since view_dir is (0, 0, 1) for the ortho top-down perspective, dot(a, view_dir) == a.z
        // float spec = 0.9 * max(0.0, pow(reflect(-light,norm).z, 64.0));

        // Sun reflection (where 0.99 is the cosine of the angular radius of the sun (which here would be a huge 8.1 degrees))
        float spec = 0.9 * smoothstep(0.99, 1.0, dot(light, reflect(vec3(0.0, 0.0, -1.0), norm)));

        // Blinn-phong specular
        // vec3 half_dir = normalize(light + vec3(0.0, 0.0, 1.0));
        // float spec = 0.5 * pow(max(dot(half_dir, norm), 0.0), 128.0);

        vec3 c = mix(terrain_color, vec3(0.7, 0.8, 1.0), 0.1) * max(diffuse, 0.0) + spec;
        color = vec4(c, 1.0);
    }
"#;