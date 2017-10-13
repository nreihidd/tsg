use glium::backend::Context;
use std::rc::Rc;
use glium;
use na;
use glium::{Texture2d, VertexBuffer, IndexBuffer, Program, Surface};
use glium::texture::{UncompressedFloatFormat};
use rand::random;
use na::{Vec2, Vec3, Mat4, Eye};
use std::f32::consts::PI as PI32;
use drawing::{load_image, quad_indices, translate, scale, polygon_to_triangles, compute_program};
use drawing::model::Model;
use drawing::simulation::Simulation;

#[derive(Copy, Clone)]
struct WindVertex {
    position: [f32; 2],
    uv: [f32; 2],
}
implement_vertex!(WindVertex, position, uv);

const WIND_VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
	uniform mat4 view;
    in vec2 position;
    in vec2 uv;
    out vec2 vUv;
    void main() {
        gl_Position = view * vec4(position, 0.0, 1.0);
        vUv = uv;
    }
"#;

const WIND_FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
    uniform sampler2D tex;
    uniform vec2 dir;
    uniform float alpha;
    uniform vec2 uv_offset;
    in vec2 vUv;
    out vec4 value;
    void main() {
        float mag = texture(tex, vUv + uv_offset).x;
        value = vec4(dir, 0.0, alpha * mag);
    }
"#;

fn make_wind_model(facade: &Rc<Context>) -> Model<WindVertex> {
	let vertices = vec![
		WindVertex { position: [ -1.0, -1.0], uv: [  0.0,  0.0 ] },
		WindVertex { position: [  1.0, -1.0], uv: [  1.0,  0.0 ] },
		WindVertex { position: [  1.0,  1.0], uv: [  1.0,  1.0 ] },
		WindVertex { position: [ -1.0,  1.0], uv: [  0.0,  1.0 ] },
	];

	let indices = quad_indices(1);

	Model {
		vertex_buffer: glium::VertexBuffer::new(facade, &vertices).unwrap(),
		index_buffer: glium::index::IndexBuffer::new(facade, glium::index::PrimitiveType::TrianglesList, &indices).unwrap(),
	}
}

#[derive(Copy, Clone)]
struct BendVertex {
    position: [f32; 2],
    mag: f32,
}
implement_vertex!(BendVertex, position, mag);

const BEND_VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
	uniform mat4 view;
    in vec2 position;
    in float mag;
    out float vMag;
    out vec2 vPosition;
    void main() {
        gl_Position = view * vec4(position, 0.0, 1.0);
        vMag = mag;
        vPosition = position;
    }
"#;

const BEND_FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
    uniform vec2 dir;
    uniform float alpha;
    uniform float position_influence;
    in float vMag;
    in vec2 vPosition;
    out vec4 value;
    void main() {
        value = vec4((dir + normalize(vPosition) * position_influence) * vMag, 0.0, alpha);
        // value = vec4(0.5, 0.0, 0.0, alpha);
    }
"#;

fn make_circle_model(facade: &Rc<Context>, n: u16) -> Model<BendVertex> {
	let mut vertices = vec![BendVertex { position: [0.0, 0.0], mag: 1.0 }];
	let mut indices = vec![0u16];
	for i in 0..n+1 {
		let a = i as f32 * PI32 * 2.0 / n as f32;
		vertices.push(BendVertex { position: [a.cos(), a.sin()], mag: 0.0 });
		indices.push(i + 1);
	}

	Model {
		vertex_buffer: glium::VertexBuffer::new(facade, &vertices).unwrap(),
		index_buffer: glium::index::IndexBuffer::new(facade, glium::index::PrimitiveType::TriangleFan, &indices).unwrap(),
	}
}

fn make_rect_model(facade: &Rc<Context>) -> Model<BendVertex> {
	let vertices = vec![
		BendVertex { position: [ -1.0, -1.0 ], mag: 1.0 },
		BendVertex { position: [  1.0, -1.0 ], mag: 1.0 },
		BendVertex { position: [  1.0,  1.0 ], mag: 1.0 },
		BendVertex { position: [ -1.0,  1.0 ], mag: 1.0 },
	];

	let indices = quad_indices(1);

	Model {
		vertex_buffer: glium::VertexBuffer::new(facade, &vertices).unwrap(),
		index_buffer: glium::index::IndexBuffer::new(facade, glium::index::PrimitiveType::TrianglesList, &indices).unwrap(),
	}
}


#[derive(Copy, Clone)]
struct GrassVertex {
    root: [f32; 2],
    index: i32,
    tangent: [f32; 2],
    neutral: [f32; 2],
    color: [f32; 3],
}
implement_vertex!(GrassVertex, root, index, tangent, neutral, color);

const VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
    const float PI = 3.1415926;
    const float SEGMENT_LENGTH = 7.0; // 6.0 // Originally 12.5
    const float SEGMENT_HALF_WIDTH = 2.5; // 5.0;
    const int NUM_SEGMENTS = 4;
	uniform mat4 view;

    uniform mat4 world_to_uv;
    uniform sampler2D bend_map;
    // uniform vec2 dbg_gw;

    in vec2 root;
    in vec2 neutral;
    in vec2 tangent;
    in vec3 color;
    in int index;

    out vec3 vColor;

    float bend_to_angle(vec2 v) {
        return min(1.0, length(v)) * PI / 2.0;
    }

    mat4 rotation_z(float a) {
        float c = cos(a);
        float s = sin(a);
        return mat4(mat2(c, s, -s, c));
    }
    mat4 rotation_y(float a) {
        float c = cos(a);
        float s = sin(a);
        mat4 m = mat4(1.0);
        m[0][0] = c;
        m[2][0] = s;
        m[0][2] = -s;
        m[2][2] = c;
        return m;
    }
    mat4 translation(float x, float y, float z) {
        mat4 m = mat4(1.0);
        m[3][0] = x;
        m[3][1] = y;
        m[3][2] = z;
        return m;
    }

    void main() {
        int segment = index / 2;

        vec2 bend_map_uv = (world_to_uv * vec4(root, 0.0, 1.0)).xy;
        vec2 bend_from_map = clamp(texture(bend_map, bend_map_uv).xy, -1.0, 1.0);
        vec2 bend = neutral + bend_from_map;

        float bend_angle = bend_to_angle(bend) / (NUM_SEGMENTS - 1);
        vec2 dir = normalize(bend);
        // mat4 t = rotation_z(atan(dir.y, dir.x));
        mat4 t = mat4(1.0);
        mat4 t_step = rotation_y(bend_angle) * translation(0.0, 0.0, SEGMENT_LENGTH);
        for (int i = 0; i < segment; i++) {
            t = t * t_step;
        }
        vec4 pos = vec4(root, 0.0, 0.0) + rotation_z(atan(dir.y, dir.x)) * t * vec4(0.0, 0.0, 0.0, 1.0);
        vec4 offset = (t * vec4(tangent, 0.0, 0.0)) * SEGMENT_HALF_WIDTH;
        if (segment < NUM_SEGMENTS) {
            pos = pos + offset * ((index & 1) * 2 - 1);
        }

        // For basic ortho, squash max height into [0.1,1] (0.1 is closer)
        // pos.z = pos.z / (SEGMENT_LENGTH * NUM_SEGMENTS);

        gl_Position = view * pos;
		vColor = color;
        // vColor = vec3(clamp(bend_from_map.xy, 0.0, 1.0), 1.0);
    }
"#;

const FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
    in vec3 vColor;
    out vec4 color;
    void main() {
        color = vec4(vColor, 1.0);
    }
"#;

const BASE_COLOR: Vec3<f32> = Vec3 { x: 0.0, y: 0.3, z: 0.0 };
fn make_grass_blade(x: f32, y: f32, vertices: &mut Vec<GrassVertex>, indices: &mut Vec<u32>) {
    const NUM_SEGMENTS: i32 = 4;
    let facing = random::<f32>() * 2.0 * PI32;
    let tangent = [-facing.sin(), facing.cos()];
    let neutral_mag = random::<f32>() * 0.5;
    let neutral_angle = facing;
    let neutral = [neutral_angle.cos() * neutral_mag, neutral_angle.sin() * neutral_mag];
    let color = Vec3::new(random::<f32>() * 0.05, 0.4 + random::<f32>() * 0.4, random::<f32>() * 0.1);
    let vi = vertices.len() as u32;
    vertices.extend((0..NUM_SEGMENTS * 2 + 1).map(|n| GrassVertex {
        root: [x, y],
        index: n,
        tangent: tangent,
        neutral: neutral,
        color: {
            // Make bottoms darker
            let segment = n / 2;
            let h = segment as f32 / NUM_SEGMENTS as f32;
            (BASE_COLOR * (1.0 - h) + color * h).as_ref().clone()
        },
    }));
    let quad = [0, 1, 2, 1, 3, 2];
    indices.extend((0..NUM_SEGMENTS as u32 - 1).flat_map(|n| quad.iter().map(move |i| vi + n * 2 + i)));
    {
        let l = vertices.len() as u32;
        indices.extend([l - 3, l - 2, l - 1].iter());
    }
}

#[derive(Copy, Clone)]
struct FloorVertex {
    a_position: [f32; 2],
}
implement_vertex!(FloorVertex, a_position);
const FLOOR_VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
	uniform mat4 u_view;
    in vec2 a_position;
    void main() {
        gl_Position = u_view * vec4(a_position, 0.0, 1.0);
    }
"#;

const FLOOR_FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
    out vec4 color;
    void main() {
        color = vec4(0.0, 0.3, 0.0, 1.0); //texture(u_texture, v_uv);
    }
"#;
/*
fn square_verts(x: f32, y: f32, hw: f32, hh: f32) -> [FloorVertex; 6] {
	let f = |x: f32, y: f32, u: f32, v: f32| FloorVertex { a_position: [x, y], a_uv: [u, v] };
	let v1 = f(x - hw, y - hh, 0.0, 0.0);
	let v2 = f(x + hw, y - hh, 1.0, 0.0);
	let v3 = f(x + hw, y + hh, 1.0, 1.0);
	let v4 = f(x - hw, y + hh, 0.0, 1.0);
    [v1, v2, v3, v1, v3, v4]
}
*/

pub struct GrassBuilder {
    vertices: Vec<GrassVertex>,
    indices: Vec<u32>,
    floor: Vec<FloorVertex>,
}
impl GrassBuilder {
    pub fn new() -> GrassBuilder {
        GrassBuilder {
            vertices: vec![],
            indices: vec![],
            floor: vec![],
        }
    }
    pub fn add_blade(&mut self, x: f32, y: f32) {
        make_grass_blade(x, y, &mut self.vertices, &mut self.indices);
        // self.floor.extend(square_verts(x, y, 10.0, 10.0).into_iter());
    }
    pub fn add_floor_polygon(&mut self, polygon: &[Vec2<f32>]) {
		let indices: Vec<_> = polygon_to_triangles(&polygon).iter().flat_map(|is| is.iter().map(|&i| i)).collect();
        for index in indices {
            self.floor.push(FloorVertex { a_position: polygon[index].as_ref().clone() });
        }
    }
    pub fn build(self, facade: &Rc<Context>) -> Grass {
        Grass::new(facade, self.vertices, self.indices, self.floor)
    }
}

pub struct Grass {
    simulation: Simulation,
    tiled_noise: Texture2d,
    noise_pos: Vec2<f32>,
    noise_vel: Vec2<f32>,
    noise_t: i32,
    floor_vertices: VertexBuffer<FloorVertex>,
    vertices: VertexBuffer<GrassVertex>,
    indices: IndexBuffer<u32>,
    grass_program: Program,
    bend_program: Program,
    wind_program: Program,
    floor_program: Program,
    simulate_program: Program,
    circle_model: Model<BendVertex>,
    rect_model: Model<BendVertex>,
    wind_model: Model<WindVertex>,
}
impl Grass {
    fn new(facade: &Rc<Context>, vertices: Vec<GrassVertex>, indices: Vec<u32>, floor_vertices: Vec<FloorVertex>) -> Grass {
        Grass {
            tiled_noise: glium::texture::Texture2d::new(facade, load_image("assets/noise.png")).unwrap(),
            simulation: Simulation::new(facade, UncompressedFloatFormat::F32F32, Vec2::new(1920.0, 1080.0), Vec2::new(256, 256)),
            floor_vertices: VertexBuffer::new(facade, &floor_vertices).unwrap(),
            vertices: VertexBuffer::new(facade, &vertices).unwrap(),
            indices: IndexBuffer::new(facade, glium::index::PrimitiveType::TrianglesList, &indices).unwrap(),
            grass_program: Program::from_source(facade, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, None).unwrap(),
            bend_program: compute_program(facade, BEND_VERTEX_SHADER_SRC, BEND_FRAGMENT_SHADER_SRC).unwrap(),
            wind_program: compute_program(facade, WIND_VERTEX_SHADER_SRC, WIND_FRAGMENT_SHADER_SRC).unwrap(),
            simulate_program: compute_program(facade, SIMULATE_VERTEX_SHADER_SRC, SIMULATE_FRAGMENT_SHADER_SRC).unwrap(),
            floor_program: Program::from_source(facade, FLOOR_VERTEX_SHADER_SRC, FLOOR_FRAGMENT_SHADER_SRC, None).unwrap(),
            circle_model: make_circle_model(facade, 32),
            rect_model: make_rect_model(facade),
            wind_model: make_wind_model(facade),
            noise_pos: Vec2::new(0.0, 0.0),
            noise_vel: Vec2::new(0.0001, 0.0002),
            noise_t: 0,
            // dbg_gw: Vec2::new(0.0, 1.0f32),
        }
    }
    pub fn simulate(&mut self, scroll_to: Vec2<f32>) {
        self.simulation.simulate(scroll_to, &self.simulate_program);
    }
    pub fn swap(&mut self) {
        self.simulation.swap();
    }
    pub fn wind(&mut self) {
        let mut draw = self.simulation.draw();
		let view = na::Mat4::<f32>::new_identity(4);
        self.noise_t += 1;
        self.noise_vel = ::vec_from_angle(self.noise_t as f32 * PI32 * 2.0 / 6000.0) * ((self.noise_t as f32 * PI32 * 2.0 / 300.0).sin() + 1.0) / 2.0;
        self.noise_pos = self.noise_pos + Vec2::new(0.0001, 0.0002);
        let uniforms = uniform! {
            view: view.as_ref().clone(),
            dir: self.noise_vel.as_ref().clone(),
            alpha: 0.01f32,
            tex: self.tiled_noise.sampled().wrap_function(glium::uniforms::SamplerWrapFunction::Repeat),
            uv_offset: self.noise_pos.as_ref().clone(),
        };
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
        draw.draw(&self.wind_model.vertex_buffer, &self.wind_model.index_buffer, &self.wind_program, &uniforms, &params).unwrap();
    }
    pub fn straighten(&mut self) {
        let mut draw = self.simulation.draw();
		let view = na::Mat4::<f32>::new_identity(4);
        let uniforms = uniform! {
            view: view.as_ref().clone(),
            dir: [0.0f32, 0.0],
            alpha: 0.008f32,
            position_influence: 0.0f32,
        };
        let params = glium::DrawParameters {
            blend: glium::Blend {
                color: glium::BlendingFunction::Addition {
                    source: glium::LinearBlendingFactor::SourceAlpha,
                    destination: glium::LinearBlendingFactor::OneMinusSourceAlpha,
                },
                .. Default::default()
            },
            .. Default::default()
        };
        draw.draw(&self.rect_model.vertex_buffer, &self.rect_model.index_buffer, &self.bend_program, &uniforms, &params).unwrap();
    }
    pub fn crush_area(&mut self, pos: Vec2<f32>, dir: Vec2<f32>, pi: f32, radius: f32) {
        let mut draw = self.simulation.draw();
        let view = self.simulation.world_to_simgl();
		let view = view * translate(pos.x, pos.y, 0.0) * scale(radius, radius, 1.0);
        let uniforms = uniform! {
            view: view.as_ref().clone(),
            dir: dir.as_ref().clone(),
            alpha: 0.2f32,
            position_influence: pi,
        };
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
        draw.draw(&self.circle_model.vertex_buffer, &self.circle_model.index_buffer, &self.bend_program, &uniforms, &params).unwrap();
    }
    pub fn draw_floor(&mut self, frame: &mut glium::Frame, view: &Mat4<f32>) {
        let uniforms = uniform! {
            u_view: view.as_ref().clone(),
        };
		let params = glium::DrawParameters {
			blend: super::premultiplied_blend(),
			depth: glium::Depth {
				test: glium::DepthTest::IfLessOrEqual,
				write: false,
				.. Default::default()
			},
			.. Default::default()
		};
        frame.draw(&self.floor_vertices, glium::index::IndicesSource::NoIndices { primitives: glium::index::PrimitiveType::TrianglesList }, &self.floor_program, &uniforms, &params).unwrap();
    }
    pub fn draw(&mut self, view: &Mat4<f32>, draw: &mut glium::Frame) {
        let params = glium::DrawParameters {
            // polygon_mode: glium::draw_parameters::PolygonMode::Line,
            depth: glium::Depth {
                test: glium::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };
        // self.dbg_gw = self.dbg_gw * 0.98;
        let uniforms = uniform! {
            view: view.as_ref().clone(),
            bend_map: self.simulation.sampled(),
            world_to_uv: self.simulation.world_to_simuv().as_ref().clone(),
            // dbg_gw: self.dbg_gw.as_array().clone()
        };
        draw.draw(&self.vertices, &self.indices, &self.grass_program, &uniforms, &params).unwrap();
    }
}

const SIMULATE_VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
    in vec2 a_position;
    void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
"#;

const SIMULATE_FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
	uniform sampler2D u_height_map;
    uniform vec2 u_resolution;
    uniform ivec2 u_scroll;
    out vec4 color;
    void main() {
        vec2 e = vec2(1.0) / u_resolution;
        vec2 uv = (gl_FragCoord.xy + u_scroll) * e;
        vec2 c = texture(u_height_map, uv).xy;

        color = vec4(c, 0.0, 1.0);
    }
"#;