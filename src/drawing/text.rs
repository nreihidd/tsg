use glium::{self, Texture2d, VertexBuffer, IndexBuffer, Program, Surface};
use na::{Mat4, Vec4};
use drawing::{quad_indices, load_image};
use glium::backend::Context;
use std::rc::Rc;

#[derive(Copy, Clone, Debug)]
struct CharVertex {
    a_position: [f32; 2],
    a_uv: [f32; 2],
}
implement_vertex!(CharVertex, a_position, a_uv);

// Took so many hours of searching to find out about how multisampling can cause extrapolation in
// the fragment shader, making it so you could and often would sample outside of the polygon along
// its edges. For texture atlases this causes texture bleeding even with nearest filtering and no
// mipmaps. The centroid keyword tells the GPU to take the more expensive route and sample from
// a more sane point, allowing use of multisampling even with tightly packed texture atlases:
// https://www.opengl.org/pipeline/article/vol003_6/
const TEXT_VERTEX_SHADER_SRC: &'static str = r#"
    #version 140
	uniform mat4 u_view;
    in vec2 a_position;
    in vec2 a_uv;
    centroid out vec2 v_uv;
    void main() {
        gl_Position = u_view * vec4(a_position, 0.0, 1.0);
        v_uv = a_uv;
    }
"#;

const TEXT_FRAGMENT_SHADER_SRC: &'static str = r#"
    #version 140
    uniform sampler2D u_sampler;
    uniform vec4 u_color;
    centroid in vec2 v_uv;
    out vec4 color;
    void main() {
        color = texture(u_sampler, v_uv);
        if (color.a < 1.0) {
            discard;
        } else {
            color = u_color;
        }
    }
"#;

pub struct Text {
    font_texture: Texture2d,
    vertices: VertexBuffer<CharVertex>,
    indices: IndexBuffer<u16>,
    program: Program,
    max_quads: u16,
}
impl Text {
    pub fn new(facade: &Rc<Context>) -> Text {
        let max_quads: u16 = 256;
        Text {
            font_texture: glium::texture::Texture2d::with_mipmaps(facade, load_image("assets/font.png"), glium::texture::MipmapsOption::NoMipmap).unwrap(),
            vertices: VertexBuffer::empty_dynamic(facade, max_quads as usize * 4).unwrap(),
            indices: IndexBuffer::new(facade, glium::index::PrimitiveType::TrianglesList, &quad_indices(max_quads)).unwrap(),
            program: Program::from_source(facade, TEXT_VERTEX_SHADER_SRC, TEXT_FRAGMENT_SHADER_SRC, None).unwrap(),
            max_quads: max_quads,
        }
    }
    pub fn draw_string(&self, view: &Mat4<f32>, draw: &mut glium::Frame, color: Vec4<f32>, s: &str) {
        let tcw = 8.0f32 / 128.0f32;
        let tch = 9.0f32 / 64.0f32;
        let tleft = 0.0f32;
        let tright = tcw;
        let ttop = 1.0f32;
        let tbottom = 1.0 - tch;
        let verts_default = [
            CharVertex { a_position: [ -1.0, -1.0], a_uv: [  tleft, tbottom ] },
            CharVertex { a_position: [  1.0, -1.0], a_uv: [  tright, tbottom ] },
            CharVertex { a_position: [  1.0,  1.0], a_uv: [  tright, ttop ] },
            CharVertex { a_position: [ -1.0,  1.0], a_uv: [  tleft, ttop ] },
        ];
        let verts: Vec<CharVertex> =
            s
            .chars()
            .map(|c| match c as u32 { i@0x20...0x7E => i - 0x20, _ => 31 })
            .map(|index| (index / 16, index % 16))
            .enumerate().flat_map(|(index, (row, col))| {
                let trow = row as f32 * tch;
                let tcol = col as f32 * tcw;
                verts_default.iter().map(move |vert| {
                    CharVertex { a_position: [vert.a_position[0] + index as f32 * 2.0, vert.a_position[1]], a_uv: [vert.a_uv[0] + tcol, vert.a_uv[1] - trow] }
                })
            }).collect();
        let uniforms = uniform! {
            u_color: color.as_ref().clone(),
            u_view: view.as_ref().clone(),
            u_sampler: self.font_texture.sampled().minify_filter(glium::uniforms::MinifySamplerFilter::Nearest).magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest),
        };
        let params = glium::DrawParameters {
            blend: glium::Blend::alpha_blending(),
            depth: glium::Depth {
                test: glium::DepthTest::IfLess,
                write: false,
                .. Default::default()
            },
            .. Default::default()
        };
        for vslice in verts.chunks(self.max_quads as usize * 4) {
            let len = vslice.len();
            let bslice = self.vertices.slice(0..len).unwrap();
            bslice.write(vslice);
            let islice = self.indices.slice(0..len / 4 * 6).unwrap();
            draw.draw(bslice, islice, &self.program, &uniforms, &params).unwrap();
        }
    }
}
