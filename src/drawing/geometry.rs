use std::convert::From;
use glium::{self, VertexBuffer, IndexBuffer};
use glium::backend::Context;
use std::rc::Rc;
use na::Vec2;

#[derive(Copy, Clone, Debug)]
pub struct SimpleVertex {
    a_position: [f32; 2],
}
implement_vertex!(SimpleVertex, a_position);

impl From<Vec2<f32>> for SimpleVertex {
    fn from(v: Vec2<f32>) -> SimpleVertex {
        SimpleVertex { a_position: [v.x, v.y] }
    }
}

pub struct Geometry<I: glium::index::Index, V: glium::vertex::Vertex> {
    pub vertices: VertexBuffer<V>,
    pub indices: IndexBuffer<I>,
}

pub struct GeometryBuilder<V> {
    vertices: Vec<V>,
    indices: Vec<u32>,
}
impl<V: From<Vec2<f32>> + glium::vertex::Vertex> GeometryBuilder<V> {
    pub fn new() -> GeometryBuilder<V> {
        GeometryBuilder {
            vertices: vec![],
            indices: vec![],
        }
    }
    pub fn add_triangles(&mut self, vertices: &[Vec2<f32>], indices: &[u32]) {
        let index_offset = self.vertices.len() as u32;
        self.vertices.extend(vertices.iter().map(|&v| {
            let v: V = From::from(v);
            v
        }));
        self.indices.extend(indices.iter().map(|&i| i + index_offset));
    }
    pub fn build(self, facade: &Rc<Context>) -> Geometry<u32, V> {
        Geometry {
            vertices: VertexBuffer::new(facade, &self.vertices).unwrap(),
            indices: IndexBuffer::new(facade, glium::index::PrimitiveType::TrianglesList, &self.indices).unwrap(),
        }
    }
}