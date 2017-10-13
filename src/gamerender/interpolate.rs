use na::{Vec2};
use ::{min_angle_diff, normalized_angle};

pub struct Interpolation<'a, T: 'a> {
    pub a: &'a T,
    pub b: &'a T,
    pub t: f32,
}
impl<'a, T> Interpolation<'a, T> {
    pub fn new(a: &'a T, b: &'a T, t: f32) -> Interpolation<'a, T> {
        Interpolation {
            a: a,
            b: b,
            t: t,
        }
    }
    pub fn field<F, U: Interpolate>(&self, f: F) -> U
        where F: Fn(&T) -> U
    {
        let a = f(&self.a);
        let b = f(&self.b);
        a.interpolate(&b, self.t)
    }
}

pub trait Interpolate<U=Self> {
    fn interpolate(&self, other: &Self, t: f32) -> U;
}

impl Interpolate for f32 {
    fn interpolate(&self, other: &f32, t: f32) -> f32 {
        let a = *self;
        let b = *other;
        a + (b - a) * t
    }
}
impl Interpolate for Vec2<f32> {
    fn interpolate(&self, other: &Vec2<f32>, t: f32) -> Vec2<f32> {
        Vec2::new(self.x.interpolate(&other.x, t), self.y.interpolate(&other.y, t))
    }
}

pub struct Angle(pub f32);
impl Interpolate for Angle {
    fn interpolate(&self, other: &Angle, t: f32) -> Angle {
        Angle(normalized_angle(self.0 + min_angle_diff(other.0, self.0) * t))
    }
}