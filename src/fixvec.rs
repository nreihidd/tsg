use fixmath::Fixed;
use std::cmp::{PartialOrd, Ord, Ordering};
use na::{Vec2, BaseNum, Absolute, Axpy};

pub trait Norm<N> {
    fn normalize(&self) -> Self;
    fn norm(&self) -> N;
}

fn normsq(v: &Vec2<Fixed>) -> i64 {
    let x = v.x.raw() as i64;
    let y = v.y.raw() as i64;
    x * x + y * y
}

pub struct CmpNorm(pub Vec2<Fixed>);
impl PartialEq<CmpNorm> for CmpNorm {
    fn eq(&self, n: &CmpNorm) -> bool {
        self.partial_cmp(n) == Some(Ordering::Equal)
    }
}
impl PartialOrd<CmpNorm> for CmpNorm {
    fn partial_cmp(&self, n: &CmpNorm) -> Option<Ordering> {
        let m1sq = normsq(&self.0);
        let m2sq = normsq(&n.0);
        Some(m1sq.cmp(&m2sq))
    }
}

impl PartialEq<Fixed> for CmpNorm {
    fn eq(&self, n: &Fixed) -> bool {
        self.partial_cmp(n) == Some(Ordering::Equal)
    }
}
impl PartialOrd<Fixed> for CmpNorm {
    fn partial_cmp(&self, n: &Fixed) -> Option<Ordering> {
        let m1sq = normsq(&self.0);
        let nraw = n.raw() as i64;
        let m2sq = nraw * nraw;
        Some(m1sq.cmp(&m2sq))
    }
}

impl Norm<Fixed> for Vec2<Fixed> {
    fn norm(&self) -> Fixed {
        let big = self.x.abs().max_(self.y.abs());
        if big == fixed!(0) { return fixed!(0); }
        // Dividing 1,0 by 0,2 (1 / 32768) or 0,1 (1 / 65536) overflows.
        let scale = fixed!(1) / big.max_(Fixed::from_raw(3));
        let x = self.x * scale;
        let y = self.y * scale;
        let d = x * x + y * y;
        d.sqrt() / scale
    }
    fn normalize(&self) -> Vec2<Fixed> {
        let m = self.norm();
        if m == fixed!(0) {
            Vec2::<Fixed>::new(fixed!(0), fixed!(0))
        } else {
            *self / m
        }
    }
}
impl Absolute<Fixed> for Fixed {
    fn abs(v: &Fixed) -> Fixed {
        v.abs()
    }
}
impl Axpy<Fixed> for Fixed {
    fn axpy(&mut self, a: &Fixed, x: &Fixed) {
        *self = *a * *x + *self
    }
}
impl BaseNum for Fixed {}

pub fn vec_from_angle(a: Fixed) -> Vec2<Fixed> {
	Vec2::new(a.cos(), a.sin())
}
pub fn vec_cross_z(v: Vec2<Fixed>) -> Vec2<Fixed> {
	Vec2::new(-v.y, v.x)
}
pub fn vec_angle(v: Vec2<Fixed>) -> Fixed {
	v.y.atan2(v.x)
}

pub trait FromGame<N> {
    fn from_game(self) -> N;
}
impl FromGame<f32> for Fixed {
    fn from_game(self) -> f32 {
        self.to_f32()
    }
}
impl FromGame<Vec2<f32>> for Vec2<Fixed> {
    fn from_game(self) -> Vec2<f32> {
        Vec2::new(self.x.to_f32(), self.y.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use fixmath::Fixed;
    use super::Norm;
    type Vec2 = ::na::Vec2<Fixed>;
    #[test]
    fn add() {
        let a = Vec2::new(fixed!(3), fixed!(5));
        let b = Vec2::new(fixed!(8), fixed!(13));
        assert_eq!(a + b, Vec2::new(fixed!(11), fixed!(18)));
    }
    #[test]
    fn scale() {
        assert_eq!(
            Vec2::new(fixed!(3), fixed!(5)) * fixed!(5),
            Vec2::new(fixed!(15), fixed!(25))
        );
    }
    #[test]
    fn div() {
        let f = Fixed::from_f32(1.2499952316466079);
        let r = fixed!(1) / f;
        println!("{:?} : {:?}", f, r);
        assert!(r != fixed!(0));
        assert!((fixed!(1) / r - f).abs() < Fixed::from_f32(0.1));
    }
    fn check_norm(v: Vec2) {
        let n = v.norm();
        let vp = v / n;
        let np = vp.norm();
        println!("v: {:?}\nn: {}\nvp: {:?}\nnp: {}", v, n, vp, np);
        assert!(vp.x <= fixed!(1));
        assert!(vp.y <= fixed!(1));
        assert!(np < fixed!(2));
    }
    #[test]
    fn norm() {
        assert_eq!(Vec2::new(fixed!(3), fixed!(4)).norm(), fixed!(5));
        check_norm(Vec2::new(fixed!(3), fixed!(4)));
        check_norm(Vec2::new(fixed!(300), fixed!(400)));
    }
}
