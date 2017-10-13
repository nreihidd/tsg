extern crate libc;
extern crate num;
use rustc_serialize::{Encodable, Decodable, Encoder, Decoder};
use std::convert::From;
use rand::{Rng, Rand};
use std::i32;
use std::ops::{Add, Sub, Mul, Div, Rem, Neg};
use std::cmp::{max, min};
use num::traits::{One, Zero};
use self::libc::{int32_t};
type Q16 = int32_t;
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fixed {
    value: Q16,
}
impl ::std::fmt::Debug for Fixed {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        write!(f, "Q:{:.5}", /* self.value,*/ self.to_f64())
    }
}
pub const PI: Fixed = Fixed { value: 205887 };
pub const E: Fixed = Fixed { value: 178145 };
pub const ONE: Fixed = Fixed { value: 0x00010000 };
pub const ONE_HUNDRENTH: Fixed = Fixed { value: 655 };
#[link(name="fixmath", kind="static")]
extern {
    fn fix16_add(a: Q16, b: Q16) -> Q16;
    fn fix16_sub(a: Q16, b: Q16) -> Q16;
    fn fix16_mul(a: Q16, b: Q16) -> Q16;
    fn fix16_div(a: Q16, b: Q16) -> Q16;

    fn fix16_sadd(a: Q16, b: Q16) -> Q16;
    fn fix16_ssub(a: Q16, b: Q16) -> Q16;
    fn fix16_smul(a: Q16, b: Q16) -> Q16;
    fn fix16_sdiv(a: Q16, b: Q16) -> Q16;

    fn fix16_sin(angle: Q16) -> Q16;
    fn fix16_cos(angle: Q16) -> Q16;
    fn fix16_tan(angle: Q16) -> Q16;
    fn fix16_asin(v: Q16) -> Q16;
    fn fix16_acos(v: Q16) -> Q16;
    fn fix16_atan(v: Q16) -> Q16;
    fn fix16_atan2(y: Q16, x: Q16) -> Q16;

    fn fix16_sqrt(v: Q16) -> Q16;

    fn fix16_exp(v: Q16) -> Q16;
}
macro_rules! fixed {
    ($e:expr) => (::fixmath::Fixed::from_i32($e as i32));
}
impl Fixed {
    pub fn raw(&self) -> i32 {
        self.value
    }
    pub fn from_raw(d: i32) -> Fixed {
        Fixed { value: d }
    }
    pub fn parse(s: &str) -> Option<Fixed> {
        use num::{BigRational, BigInt};
        use num::traits::{ToPrimitive};
        let ten = &BigInt::from(10u8);
        let mut numerator = BigInt::zero();
        let mut denominator = BigInt::one();
        let mut cs = s.chars().peekable();
        let sign = if cs.peek() == Some(&'-') {
            cs.next();
            -BigInt::one()
        } else {
            BigInt::one()
        };
        let mut in_denom = false;
        for c in cs {
            match c {
                '0'...'9' => {
                    numerator = numerator * ten;
                    numerator = numerator + BigInt::from(c as u32 - '0' as u32);
                    if in_denom {
                         denominator = denominator * ten;
                    }
                },
                '.' if !in_denom => in_denom = true,
                _ => return None,
            }
        }
        let ratio = BigRational::new(sign * numerator, denominator);
        (ratio * BigRational::from_integer(BigInt::from(65536i32))).round().to_integer().to_i32().map(|i| Fixed::from_raw(i))
    }
    pub fn from_i32(a: i32) -> Fixed {
        Fixed { value: a * ONE.value }
    }
    pub fn from_f32(a: f32) -> Fixed {
        Fixed { value: (a * 65536.0) as i32 }
    }
    pub fn from_f64(a: f64) -> Fixed {
        Fixed { value: (a * 65536.0) as i32 }
    }
    pub fn to_f32(self) -> f32 {
        self.value as f32 / ONE.value as f32
    }
    pub fn to_f64(self) -> f64 {
        self.value as f64 / ONE.value as f64
    }
    pub fn to_i32(self) -> i32 {
        self.value >> 16
    }
    pub fn sin(&self) -> Fixed {
        unsafe { Fixed { value: fix16_sin(self.value) } }
    }
    pub fn cos(&self) -> Fixed {
        unsafe { Fixed { value: fix16_cos(self.value) } }
    }
    pub fn tan(&self) -> Fixed {
        unsafe { Fixed { value: fix16_tan(self.value) } }
    }
    pub fn asin(&self) -> Fixed {
        unsafe { Fixed { value: fix16_asin(self.value) } }
    }
    pub fn acos(&self) -> Fixed {
        unsafe { Fixed { value: fix16_acos(self.value) } }
    }
    pub fn atan(&self) -> Fixed {
        unsafe { Fixed { value: fix16_atan(self.value) } }
    }
    pub fn atan2(&self, x: Fixed) -> Fixed {
        unsafe { Fixed { value: fix16_atan2(self.value, x.value) } }
    }
    pub fn sqrt(&self) -> Fixed {
        unsafe { Fixed { value: fix16_sqrt(self.value) } }
    }
    pub fn exp(&self) -> Fixed {
        unsafe { Fixed { value: fix16_exp(self.value) } }
    }
    pub fn floor(&self) -> Fixed {
        Fixed { value: self.value & 0xFFFF0000 }
    }
    pub fn ceil(&self) -> Fixed {
        Fixed {
            value: {
                let w = self.value >> 16;
                let c = if self.value & 0xFFFF > 0 {
                    w + 1
                } else {
                    w
                };
                c << 16
            }
        }
    }
    // TODO: Remove max_ and min_ once ord_max_min is stable
    pub fn max_(&self, other: Fixed) -> Fixed {
        Fixed { value: max(self.value, other.value) }
    }
    pub fn min_(&self, other: Fixed) -> Fixed {
        Fixed { value: min(self.value, other.value) }
    }
    pub fn abs(&self) -> Fixed {
        Fixed { value: self.value.abs() }
    }
}

impl Encodable for Fixed {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_i32(self.value)
        // s.emit_f64(self.to_f64())
    }
}
impl Decodable for Fixed {
    fn decode<D: Decoder>(d: &mut D) -> Result<Fixed, D::Error> {
        d.read_i32().map(|v| Fixed { value: v })
        // d.read_f64().map(|v| Fixed::from_f64(v))
    }
}

use std::fmt::{Display, Formatter, Error};
impl Display for Fixed {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.write_str(&format!("{},{}", self.value >> 16, self.value & 0xFFFF))
    }
}

impl Rand for Fixed {
    fn rand<R: Rng>(rng: &mut R) -> Fixed {
        let n: u32 = rng.gen();
        Fixed { value: (n & 0xFFFF) as i32 }
    }
}

impl From<i32> for Fixed {
    fn from(v: i32) -> Fixed {
        Fixed { value: v * ONE.value }
    }
}
impl From<u32> for Fixed {
    fn from(v: u32) -> Fixed {
        From::<i32>::from(v as i32)
    }
}
impl Add for Fixed {
    type Output = Fixed;
    fn add(self, rhs: Fixed) -> Fixed {
        Fixed { value: unsafe { fix16_add(self.value, rhs.value) } }
    }
}
impl Sub for Fixed {
    type Output = Fixed;
    fn sub(self, rhs: Fixed) -> Fixed {
        Fixed { value: unsafe { fix16_sub(self.value, rhs.value) } }
    }
}
impl Mul for Fixed {
    type Output = Fixed;
    fn mul(self, rhs: Fixed) -> Fixed {
        let r = unsafe { fix16_mul(self.value, rhs.value) };
        if r == i32::MIN {
            panic!(format!("Multiplication overflow {} * {}", self, rhs));
        }
        Fixed { value: r }
    }
}
impl Div for Fixed {
    type Output = Fixed;
    fn div(self, rhs: Fixed) -> Fixed {
        let r = unsafe { fix16_div(self.value, rhs.value) };
        if r == i32::MIN {
            panic!(format!("Division overflow {} / {}", self, rhs));
        }
        Fixed { value: r }
    }
}
impl Rem for Fixed {
    type Output = Fixed;
    fn rem(self, rhs: Fixed) -> Fixed {
        let n = (self / rhs).floor();
        self - n * rhs
    }
}
impl Neg for Fixed {
    type Output = Fixed;
    fn neg(self) -> Fixed {
        Fixed { value: -self.value }
    }
}
impl One for Fixed {
    fn one() -> Fixed {
        fixed!(1)
    }
}
impl Zero for Fixed {
    fn zero() -> Fixed {
        fixed!(0)
    }
    fn is_zero(&self) -> bool {
        *self == fixed!(0)
    }
}

pub fn normalized_angle(angle: Fixed) -> Fixed {
	let a = angle % (PI * fixed!(2));
	if a < -PI {
        a + PI * fixed!(2)
    } else if a > PI {
        a - PI * fixed!(2)
    } else {
        a
    }
}
pub fn min_angle_diff(a: Fixed, b: Fixed) -> Fixed {
	normalized_angle(a - b)
}

#[cfg(test)]
mod tests {
    use super::{Fixed};
    #[test]
    fn add() {
        let a = fixed!(6);
        let b = fixed!(132);
        println!("{:?} {:?}", a, b);
        assert_eq!(a.to_i32(), 6);
        assert_eq!(b.to_i32(), 132);
        assert_eq!((a + b).to_i32(), 138);
    }
    #[test]
    fn fraction() {
        let a = fixed!(1) / fixed!(2);
        assert_eq!(a.to_f32(), 0.5);
        assert_eq!(fixed!(1) / a, fixed!(2));
    }
    #[test]
    fn parse() {
        assert_eq!(fixed!(1) / fixed!(2), Fixed::parse("0.5").unwrap());
        assert_eq!(fixed!(-1) / fixed!(2), Fixed::parse("-0.5").unwrap());
        assert_eq!(fixed!(13) / fixed!(10), Fixed::parse("1.3").unwrap());
        assert_eq!(fixed!(-13) / fixed!(10), Fixed::parse("-1.3").unwrap());
        assert_eq!(fixed!(10001) / fixed!(10), Fixed::parse("1000.1").unwrap());
        assert_eq!(fixed!(-10001) / fixed!(10), Fixed::parse("-1000.1").unwrap());
        assert_eq!(fixed!(10001) / fixed!(10000), Fixed::parse("1.0001").unwrap());
        assert_eq!(fixed!(-10001) / fixed!(10000), Fixed::parse("-1.0001").unwrap());
        assert_eq!(fixed!(5), Fixed::parse("5").unwrap());
    }
}
