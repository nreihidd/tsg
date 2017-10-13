use fixvec::FromGame;
use fixmath::Fixed;
use na::{self, Vec2, Vec4, Mat4, Norm, Dot};
use std::f32::consts::PI as PI32;
use gamerender::effect::{Effect, dust_effect, charge_effect, parried_effect, directed_effect};
use ::{dbg_lines, DbgTimestops, vec_from_angle, vec_cross_z, vec_angle};
use game::CommitEffect;
use rand::random;

pub fn parry(dir: Vec2<Fixed>, offender_position: Vec2<Fixed>, radius: Fixed) -> CommitEffect {
    Box::new(move |presentation| {
        fn f(t: f32) -> Vec4<f32> { Vec4::new(0.0, 0.0, 1.0, 1.0 * t) }
        for _ in 0..10 {
            let a = vec_angle(dir.from_game()) + random::<f32>() * PI32 - PI32 / 2.0;
            presentation.add_effect(directed_effect(offender_position.from_game(), vec_from_angle(a) * 3.0, radius.from_game(), 20, f));
        }
        presentation.play_sound("assets/parry.ogg", offender_position.from_game());
    })
}

pub fn hit(blocked: bool, dir: Vec2<Fixed>, position: Vec2<Fixed>, radius: Fixed) -> CommitEffect {
    Box::new(move |presentation| {
        if blocked {
            fn f(t: f32) -> Vec4<f32> { Vec4::new(0.0, 1.0, 0.0, 1.0 * t) }
            for _ in 0..10 {
                let a = vec_angle(dir.from_game()) + random::<f32>() * PI32 - PI32 / 2.0;
                presentation.add_effect(directed_effect(position.from_game(), vec_from_angle(a) * 3.0, radius.from_game(), 20, f));
            }
            presentation.play_sound("assets/block.ogg", position.from_game());
        } else {
            fn f(t: f32) -> Vec4<f32> { Vec4::new(1.0, 0.0, 0.0, 1.0 * t) }
            for _ in 0..10 {
                let a = vec_angle(dir.from_game()) + random::<f32>() * PI32 - PI32 / 2.0;
                presentation.add_effect(directed_effect(position.from_game(), vec_from_angle(a) * 3.0, radius.from_game(), 20, f));
            }
            presentation.play_sound("assets/hit.ogg", position.from_game());
        }
    })
}

pub fn attack_wall(dir: Vec2<Fixed>, position: Vec2<Fixed>, radius: Fixed) -> CommitEffect {
    Box::new(move |presentation| {
        fn f(t: f32) -> Vec4<f32> { Vec4::new(0.5, 0.5, 0.5, 1.0 * t) }
        for _ in 0..10 {
            let a = vec_angle(dir.from_game()) + random::<f32>() * PI32 - PI32 / 2.0;
            presentation.add_effect(directed_effect(position.from_game(), vec_from_angle(a) * 3.0, radius.from_game(), 30, f));
        }
        presentation.play_sound("assets/hit-wall.ogg", position.from_game());
    })
}

pub fn death(position: Vec2<Fixed>, radius: Fixed) -> CommitEffect {
    Box::new(move |presentation| {
        fn f(t: f32) -> Vec4<f32> { Vec4::new(1.0, 0.0, 0.0, 1.0 * t) }
        for _ in 0..10 { presentation.add_effect(parried_effect(position.from_game(), radius.from_game(), 30, f)); }
        presentation.play_sound("assets/death.ogg", position.from_game());
    })
}