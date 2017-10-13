use drawing::*;
use na::{Vec2, Vec4, Mat4, Dot};
use rand::random;
use ::{random_direction, vec_cross_z};
use std::f32::consts::PI as PI32;
use gamerender;
use glium;

pub struct Effect {
	pub tick: Box<FnMut(&mut glium::Frame, &Mat4<f32>, &mut gamerender::Renderers) -> bool>,
}
pub fn fading_circle_effect(position: Vec2<f32>, radius: f32, duration: u32, color: fn(f32) -> Vec4<f32>) -> Effect {
	let mut ttl = duration;
	Effect {
		tick: Box::new(move |frame, view, renderer| {
			ttl -= 1;
			let t = ttl as f32 / duration as f32;
			renderer.model_renderer.draw_model_at(frame, view, &renderer.model_renderer.circle, position, radius, radius, color(t));
			ttl <= 0
		})
	}
}
pub fn parried_effect(position: Vec2<f32>, radius: f32, duration: u32, color: fn(f32) -> Vec4<f32>) -> Effect {
	directed_effect(position, random_direction() * 3.0, radius, duration, color)
}
pub fn directed_effect(position: Vec2<f32>, vel: Vec2<f32>, radius: f32, duration: u32, color: fn(f32) -> Vec4<f32>) -> Effect {
	let mut ttl = duration;
	let mut p = position.clone();
    let mut r = radius;
	Effect {
		tick: Box::new(move |frame, view, renderer| {
			ttl -= 1;
            let t = ttl as f32 / duration as f32;
			p = p + vel * t;
            r = radius * t;
			renderer.sprite_renderer.draw_sprite(frame, view,  &mut renderer.textures, "assets/circle.png", p, r, r, 0.0, color(ttl as f32 / duration as f32));
			ttl <= 0
		})
	}
}

fn sine_ease(t: f32) -> f32 {
	(t * PI32 / 2.0).sin()
}

pub fn dust_effect(mut position: Vec2<f32>, dir: Vec2<f32>) -> Effect {
	let vel = dir * -0.5 + random_direction() * 6.0 * random::<f32>();
	let mut ttl = 30;
	let mut rot = random::<f32>() * PI32 * 2.0;
	let drot = random::<f32>() * 0.1 * if vec_cross_z(dir).dot(&vel) < 0.0 { 1.0 } else { -1.0 };
	Effect {
		tick: Box::new(move |frame, view, renderer| {
			ttl -= 1;
			let t = ttl as f32 / 30.0;
			position = position + vel * t;
			let alpha = if t >= 0.8 {
				sine_ease((1.0 - t) / 0.2) * 0.25
			} else {
				sine_ease(t / 0.8) * 0.25
			};
			rot += drot * t;
			let radius = 32.0 + 16.0 * (1.0 - t);
			let color = Vec4::new(1.0, 1.0, 1.0, alpha * 0.5);
			renderer.sprite_renderer.draw_sprite(frame, &view, &mut renderer.textures, "assets/dust4.png", position, radius, radius, rot, color);
			ttl <= 0
		})
	}
}
struct Countdown {
	m: u32,
	t: u32,
}
impl Countdown {
	fn new(m: u32) -> Countdown {
		Countdown {
			m: m,
			t: m,
		}
	}
	fn tick(&mut self) -> f32 {
		if self.is_done() { return 0.0 }
		self.t -= 1;
		self.t as f32 / self.m as f32
	}
	fn is_done(&self) -> bool {
		self.t == 0
	}
}

pub fn charge_effect(position: Vec2<f32>) -> Effect {
	let position = position;
	let start = position + random_direction() * (25.0 + 50.0 * random::<f32>());
	let mut ttl = Countdown::new(20);
	let mut rotation = random::<f32>() * PI32;
	let rot_spin = random::<f32>() * PI32 * 2.0 / 60.0f32;
	Effect {
		tick: Box::new(move |frame, view, renderer| {
			let t = ttl.tick();
			rotation += rot_spin;
			let color = Vec4::new(1.0, 1.0, 1.0, t);
			renderer.sprite_renderer.draw_sprite(frame, &view, &mut renderer.textures, "assets/star.png", (start * t + position * (1.0 - t)), 16.0, 16.0, rotation, color);
			ttl.is_done()
		})
	}
}

pub fn floating_text_effect(s: String, position: Vec2<f32>, size: f32, color: Vec4<f32>) -> Effect {
	let len = s.chars().count() as f32;
	let mut p = position.clone();
	let vel = Vec2::new(0.0, 0.5);
	let mut ttl = Countdown::new(60);
	Effect {
		tick: Box::new(move |frame, view, renderer| {
			let t = ttl.tick();
			p = p + vel;
			renderer.text.draw_string(
				&(*view * translate(p.x, p.y, 50.0) * scale(size, size, 1.0) * translate(-(len * 2.0) / 2.0, 0.0, 0.0)),
				frame,
				Vec4::new(color.x, color.y, color.z, color.w * t),
				&s,
			);
			ttl.is_done()
		})
	}
}
