#[macro_use]
extern crate glium;

#[macro_use]
mod resource;
#[macro_use]
mod fixmath;
mod fixvec;
use fixmath::Fixed;
extern crate sdl2;
extern crate nalgebra as na;
extern crate image;
extern crate time;
extern crate byteorder;
extern crate rmp_serialize as msgpack;
extern crate sha1;
use na::{Vec2, Mat4, Norm, Inv};
use std::convert::From;

use glium::Surface;

extern crate rand;
use rand::random;
use std::f32::consts::PI as PI32;

extern crate num;

use std::collections::BTreeMap;

mod drawing;
mod game;
mod gamerender;
mod network;
mod engine;
mod conductor;
mod editor;
mod load_level;

use std::sync::mpsc::{channel};
use gamerender::GameRenderer;
use game::{Inputs};
use drawing::{color, translate, scale};

fn to_game_vec(v: Vec2<f32>) -> Vec2<Fixed> {
	Vec2::new(Fixed::from_f32(v.x), Fixed::from_f32(v.y))
}

fn joystick_axis(joystick: &sdl2::joystick::Joystick, axis: u32) -> f32 {
	match joystick.axis(axis) {
		Ok(v) => v as f32 / i16::max_value() as f32,
		_ => 0.0,
	}
}

fn read_inputs(j: &sdl2::joystick::Joystick) -> Inputs {
	fn deadzone(v: Vec2<f32>, deadzone_radius: f32) -> Vec2<f32> {
		if v.norm() < deadzone_radius {
			Vec2::new(0.0, 0.0)
		} else {
			let mut a = v.normalize() * ((v.norm() - deadzone_radius) / (1.0 - deadzone_radius));
			a.y *= -1.0;
			a
		}
	}
	let movement = deadzone(Vec2::new(joystick_axis(j, 0), joystick_axis(j, 1)), 0.3);
	let button = |i| j.button(i).unwrap_or(false);
	Inputs {
		movement: to_game_vec(movement),
		facing: to_game_vec(deadzone(Vec2::new(joystick_axis(j, 3), joystick_axis(j, 4)), 0.3)),
		roll: movement.sqnorm() > 0.0 && joystick_axis(j, 5) > 0.0,
		attack: button(5),
		attack_dumb: button(0),
		attack_special: button(1),
		parry: joystick_axis(j, 2) > 0.0,
		block: button(4),
		dbg_buttons: (0..j.num_buttons()).map(button).collect(),
	}
}

fn draw_square(batch: &mut drawing::dynamic_geometry::DynamicBatch, x: f32, y: f32, hw: f32, hh: f32, color: na::Vec4<f32>) {
	let f = |v: [f32; 2]| drawing::dynamic_geometry::ColoredVertex { a_position: v, a_color: color.as_ref().clone() };
	let v1 = f([x - hw, y - hh]);
	let v2 = f([x + hw, y - hh]);
	let v3 = f([x + hw, y + hh]);
	let v4 = f([x - hw, y + hh]);
	batch.draw_triangle(v1, v2, v3);
	batch.draw_triangle(v1, v3, v4);
}

fn dbg_draw_joystick(batch: &mut drawing::dynamic_geometry::DynamicBatch, joystick: &sdl2::joystick::Joystick) {
	let red = color(0xff000088);
	for axis in 0..joystick.num_axes() {
		let v = joystick_axis(&joystick, axis);
		draw_square(batch, (axis as f32 - (joystick.num_axes() - 1) as f32 / 2.0) * 50.0, 100.0 + v * 50.0, 10.0, 10.0, red);
	}
	for button in 0..joystick.num_buttons() {
		let on = match joystick.button(button) { Ok(true) => true, _ => false };
		draw_square(batch, (button as f32 - (joystick.num_buttons() - 1) as f32 / 2.0) * 50.0, if on { 0.0 } else { -50.0 }, 10.0, 10.0, red);
	}
	for hat in 0..joystick.num_hats() {
		use sdl2::joystick::HatState;
		let x = (hat as f32 - (joystick.num_hats() - 1) as f32) / 2.0 * 50.0;
		let y = -100.0;
		let s = match joystick.hat(hat) { Ok(s) => s as u32, _ => 0 };
		let dx = if s & HatState::Right as u32 != 0 { 1.0 } else if s & HatState::Left as u32 != 0 { -1.0 } else { 0.0 };
		let dy = if s & HatState::Up as u32 != 0 { 1.0 } else if s & HatState::Down as u32 != 0 { -1.0 } else { 0.0 };
		draw_square(batch, x + dx * 20.0, y + dy * 20.0, 10.0, 10.0, red);
	}
}

fn get_a_joystick(joystick_subsystem: &sdl2::JoystickSubsystem) -> sdl2::joystick::Joystick {
	let available = joystick_subsystem.num_joysticks().unwrap_or(0);
	let joystick = (0..available).flat_map(|id| joystick_subsystem.open(id).ok()).next();
	match joystick {
		Some(joystick) => {
			println!("Using joystick '{}'", joystick.name());
			joystick
		},
		None => {
			panic!("No joysticks available!");
		}
	}
}

fn normalized_angle(angle: f32) -> f32 {
	use std::f32::consts::PI;
	let a = angle % (PI * 2.0);
	if a < -PI { a + PI * 2.0 } else if  a > PI { a - PI * 2.0 } else { a }
}
fn min_angle_diff(a: f32, b: f32) -> f32 {
	normalized_angle(a - b)
}

fn vec_from_angle(a: f32) -> Vec2<f32> {
	Vec2::new(a.cos(), a.sin())
}
fn vec_cross_z(v: Vec2<f32>) -> Vec2<f32> {
	Vec2::new(-v.y, v.x)
}

fn vec_angle(v: Vec2<f32>) -> f32 {
	v.y.atan2(v.x)
}
fn random_direction() -> Vec2<f32> {
	let a = random::<f32>() * PI32 * 2.0;
	Vec2::<f32>::new(a.cos(), a.sin())
}

#[macro_use]
extern crate lazy_static;
use std::sync::Mutex;
lazy_static! {
	static ref dbg_lines : Mutex<Vec<[f32; 4]>> = Mutex::new(vec![]);
}

trait LogOk<U> {
	fn log_ok(self: Self) -> U;
}
impl<T, E: std::fmt::Debug> LogOk<Option<T>> for Result<T, E> {
	fn log_ok(self) -> Option<T> {
		match self {
			Ok(t) => Some(t),
			Err(e) => {
				println!("{:?}", e);
				None
			},
		}
	}
}

extern crate rustc_serialize;
use rustc_serialize::{json, Decodable, Encodable};
use std::io::Write;
use std::io::Read;
use std::fs::File;
fn load_from_json_bytes<T: Decodable>(bytes: Vec<u8>) -> Option<T> {
	String::from_utf8(bytes).log_ok().and_then(|s| json::decode(&s).log_ok())
}
fn load_from_json_file<T: Decodable, P: AsRef<std::path::Path>>(path: P) -> Option<T> {
	File::open(path).log_ok()
		.and_then(|mut file| {
			let mut s = String::new();
			file.read_to_string(&mut s).log_ok().and_then(|_| json::decode(&s).log_ok())
		})
}
fn dbg_write_json_file<T: Encodable, P: AsRef<std::path::Path>>(path: P, t: &T) {
	json::encode(t).log_ok().map(|s| {
		File::create(path).log_ok().map(|mut file| {
			file.write(&s.into_bytes()).unwrap();
		});
	});
}

use std::collections::HashMap;
lazy_static! {
	static ref dbg_colors: Mutex<Vec<(&'static str, na::Vec4<f32>)>> = Mutex::new(vec![]);
}
fn str_to_color(s: &'static str) -> na::Vec4<f32> {
	let mut m = dbg_colors.lock().unwrap();
	match m.iter().find(|x| x.0 == s) { Some(&(_, c)) => return c, None => () };
	let len = m.len();
	let c = {
		//use rand::{Rng, XorShiftRng, random};
		//let mut rng: XorShiftRng = random();
		let hue = (len as f32 / 16.0 + (len % 2) as f32 / 2.0) % 1.0; // rng.gen::<f32>();
		let sat = 1.0; // rng.gen::<f32>() * 0.5 + 0.5;
		let val = 0.5; // rng.gen::<f32>() * 0.5 + 0.5;
		// https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSV
		let c = val * sat;
		let h = hue * 6.0;
		let x = c * (1.0 - (h % 2.0 - 1.0).abs());
		let (r, g, b) = match h {
			0.0...1.0 => (c, x, 0.0),
			1.0...2.0 => (x, c, 0.0),
			2.0...3.0 => (0.0, c, x),
			3.0...4.0 => (0.0, x, c),
			4.0...5.0 => (x, 0.0, c),
			_         => (c, 0.0, x),
		};
		let m = val - c;
		na::Vec4::new(r + m, g + m, b + m, 1.0)
	};
	m.push((s, c));
	c
}
fn render_key(view: &Mat4<f32>, frame: &mut glium::Frame, text: &mut drawing::text::Text) {
	let mut view = view.clone();
	let m = dbg_colors.lock().unwrap();
	for &(s, c) in m.iter() {
		view = view * translate(0.0, 2.0, 0.0);
		text.draw_string(&view,  frame, c, s);
	}
}

use std::collections::VecDeque;
const TIMESTOP_FRAMES: usize = 180;
const TIMESTOP_DISPLAY_HEIGHT: f32 = 100.0;
struct DbgTimestopsRenderer {
	frames: VecDeque<Vec<(na::Vec4<f32>, u64)>>,
}
impl DbgTimestopsRenderer {
	fn new() -> DbgTimestopsRenderer {
		DbgTimestopsRenderer { frames: VecDeque::new() }
	}
	fn add_frame(&mut self, dbg_timestops: DbgTimestops) {
		if self.frames.len() > TIMESTOP_FRAMES {
			self.frames.pop_front();
		}
		self.frames.push_back(dbg_timestops.frame);
	}
	fn render(&mut self, x: f32, y: f32, batch: &mut drawing::gl_lines::LineBatch) {
		let w = TIMESTOP_FRAMES as f32;
		let h = TIMESTOP_DISPLAY_HEIGHT;
		batch.draw_line(Vec2::new(x, y), Vec2::new(x + w, y), color(0x000000ff));
		batch.draw_line(Vec2::new(x, y + h), Vec2::new(x + w, y + h), color(0x000000ff));

		for (i, frame) in self.frames.iter().enumerate() {
			let x = x + i as f32;
			let mut y = y;
			for stack in frame.iter() {
				let c = &stack.0;
				let sh = stack.1 as f32 / 16_666_667.0 * TIMESTOP_DISPLAY_HEIGHT;
				batch.draw_line(Vec2::new(x, y), Vec2::new(x, y + sh), *c);
				y += sh;
			}
		}
	}
}

struct DbgLinegraphRenderer {
	frames: VecDeque<BTreeMap<u64, i64>>, // id -> val
}
impl DbgLinegraphRenderer {
	fn new() -> DbgLinegraphRenderer {
		DbgLinegraphRenderer { frames: VecDeque::new() }
	}
	fn add_frame(&mut self) {
		if self.frames.len() > TIMESTOP_FRAMES {
			self.frames.pop_front();
		}
		self.frames.push_back(BTreeMap::new());
	}
	fn add_value(&mut self, key: u64, val: i64) {
		self.frames.back_mut().map(|m| m.insert(key, val));
	}
	fn render(&mut self, x: f32, y: f32, batch: &mut drawing::gl_lines::LineBatch) {
		let w = TIMESTOP_FRAMES as f32;
		let h = TIMESTOP_DISPLAY_HEIGHT;

		let colors = vec![
			color(0xff0000ff), color(0x00ff00ff),
			color(0x0000ffff)
		];

		let y = y + h / 2.0;
		batch.draw_line(Vec2::new(x, y), Vec2::new(x + w, y), color(0x000000ff));
		for (i, (frame_a, frame_b)) in self.frames.iter().zip(self.frames.iter().skip(1)).enumerate() {
			let x = x + i as f32;
			for ((v_a, v_b), c) in frame_b.iter().flat_map(|(id, &v_b)| frame_a.get(id).map(move |&v_a| (v_a, v_b))).zip(colors.iter().cycle()) {
				batch.draw_line(Vec2::new(x, y + v_a as f32), Vec2::new(x + 1.0, y + v_b as f32), *c);
			}
		}
	}
}
pub struct DbgNetworkTimings {
	linegraph: DbgLinegraphRenderer,
}
impl DbgNetworkTimings {
	fn new() -> DbgNetworkTimings {
		DbgNetworkTimings { linegraph: DbgLinegraphRenderer::new() }
	}
	fn add_value(&mut self, key: u64, val: i64) {
		self.linegraph.add_value(key, val);
	}
}

#[derive(Debug)]
pub struct DbgTimestops {
	frame: Vec<(na::Vec4<f32>, u64)>,
	last_stop: u64,
}
impl DbgTimestops {
	fn new() -> DbgTimestops {
		DbgTimestops {
			last_stop: time::precise_time_ns(),
			frame: vec![],
		}
	}
	fn timestop(&mut self, s: &'static str) {
		let c = str_to_color(s);
		let next_stop = time::precise_time_ns();
		let elapsed = next_stop - self.last_stop;
		self.frame.push((c, elapsed));
		self.last_stop = next_stop;
	}
}

pub struct Presentation {
	audio: Audio,
	renderer: GameRenderer,
	last_focus: Vec2<f32>,
}
impl Presentation {
	fn play_sound(&mut self, path: &str, position: Vec2<f32>) {
		self.audio.play_sound(path, position - self.last_focus);
	}
	fn add_effect(&mut self, effect: gamerender::effect::Effect) {
		self.renderer.add_effect(effect);
	}
}

struct Audio {
	_context: sdl2::mixer::Sdl2MixerContext,
	sound_cache: std::collections::HashMap<String, sdl2::mixer::Chunk>,
	num_channels: i32,
	muted: bool,
}
impl Audio {
	fn new() -> Audio {
		// https://jadpole.github.io/arcaders/arcaders-1-13
		let context = sdl2::mixer::init(sdl2::mixer::INIT_OGG).unwrap();
		sdl2::mixer::open_audio(44100, sdl2::mixer::AUDIO_S16LSB, 2, 1024).unwrap();
		let num_channels = 32;
		sdl2::mixer::allocate_channels(num_channels);
		Audio {
			_context: context,
			num_channels: num_channels,
			sound_cache: std::collections::HashMap::new(),
			muted: false,
		}
	}
	fn play_sound(&mut self, path: &str, offset: Vec2<f32>) {
		if !self.sound_cache.contains_key(path) {
			self.sound_cache.insert(path.to_string(), sdl2::mixer::Chunk::from_file(std::path::Path::new(path)).unwrap());
		}
		let chunk = self.sound_cache.get(path).unwrap();
		let (angle, distance) = if offset.x == 0.0 && offset.y == 0.0 {
			(0, 0)
		} else {
			let angle = (-vec_angle(offset) / PI32 * 180.0 + 90.0 + 360.0) as i16;
			let distance = ((offset.norm() / 2000.0).max(0.0).min(1.0) * std::u8::MAX as f32) as u8;
			(angle, distance)
		};
		match sdl2::mixer::Channel::all().play(chunk, 0) {
			Ok(channel) => {
				channel.set_position(angle, distance).unwrap();
			},
			Err(_) => {
				// TODO: Not enough channels, allocate more and try again or just ignore the sound
			}
		}
	}
	fn toggle_mute(&mut self) {
		self.muted = !self.muted;
		if self.muted {
			sdl2::mixer::Channel::all().set_volume(0);
		} else {
			sdl2::mixer::Channel::all().set_volume(sdl2::mixer::MAX_VOLUME);
		}
	}
}

fn main() {
	if std::env::args().nth(1).map(|s| s == "editor").unwrap_or(false) {
		editor::main_editor();
	} else {
		main_game();
	}
}

fn main_game() {
    use drawing::glium_sdl2::DisplayBuild;
	use glium::backend::Facade;

	// This might actually be 59.94Hz (16_683_350) and not 60Hz, but this value is only used as an estimate to sleep close to the next frame, so small differences don't matter
	let mut frame_duration = 16_666_667; // 6_944_444; // Might not be necessary, just draw as fast as allowed?
	// Logic on the other hand can run at any arbitrary frame rate, so long as all clients run it at roughly the same one.
	let logic_duration = 16_666_667;
    let sdl_context = sdl2::init().expect("Could not initialize SDL2");
    let video_subsystem = sdl_context.video().expect("Could not initialize video subsystem");

	// This is to try to force vsync on windows (it doesn't work on ubuntu, only the vsync checkbox in nvidia settings seems to)
	// Maybe the game needs to be fullscreen before setting this?
	// TODO: Check whether this works for windows
	if !video_subsystem.gl_set_swap_interval(-1) { video_subsystem.gl_set_swap_interval(1); }

	video_subsystem.gl_attr().set_multisample_buffers(1);
	video_subsystem.gl_attr().set_multisample_samples(8);
	video_subsystem.gl_attr().set_stencil_size(8);
	video_subsystem.gl_attr().set_double_buffer(true);
    let mut display = video_subsystem.window("TSG", 800, 600).resizable().build_glium().unwrap();
	let mut fullscreen = false;
	let display_mode = video_subsystem.desktop_display_mode(0).unwrap();
	println!("{:?}", display_mode);
	display.window_mut().set_display_mode(Some(display_mode)).unwrap();
	let ctx = display.get_context().clone();

    let mut running = true;
    let mut event_pump = sdl_context.event_pump().unwrap();

	let joystick = get_a_joystick(&sdl_context.joystick().unwrap());

    let mut next_logic = time::precise_time_ns();
	let mut next_print = time::precise_time_ns() + 1_000_000_000;

	let mut frame_process_time = 0;
	let mut frame_total_time = 0;
	let mut frames = 0;

	let mut last_flip = time::precise_time_ns();


	// let mut text = text::Text::new(&display);

	let mut dbg_trail = drawing::trail::Trail::new(20.0, 60, 10, 0.5);

	let mut fps_message = String::new();
	let mut pred_message = String::new();

	let (game, level) = game::initial::make_initial_state();
	let (mut conductor, dbg_server) = if let Some(s) = std::env::args().nth(1) {
		let mut conductor = conductor::Conductor::remote(network::server::stream_to_channels(std::net::TcpStream::connect::<&str>(s.as_ref()).unwrap()));
		conductor.join();
		(conductor, channel().1)
	} else {
		(conductor::Conductor::local(game), network::server::server())
	};
	let mut buffered_frames = 0;
	conductor.set_num_buffered_frames(buffered_frames);

	let mut should_draw_key = false;

	let mut presentation = Presentation {
		audio: Audio::new(),
		renderer: GameRenderer::new(&level, &ctx),
		last_focus: Vec2::new(0.0, 0.0),
	};

	let mut dbg_timestops_renderer = DbgTimestopsRenderer::new();
	let mut dbg_network_timings = DbgNetworkTimings::new();
    while running {
		let mut dbg_timestops = DbgTimestops::new();
		let frame_start = time::precise_time_ns();
		dbg_timestops.timestop("Dbg Network Start");
		while let Ok(conn) = dbg_server.try_recv() {
			conductor.add_net_conn(conn);
		}

		dbg_timestops.timestop("Logic Start");

		while frame_start >= next_logic {
			dbg_network_timings.linegraph.add_frame();
			for e in conductor.advance(read_inputs(&joystick), &mut dbg_network_timings).into_iter().flat_map(|x| x) {
				e(&mut presentation);
			}
			next_logic += logic_duration;
        }
		dbg_timestops.timestop("Logic End");

		let last_view = {
			let interp_t = 1.0 - ((next_logic - frame_start) as f64 / logic_duration as f64) as f32;
			dbg_timestops.timestop("Render Start");
			let mut frame = display.draw();
	        frame.clear_color(1.0, 1.0, 1.0, 1.0);
			frame.clear_depth(1.0);

			let frame_number = conductor.get_buffered_frame().saturating_sub(1);
			conductor.simulate_to(frame_number);
			presentation.renderer.add_state_effects(conductor.get_state(frame_number));
			let (last_focus, view) = {
				let interp_test: Vec<_> = (0..10).rev().map(|i| conductor.get_state(frame_number.saturating_sub(i))).collect();
				if interp_test[interp_test.len() - 1] as *const game::GameState as usize == interp_test[interp_test.len() - 2] as *const game::GameState as usize {
					println!("No interpolation {}", frame_number);
				}
				presentation.renderer.render_state(conductor.get_local_player(), &mut frame, &interp_test[..], interp_t, &mut dbg_timestops)
			};
			presentation.last_focus = last_focus;
			let (hud_width, hud_height) = { let (w, h) = frame.get_dimensions(); (w as f32, h as f32) };
			let hud_view = na::OrthoMat3::<f32>::new(hud_width, hud_height, -50.0, 50.0).to_mat() * translate(-hud_width / 2.0, -hud_height / 2.0, 0.0);
			{
				let s = conductor.dbg_frames();
				let s2 =  &format!("{} projectiles", conductor.get_state(frame_number).projectiles.len());
				let mut draw_text_left_align = |x, y, s| {
					presentation.renderer.renderers.text.draw_string(&(hud_view * translate(x, y, 49.0) * scale(4.0, 4.5, 1.0) * translate(1.0, 0.0, 0.0)), &mut frame, color(0x00000099), s);
				};
				let x = 5.0;
				let mut y = hud_height - 5.0 - 4.5;
				draw_text_left_align(x, y, &fps_message[..]);
				y -= 11.0;
				draw_text_left_align(x, y, &s);
				y -= 11.0;
				draw_text_left_align(x, y, &pred_message);
				y -= 11.0;
				draw_text_left_align(x, y, s2);
			}
			presentation.renderer.renderers.trail_renderer.start(&mut frame);
			dbg_trail.add_point((::vec_from_angle(conductor.get_buffered_frame().saturating_sub(1) as f32 / 60.0 * PI32) * 400.0), color(0x00ff0000));
		    dbg_trail.fade_all();
	        presentation.renderer.renderers.trail_renderer.draw_trail(&view, &mut frame, &dbg_trail);

	        dbg_draw_joystick(&mut presentation.renderer.renderers.triangle_renderer.batch(&mut frame, &view), &joystick);
	        dbg_timestops.timestop("Drew Extras");
			// Render timestops
			{
				let mut batch = presentation.renderer.renderers.line_renderer.batch(&mut frame, &hud_view);
				dbg_timestops_renderer.render(20.0, 20.0, &mut batch);
				dbg_network_timings.linegraph.render(hud_width - TIMESTOP_FRAMES as f32 - 20.0, 20.0, &mut batch);
			}
	        dbg_timestops.timestop("Drew Timestops");
			if should_draw_key {
				let key_view = hud_view * translate(220.0, 20.0, 49.0) * scale(4.0, 4.5, 1.0);
				render_key(&key_view, &mut frame, &mut presentation.renderer.renderers.text);
		        dbg_timestops.timestop("Drew Timestops Key");
			}
	        frame.finish().unwrap();
	        dbg_timestops.timestop("Frame Post");

			// Sleep here
			let predicted_vblank = last_flip + frame_duration;
			loop {
				let now = time::precise_time_ns();
				if now < predicted_vblank {
					let time_until_vblank = predicted_vblank - now;
					if time_until_vblank > 3_000_000 {
						std::thread::sleep(std::time::Duration::new(0, (time_until_vblank - 3_000_000) as u32));
					} else {
						break;
					}
				} else {
					break;
				}
			}
			dbg_timestops.timestop("Sleep");

			glium::backend::Facade::get_context(&display).finish(); // Will vsync depending on "Sync to VBlank" in Nvidia X Server Settings
			dbg_timestops.timestop("Finish");
			last_flip = time::precise_time_ns();
			na::OrthoMat3::<f32>::new(hud_width, hud_height, -50.0, 50.0).to_mat().inv().unwrap() * view
		};
		dbg_timestops.timestop("Frame End");

		dbg_timestops_renderer.add_frame(dbg_timestops);

        for event in event_pump.poll_iter() {
            use sdl2::event::Event;
			use sdl2::keyboard::Keycode;

            match event {
                Event::Quit { .. } => running = false,
				Event::KeyDown { keycode: Some(keycode), .. } => match keycode {
					Keycode::Escape => running = false,
					Keycode::F => {
						fullscreen = !fullscreen;
						if fullscreen {
							let display_mode = video_subsystem.desktop_display_mode(0).unwrap();
							display.window_mut().set_display_mode(Some(display_mode)).unwrap();
							display.window_mut().set_fullscreen(sdl2::video::FullscreenType::True).unwrap();
						} else {
							display.window_mut().set_fullscreen(sdl2::video::FullscreenType::Off).unwrap();
						}
					},
					Keycode::LeftBracket => {
						buffered_frames = buffered_frames.saturating_sub(1);
						conductor.set_num_buffered_frames(buffered_frames);
					},
					Keycode::RightBracket => {
						buffered_frames = buffered_frames + 1;
						conductor.set_num_buffered_frames(buffered_frames);
					},
					Keycode::D => {
						conductor.dbg_drop_frame();
					},
					Keycode::S => {
						dbg_write_json_file("test.state.json", conductor.get_state(conductor.get_active_frame()));
					},
					Keycode::L => {
						// load_from_json("test.state.json").map(|g| game = g);
					},
					Keycode::M => {
						presentation.audio.toggle_mute();
					},
					Keycode::V => {
						frame_duration = if frame_duration == 16_666_667 { 6_944_444 } else { 16_666_667 };
					},
					Keycode::K => {
						should_draw_key = !should_draw_key;
					}
					_ => ()
				},
                _ => ()
            }
        }

		let frame_end = time::precise_time_ns(); // Including vsync

		let sleep_end = time::precise_time_ns();

		if sleep_end > next_print && frames > 0 {
			fps_message = String::from(format!("{:.2}ms per frame. Displayed at {:.2} fps.", frame_process_time as f64 / 1_000_000.0, 1_000_000_000.0 / frame_total_time as f64));
			frames = 0;
			frame_process_time = 0;
			frame_total_time = 0;
			next_print += 1_000_000_000;

			pred_message = conductor.dbg_second();
		}
		frame_process_time = (frame_process_time * frames + frame_end - frame_start) / (frames + 1);
		frame_total_time = (frame_total_time * frames + sleep_end - frame_start) / (frames + 1);
		frames += 1;
    }
}
