use time;
use sdl2;
use drawing::{self, translate, scale, color};
use glium::{self, Surface};
use glium::backend::Facade;
use na::{self, Vec2, Vec4, Mat4, Inv};
use game;
use fixvec::FromGame;
use std;
use num::traits::One;
use fixmath::Fixed;

trait ToGame<N> {
    fn to_game(self) -> N;
}
impl ToGame<Fixed> for f32 {
    fn to_game(self) -> Fixed {
        Fixed::from_f32(self)
    }
}
impl ToGame<Vec2<Fixed>> for Vec2<f32> {
    fn to_game(self) -> Vec2<Fixed> {
        Vec2::new(self.x.to_game(), self.y.to_game())
    }
}

fn draw_string(frame: &mut glium::Frame, view: &Mat4<f32>, text_renderer: &mut drawing::text::Text, x: f32, y: f32, color: Vec4<f32>, text: &str) {
    text_renderer.draw_string(&(*view * translate(x, y + 0.5, 0.0) * scale(4.0, 4.5, 1.0) * translate(1.0, 0.0, 0.0)), frame, color, text);
}

fn modal_file_path(default: Option<&std::path::Path>, display: &mut Display, filter: &str) -> Option<std::path::PathBuf> {
    use sdl2::event::Event;
    use sdl2::keyboard::Keycode;

    let current_dir = default.map(|p| { let mut p = p.to_path_buf(); p.pop(); p }).unwrap_or_else(|| std::path::PathBuf::from("data"));
    // std::iter::once(std::ffi::OsString::from("new")).chain( )
    let files: Vec<_> = current_dir.read_dir().unwrap().map(|f| f.unwrap().file_name()).filter(|f| f.to_str().unwrap().ends_with(filter)).collect();
    let mut selected_index = files.iter().position(|f| default.map(|p| p.file_name() == Some(f)).unwrap_or(false)).unwrap_or(0);

    loop {
        for event in display.event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => std::process::exit(0),
                Event::KeyDown { keycode: Some(keycode), keymod, .. } => match keycode {
                    Keycode::Escape => return None,
                    Keycode::Return => {
                        let mut p = current_dir.to_path_buf();
                        p.push(files[selected_index].clone());
                        return Some(p);
                    },
                    Keycode::Up => {
                        selected_index = if selected_index == 0 {
                            files.len() - 1
                        } else {
                            selected_index - 1
                        }
                    },
                    Keycode::Down => {
                        selected_index = if selected_index == files.len() - 1 {
                            0
                        } else {
                            selected_index + 1
                        }
                    },
                    _ => (),
                },
                _ => (),
            }
        }

        display.draw(|display, frame| {
            let (hud_width, hud_height) = { let (w, h) = frame.get_dimensions(); (w as f32, h as f32) };
        	let hud_view = na::OrthoMat3::<f32>::new(hud_width, hud_height, -50.0, 50.0).to_mat() * translate(-hud_width / 2.0, -hud_height / 2.0, 0.0);

            let mut y = hud_height - 15.0;
            for (i, file) in files.iter().enumerate() {
                let color = if i == selected_index { color(0xFF0000FF) } else { color(0x000000FF) };
                draw_string(frame, &hud_view, &mut display.text_renderer, 10.0, y, color, file.to_str().unwrap_or("?"));
                y -= 11.0;
            }
            draw_string(frame, &hud_view, &mut display.text_renderer, 10.0, 10.0, color(0x000000FF), "Choose a file...");
        });
    }
}

#[derive(Clone)]
struct ProjectileSpawnData {
    vel: Vec2<f32>,
}

struct FistAnimation {
    pos: Vec<Vec2<f32>>,
    state: Vec<game::attacks::FistStateData>,
    projectile_spawns: Vec<Option<ProjectileSpawnData>>,
}

struct AttackAnimation {
    origin_pos: Vec<Vec2<f32>>,
    fists: Vec<FistAnimation>,
}

#[derive(Clone)]
struct GenFist { pos: Vec2<f32>, state: game::attacks::FistStateData, spawn_projectile: Option<ProjectileSpawnData> }
#[derive(Clone)]
struct GenState { origin: Vec2<f32>, fists: Vec<GenFist> }

fn load_animation(path: &std::path::Path) -> AttackAnimation {
    use game::attacks::ActionCommand::*;
    use game::attacks::ActionCommand;
    use game::attacks::FistStateData;

    let attack: Vec<ActionCommand> = ::load_from_json_file(path).unwrap();
    let mut state = GenState { origin: Vec2::new(0.0, 0.0), fists: vec![GenFist { pos: Vec2::new(0.0, 0.0), state: FistStateData::Charge, spawn_projectile: None }] };
    let mut anim_states = vec![];
    let mut origin_velocity = Vec2::new(0.0, 0.0);
    for command in attack {
        match command {
            PushFist => (),
            PopFist => (),
            SetPosition(i, p) => { state.fists[i].pos = p.from_game(); },
            SetState(i, s) => { state.fists[i].state = s; },
            SetEntityVelocity(v) => { origin_velocity = v.from_game(); },
            SpawnProjectile(i, vel) => { state.fists[i].spawn_projectile = Some(ProjectileSpawnData { vel: vel.from_game() }); },
            EndFrame => {
                state.origin = state.origin + origin_velocity;
                anim_states.push(state.clone());
                for fist in state.fists.iter_mut() { fist.spawn_projectile = None; }
            },
            _ => (),
        }
    }
    let mut anim = AttackAnimation { origin_pos: vec![], fists: vec![FistAnimation { pos: vec![], state: vec![], projectile_spawns: vec![] }] };
    for state in anim_states {
        anim.origin_pos.push(state.origin);
        for (i, fist) in state.fists.into_iter().enumerate() {
            anim.fists[i].pos.push(fist.pos);
            anim.fists[i].state.push(fist.state);
            anim.fists[i].projectile_spawns.push(fist.spawn_projectile);
        }
    }
    anim
}

fn save_animation(path: &std::path::Path, anim: &AttackAnimation) {
    use game::attacks::ActionCommand::*;
    use game::attacks::FistStateData;
    let mut actions = vec![];
    let mut state = GenState { origin: anim.origin_pos[0], fists: vec![GenFist { pos: Vec2::new(0.0, 0.0), state: FistStateData::Charge, spawn_projectile: None }] };
    let mut velocity = Vec2::new(0.0, 0.0);
    macro_rules! assign_if_different {
        ($a:expr, $b:expr) => {{
            let a = $a;
            let b = $b;
            if a != b {
                $a = b;
                Some(b)
            } else {
                None
            }
        }}
    };
    for frame in 0..anim.origin_pos.len() {
        let vel = anim.origin_pos[frame] - state.origin;
        assign_if_different!(velocity, vel).map(|v| actions.push(SetEntityVelocity(v.to_game())));
        state.origin = anim.origin_pos[frame];
        for (i, fist) in anim.fists.iter().enumerate() {
            assign_if_different!(state.fists[i].pos, fist.pos[frame]).map(|v| actions.push(SetPosition(i, v.to_game())));
            assign_if_different!(state.fists[i].state, fist.state[frame]).map(|v| {
                actions.push(SetState(i, v));
                match v {
                    FistStateData::Attack => {
                        actions.push(SetDamage(i, fixed!(1)));
                    },
                    _ => (),
                }
            });
            if let Some(ProjectileSpawnData { vel }) = fist.projectile_spawns[frame] { actions.push(SpawnProjectile(i, vel.to_game())); }
        }
        actions.push(EndFrame);
    }
    ::dbg_write_json_file(path, &actions);
}

const FRAME_DURATION: u64 = 16_666_667;

struct Display {
    context: sdl2::Sdl,
    display: drawing::glium_sdl2::SDL2Facade,
    text_renderer: drawing::text::Text,
    model_renderer: drawing::model::ModelRenderer,
    event_pump: sdl2::EventPump,
    last_flip: u64,
}

impl Display {
    fn draw<F>(self: &mut Display, f: F)
        where F: FnOnce(&mut Display, &mut glium::Frame)
    {
    	let mut frame = self.display.draw();
        frame.clear_color(1.0, 1.0, 1.0, 1.0);
    	frame.clear_depth(1.0);

        f(self, &mut frame);

        frame.finish().unwrap();

    	// Sleep here
    	let predicted_vblank = self.last_flip + FRAME_DURATION;
    	loop {
    		let now = time::precise_time_ns();
    		if now < predicted_vblank {
    			let time_until_vblank = predicted_vblank - now;
    			if time_until_vblank > 3_000_000 {
    				std::thread::sleep(std::time::Duration::new(0, 1_000_000));
    			} else {
    				break;
    			}
    		} else {
    			break;
    		}
    	}

    	glium::backend::Facade::get_context(&self.display).finish(); // Will vsync depending on "Sync to VBlank" in Nvidia X Server Settings
    	self.last_flip = time::precise_time_ns();
    }
}

fn make_display() -> Display {
    use drawing::glium_sdl2::DisplayBuild;

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

	video_subsystem.gl_attr().set_multisample_buffers(1);
	video_subsystem.gl_attr().set_multisample_samples(8);
	video_subsystem.gl_attr().set_stencil_size(8);
	video_subsystem.gl_attr().set_double_buffer(true);
    let mut display = video_subsystem.window("TSG Editor", 800, 600).resizable().build_glium().unwrap();
	let display_mode = video_subsystem.desktop_display_mode(0).unwrap();
	display.window_mut().set_display_mode(Some(display_mode)).unwrap();
	let ctx = display.get_context().clone();

    Display {
        display: display,
        text_renderer: drawing::text::Text::new(&ctx),
        model_renderer: drawing::model::ModelRenderer::new(&ctx),
        event_pump: sdl_context.event_pump().unwrap(),
        last_flip: time::precise_time_ns(),
        context: sdl_context,
    }
}

fn fist_offset(weapon: &game::Weapon, i: usize) -> Vec2<f32> {
    weapon.fists.get(i).map(|f| f.idle_position.from_game()).unwrap_or_else(|| Vec2::new(0.0, 0.0))
}

fn set_vec_length<T: Clone>(v: &mut Vec<T>, l: usize) {
    while v.len() > l {
        v.pop();
    }
    if v.len() < l {
        let to_dup = v[v.len() - 1].clone();
        while v.len() < l {
            v.push(to_dup.clone());
        }
    }
}

fn set_anim_length(anim: &mut AttackAnimation, l: usize) {
    assert!(l > 0);
    set_vec_length(&mut anim.origin_pos, l);
    for fist in anim.fists.iter_mut() {
        set_vec_length(&mut fist.pos, l);
        set_vec_length(&mut fist.state, l);
        set_vec_length(&mut fist.projectile_spawns, l);
    }
}

pub fn main_editor() {
    let mut display = make_display();

    let mut running = true;

    let mut animation = AttackAnimation {
        origin_pos: vec![Vec2::new(0.0, 0.0); 100],
        fists: vec![FistAnimation {
            pos: vec![Vec2::new(0.0, 0.0); 100],
            state: vec![game::attacks::FistStateData::Charge; 100],
            projectile_spawns: vec![None; 100]
        }],
    };

    enum RecordingState {
        None,
        Origin,
        Fist,
        State,
    }

    let mut recording_state = RecordingState::None;
    let mut mouse_pos = Vec2::new(0.0, 0.0);
    let mut kbd_state = game::attacks::FistStateData::Charge;
    let mut current_frame: usize = 0;
    let mut current_file: Option<std::path::PathBuf> = None;
    let mut weapon_file = std::path::Path::new("data/test.weapon.json").to_path_buf();
    let mut weapon: game::Weapon = ::load_from_json_file(&weapon_file).unwrap();

    macro_rules! open_dialog {
        ($filter:expr) => {
            modal_file_path(current_file.as_ref().map(|p| p.as_path()), &mut display, $filter).map(|f| { current_file = Some(f.clone()); f })
        }
    }

    let mut window_to_world = Mat4::<f32>::one();

    while running {
        loop {
            use sdl2::event::Event;
			use sdl2::keyboard::Keycode;
            use sdl2::mouse::MouseButton;

            let event = match display.event_pump.poll_event() { Some(e) => e, None => break };

            fn ctrl_key_held(keymod: &sdl2::keyboard::Mod) -> bool {
                keymod.contains(sdl2::keyboard::LCTRLMOD) || keymod.contains(sdl2::keyboard::RCTRLMOD)
            }
            fn shift_key_held(keymod: &sdl2::keyboard::Mod) -> bool {
                keymod.contains(sdl2::keyboard::LSHIFTMOD) || keymod.contains(sdl2::keyboard::RSHIFTMOD)
            }
            match event {
                Event::Quit { .. } => running = false,
				Event::KeyDown { keycode: Some(keycode), keymod, .. } => match keycode {
					Keycode::Escape => running = false,
                    Keycode::Return => {
                        let start = animation.fists[0].pos[0];
                        for (fist_pos, origin) in animation.fists[0].pos.iter().zip(animation.origin_pos.iter_mut()) {
                            *origin = *origin + (*fist_pos - start) / 10.0;
                        }
                    },
                    Keycode::F => {
                        let start = animation.fists[0].pos[0];
                        for (i, fist_pos) in animation.fists[0].pos.iter_mut().rev().take(20).enumerate() {
                            let t = i as f32 / 20.0;
                            *fist_pos = start * (1.0 - t) + *fist_pos * t;
                        }
                    },
                    Keycode::Num1 => {
                        current_frame = 0;
                        recording_state = RecordingState::State;
                        kbd_state = game::attacks::FistStateData::Charge;
                    },
                    Keycode::C => {
                        for fist in animation.fists.iter_mut() {
                            for b in fist.projectile_spawns.iter_mut() {
                                *b = None;
                            }
                        }
                    },
                    Keycode::P => {
                        let f = current_frame.saturating_sub(1);
                        let pa = animation.fists[0].pos[f] + animation.origin_pos[f];
                        let pb = animation.fists[0].pos[current_frame] + animation.origin_pos[current_frame];
                        animation.fists[0].projectile_spawns[current_frame] = Some(ProjectileSpawnData { vel: pb - pa });
                    },
                    Keycode::Num2 => kbd_state = game::attacks::FistStateData::Attack,
                    Keycode::Num3 => kbd_state = game::attacks::FistStateData::Parry,
                    Keycode::Num4 => kbd_state = game::attacks::FistStateData::Followup,
                    Keycode::S if ctrl_key_held(&keymod) && shift_key_held(&keymod) => {
                        if let Some(f) = open_dialog!(".attack.json") {
                            save_animation(&f, &animation);
                        }
                    },
                    Keycode::S if ctrl_key_held(&keymod) => {
                        if current_file.is_none() {
                            if let Some(f) = open_dialog!(".attack.json") {
                                save_animation(&f, &animation);
                            }
                        } else {
                            if let Some(ref f) = current_file {
                                save_animation(f, &animation);
                            }
                        }
                    },
                    Keycode::O if ctrl_key_held(&keymod) => {
                        if let Some(f) = open_dialog!(".attack.json") {
                            animation = load_animation(&f);
                            current_frame = 0;
                            recording_state = RecordingState::None;
                        }
                    },
                    Keycode::W => {
                        if let Some(f) = modal_file_path(Some(&weapon_file), &mut display, ".weapon.json") {
                            weapon_file = f;
                            weapon = ::load_from_json_file(&weapon_file).unwrap();
                        }
                    },
                    Keycode::RightBracket => {
                        let l = animation.origin_pos.len();
                        set_anim_length(&mut animation, l + 5);
                    },
                    Keycode::LeftBracket => {
                        let l = animation.origin_pos.len();
                        if l > 5 {
                            set_anim_length(&mut animation, l - 5);
                            current_frame = std::cmp::min(animation.origin_pos.len() - 1, current_frame);
                        }
                    },
					_ => ()
				},
                Event::MouseMotion { x, y, .. } => {
                    let mouse_world_pos = window_to_world * Vec4::new(x as f32, y as f32, 0.0, 1.0);
                    mouse_pos = Vec2::new(mouse_world_pos.x, mouse_world_pos.y);
                },
                Event::MouseButtonDown { mouse_btn: MouseButton::Left, .. } => {
                    current_frame = 0;
                    recording_state = RecordingState::Fist;
                    let starting_pos = animation.origin_pos[0] + fist_offset(&weapon, 0);
                    let mouse_window = window_to_world.inv().unwrap() * Vec4::new(starting_pos.x, starting_pos.y, 0.0, 1.0);
                    display.context.mouse().warp_mouse_in_window(display.display.window(), mouse_window.x as i32, mouse_window.y as i32);
                },
                Event::MouseButtonDown { mouse_btn: MouseButton::Middle, .. } => {
                    current_frame = 0;
                    recording_state = RecordingState::Origin;
                },
                _ => ()
            }
        }

        match recording_state {
            RecordingState::Fist => {
                let i = 0;
                animation.fists[i].pos[current_frame] = mouse_pos - animation.origin_pos[current_frame] - fist_offset(&weapon, i);
            },
            RecordingState::Origin => {
                animation.origin_pos[current_frame] = mouse_pos;
            },
            RecordingState::State => {
                animation.fists[0].state[current_frame] = kbd_state.clone();
            },
            RecordingState::None => (),
        }

        display.draw(|display, frame| {
        	let (hud_width, hud_height) = { let (w, h) = frame.get_dimensions(); (w as f32, h as f32) };
        	let hud_view = na::OrthoMat3::<f32>::new(hud_width, hud_height, -50.0, 50.0).to_mat() * translate(-hud_width / 2.0, -hud_height / 2.0, 0.0);

            draw_string(frame, &hud_view, &mut display.text_renderer, 10.0, 10.0, color(0x000000FF), &format!("{:?}", weapon_file));
            draw_string(frame, &hud_view, &mut display.text_renderer, 10.0, 21.0, color(0x000000FF), &format!("{:?}", current_file));
            draw_string(frame, &hud_view, &mut display.text_renderer, 10.0, 32.0, color(0x000000FF), &format!("{:3} / {:3}", current_frame, animation.origin_pos.len()));

            let view = na::OrthoMat3::<f32>::new(hud_width, hud_height, -50.0, 50.0).to_mat();
            let origin = animation.origin_pos[current_frame];
            display.model_renderer.draw_model_at(frame, &view, &display.model_renderer.circle, origin, 20.0, 20.0, color(0x0000FFFF));
            for (i, fist) in animation.fists.iter().enumerate() {
                use game::attacks::FistStateData::*;
                let fist_pos = origin + fist.pos[current_frame] + fist_offset(&weapon, i);
                let c = match fist.state[current_frame] {
                    Charge => color(0xFFAA00FF),
                    Attack => color(0xFF0000FF),
                    Parry => color(0x0000FFFF),
                    Followup => color(0x999999FF),
                };
                display.model_renderer.draw_model_at(frame, &view, &display.model_renderer.circle, fist_pos, 10.0, 10.0, c);
            }
            let anim_length = animation.origin_pos.len() as f32;
            for (i, projectile_spawn) in animation.fists.iter().flat_map(|f| f.projectile_spawns.iter().enumerate()) {
                if projectile_spawn.is_some() {
                    let x = i as f32 / anim_length * hud_width;
                    display.model_renderer.draw_model_at(frame, &hud_view, &display.model_renderer.rect, Vec2::new(x, 60.0), 5.0, 5.0, color(0x00FF00FF));
                }
            }

            let percentage_done = current_frame as f32 / anim_length;
            let progress_width = percentage_done * hud_width;
            display.model_renderer.draw_model_at(frame, &hud_view, &display.model_renderer.rect, Vec2::new(progress_width, 50.0), 10.0, 10.0, color(0x000000FF));
            window_to_world = view.inv().unwrap() * (hud_view * scale(1.0, -1.0, 1.0) * translate(0.0, -hud_height, 0.0));
        });

        current_frame += 1;
        if current_frame >= animation.origin_pos.len() {
            current_frame = 0;
            recording_state = RecordingState::None;
        }
    }
}