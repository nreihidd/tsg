use game::{self, Controller, Entity, EntityStats, EntityState, BasicAI, GameState};
use na::{Vec2};
use load_level::{LevelData, LevelLine, load_level_from_file};

pub fn get_level_lines(level: &LevelData) -> Vec<game::collision::Line> {
	level.collision_lines.iter().map(|line: &LevelLine| {
		game::collision::Line {
			point_a: line.a,
			point_b: line.b,
		 }
	}).collect()
}

pub fn make_initial_state() -> (GameState, LevelData) {
    let level: LevelData = load_level_from_file("assets/svg-test.svg");
	let game = GameState {
		lines: ::resource::Loadable::load("assets/svg-test.svg"),
		entities: vec![
			Entity {
				id: 1,
				position: Vec2::new(fixed!(300), fixed!(300)),
		        push_vel: Vec2::new(fixed!(0), fixed!(0)),
				radius: fixed!(40),
				stats: EntityStats {
					move_speed: fixed!(2),
					blocking_move_speed: fixed!(1),
					turn_rate: fixed!(1) / fixed!(20),

					stamina: fixed!(50),
					stamina_display: game::DisplayHold { state: game::DisplayHoldState::Hide },
					stamina_max: fixed!(50),
					stamina_regen: fixed!(1) / fixed!(2),
					blocking_stamina_regen: fixed!(1) / fixed!(10),

					health: fixed!(200),
					health_display: game::DisplayHold { state: game::DisplayHoldState::Hide },
					health_max: fixed!(200),

					poise: fixed!(40),
				},
				facing: fixed!(0),
				state: EntityState::Idle { desired_facing: fixed!(0), vel: Vec2::new(fixed!(0), fixed!(0)), blocking: false },
				weapon: ::resource::Loadable::load("data/test.weapon.json"),
		        fists: vec![], // TODO: If an action (attack) is taken before fists are initialized in update_fists, the attack interpreter will try to modify fists that don't exist
			},
			Entity {
				id: 2,
				position: Vec2::new(fixed!(600), fixed!(300)),
		        push_vel: Vec2::new(fixed!(0), fixed!(0)),
				radius: fixed!(40),
				stats: EntityStats {
					move_speed: fixed!(1),
					blocking_move_speed: fixed!(1),
					turn_rate: fixed!(1) / fixed!(20),

					stamina: fixed!(50),
					stamina_display: game::DisplayHold { state: game::DisplayHoldState::Hide },
					stamina_max: fixed!(50),
					stamina_regen: fixed!(1) / fixed!(2),
					blocking_stamina_regen: fixed!(1) / fixed!(10),

					health: fixed!(400),
					health_display: game::DisplayHold { state: game::DisplayHoldState::Hide },
					health_max: fixed!(400),

					poise: fixed!(30),
				},
				facing: fixed!(0),
				state: EntityState::Idle { desired_facing: fixed!(0), vel: Vec2::new(fixed!(0), fixed!(0)), blocking: false },
				weapon: ::resource::Loadable::load("data/thug1.weapon.json"),
		        fists: vec![],
			},
		],
		controllers: vec![
			Controller::AI(BasicAI::new(::rand::random::<u32>())),
			Controller::AI(BasicAI::new(::rand::random::<u32>())),
		],
		projectiles: vec![],
		next_entity_id: 3,
		next_projectile_id: 1,
		horde: game::horde::Horde { current_wave: 0, state: game::horde::State::Countdown(300) },
	};
    (game, level)
}