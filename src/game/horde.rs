use ::fixmath::Fixed;
use na;
pub type Vec2 = na::Vec2<Fixed>;
pub type GameReal = Fixed;

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub enum State {
    Countdown(u32),
    Active,
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct Horde {
    pub current_wave: usize,
    pub state: State,
}

fn spawn_entity(game: &mut super::GameState, ai: super::BasicAI) -> &mut super::Entity {
    let id = game.next_entity_id;
    game.next_entity_id += 1;
    let entity = super::spawn_entity(id);
    game.entities.push(entity);
    game.controllers.push(super::Controller::AI(ai));
    game.entities.iter_mut().last().unwrap()
}

pub fn spawn(wave: usize, game: &mut super::GameState) {
    match wave % 2 {
        0 => {
            let e = spawn_entity(game, super::BasicAI::new(1));
            e.position = Vec2::new(fixed!(0), fixed!(500));
        },
        _ => {
            for i in 0..10 {
                let mut ai = super::BasicAI::new(1);
                ai.strategy = super::AIStrategy::Circle;
                let e = spawn_entity(game, ai);
                e.position = Vec2::new(fixed!(-100 + i * 20), fixed!(500));
                e.stats.health_max = fixed!(50);
                e.stats.health = fixed!(50);
                e.stats.move_speed = fixed!(5);
                e.weapon = ::resource::Loadable::load("data/tiny.weapon.json");
                e.radius = fixed!(20);
            }
        }
    }
}

pub fn update(game: &mut super::GameState) {
    game.horde.state = match game.horde.state {
        State::Active => {
            let no_enemies = !game.controllers.iter().any(|x| x.get_owner().is_none());
            if no_enemies {
                State::Countdown(300)
            } else {
                State::Active
            }
        },
        State::Countdown(1) => {
            game.horde.current_wave += 1;
            spawn(game.horde.current_wave, game);
            State::Active
        },
        State::Countdown(t) => { State::Countdown(t - 1) },
    }
}