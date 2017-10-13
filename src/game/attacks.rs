use na;
use ::fixvec::{vec_from_angle, vec_angle, vec_cross_z};
use ::fixmath::{Fixed};
pub type Vec2 = na::Vec2<Fixed>;
pub type GameReal = Fixed;
use rustc_serialize::{Encoder, Decoder, Encodable};

use super::{ActionState, Hitcircle, HitcircleState, Weapon};
use resource::{Resource, Loadable, Cacheable};

use std::collections::HashMap;
use std::cell::RefCell;
impl Cacheable for Vec<ActionCommand> {
    fn load_for_cache(bytes: Vec<u8>) -> Vec<ActionCommand> {
        ::load_from_json_bytes(bytes).unwrap()
    }
}
implement_cache!(action_command_cache, Vec<ActionCommand>);

#[derive(Debug, RustcDecodable, RustcEncodable, PartialEq, Copy, Clone)]
pub enum FistStateData {
    Attack,
    Parry,
    Charge,
    Followup,
}
#[derive(Debug, RustcDecodable, RustcEncodable)]
pub enum ActionCommand {
    PushFist,
    PopFist,
    SetPosition(usize, Vec2),
    SetRadius(usize, GameReal),
    SetState(usize, FistStateData),
    SetDamage(usize, GameReal),
    SetEntityVelocity(Vec2),
    SetEntityFacing(GameReal),
    SpawnProjectile(usize, Vec2),
    EndFrame,
}
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct ActionInterpreter {
    orientation: Vec2,
    commands: Resource<Vec<ActionCommand>>,
	weapon: Resource<Weapon>,
	base_damage: GameReal,
    ipc: u32,
}
impl ActionInterpreter {
    pub fn new(facing: GameReal, commands: Resource<Vec<ActionCommand>>, weapon: &Resource<Weapon>, base_damage: GameReal) -> ActionInterpreter {
        ActionInterpreter {
            orientation: vec_from_angle(facing),
            commands: commands,
			weapon: weapon.clone(),
			base_damage: base_damage,
            ipc: 0,
        }
    }
	fn fist_position(&self, fist_index: usize, pos: Vec2) -> Vec2 {
		let cross_orientation = vec_cross_z(self.orientation);
		let p = pos + match self.weapon.fists.get(fist_index) {
			Some(f) => f.idle_position,
			None => Vec2::new(fixed!(0), fixed!(0)),
		};
		self.orientation * p.x + cross_orientation * p.y
	}
	fn fist_radius(&self, fist_index: usize, radius: GameReal) -> GameReal {
		radius * match self.weapon.fists.get(fist_index) {
			Some(f) => f.radius,
			None => fixed!(1),
		}
	}
    fn orient(&self, v: Vec2) -> Vec2 {
		let cross_orientation = vec_cross_z(self.orientation);
		self.orientation * v.x + cross_orientation * v.y
    }
}
pub struct ProjectileSpawn(pub Vec2, pub Vec2);
impl ActionInterpreter {
    pub fn mutate(&mut self, state: &mut ActionState, fists: &mut Vec<Hitcircle>, position: Vec2) -> Option<Option<ProjectileSpawn>> {
        use self::ActionCommand::*;
        let mut projectile = None;
        loop {
            match self.commands.get(self.ipc as usize) {
                Some(command) => match command {
                    &PushFist => {
						/* let new_index = fists.len();
                        fists.push(Fist {
                            state: FistState::Charge,
                            position: position + self.fist_position(new_index, Vec2::new(fixed!(0), fixed!(0))),
                            radius: self.fist_radius(new_index, fixed!(1)),
                        }); */
                    },
                    &PopFist => {
                        // fists.pop();
                    },
                    &SetPosition(i, pos) => {
						fists[i].position = position + self.fist_position(i, pos);
                    },
                    &SetRadius(i, radius) => {
						fists[i].radius = self.fist_radius(i, radius);
                    },
                    &SetState(i, ref s) => {
                        fists[i].state = match s {
                            &FistStateData::Attack => HitcircleState::Attack,
                            &FistStateData::Parry => HitcircleState::Parry,
                            &FistStateData::Charge => HitcircleState::Charge,
                            &FistStateData::Followup => HitcircleState::Followup,
                        };
                    },
                    &SetDamage(i, amount) => {
                        fists[i].damage = amount * self.base_damage;
                    },
                    &SetEntityVelocity(vel) => {
                        let cross_orientation = vec_cross_z(self.orientation);
                        state.velocity = self.orientation * vel.x + cross_orientation * vel.y;
                    },
                    &SetEntityFacing(facing) => {
                        state.facing = vec_angle(self.orientation) + facing;
                    },
                    &SpawnProjectile(i, vel) => {
                        projectile = Some(ProjectileSpawn(fists[i].position, self.orient(vel)));
                    },
                    &EndFrame => {
                        self.ipc += 1;
                        return Some(projectile);
                    }
                },
                None => { break; },
            }
            self.ipc += 1;
        }
        None
    }
}

// TODO: Instead of a single frame hit, maybe a hit does damage for every tick it overlaps an entity,
//       so that hits can do different amounts of damage depending on how well they were aimed, and
//       the attacks could have differing damage powers per frame.
//
//       Or maybe instead there could be differing powers per frame but in the end the target only
//       should have taken damage equal to the highest single-frame damage value that overlapped,
//       so that different attacks could still have different sweet spots without it being all about
//       keeping the attack overlapping as long as possible.
//
// TODO: On block when should an attack follow through vs bounce off?
