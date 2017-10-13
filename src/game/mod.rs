pub mod collision;
pub mod attacks;
pub mod initial;
pub mod horde;

use self::attacks::{ActionInterpreter};

use network;
use std::collections::HashMap;
use std::collections::BTreeMap;
use std::cmp::{max};
use na;
use rand::{Rng, XorShiftRng, SeedableRng};
use ::gamerender::presentation;
use ::Presentation;
use ::fixvec::{Norm, CmpNorm, vec_from_angle, vec_angle, vec_cross_z};
use ::fixmath::{Fixed, PI, min_angle_diff, normalized_angle};
pub type Vec2 = na::Vec2<Fixed>;
pub type GameReal = Fixed;
use rustc_serialize::{Encodable};

pub type CommitEffect = Box<Fn(&mut Presentation)>;

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct GameState {
    pub lines: Resource<Vec<collision::Line>>,
    pub entities: Vec<Entity>,
    pub projectiles: Vec<Projectile>,
    pub controllers: Vec<Controller>,
    pub next_projectile_id: u32,
    pub next_entity_id: u32,
    pub horde: horde::Horde,
}

use std::cell::RefCell;
use resource::{Resource, Loadable, Cacheable};
impl Cacheable for Vec<collision::Line> {
    fn load_for_cache(bytes: Vec<u8>) -> Vec<collision::Line> {
        initial::get_level_lines(&::load_level::load_level(bytes))
    }
}
implement_cache!(level_cache, Vec<collision::Line>);
impl Cacheable for Weapon {
    fn load_for_cache(bytes: Vec<u8>) -> Weapon {
        ::load_from_json_bytes(bytes).unwrap()
    }
}
implement_cache!(weapon_cache, Weapon);

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub enum Controller {
    Player(PlayerController),
    AI(BasicAI),
}
impl Controller {
    fn control(&mut self, entity: &Entity, entities: &Vec<Entity>, inputs: &BTreeMap<u64, Inputs>) -> Command {
        match self {
            &mut Controller::Player(ref mut c) => c.control(entity, entities, inputs),
            &mut Controller::AI(ref mut c) => c.control(entity, entities, inputs),
        }
    }
    pub fn get_owner(&self) -> Option<usize> {
        match self {
            &Controller::Player(ref c) => Some(c.player_id),
            &Controller::AI(_) => None,
        }
    }
}

enum Command {
    Idle { facing: GameReal, dir: Vec2, block: bool },
    Roll { dir: Vec2 },
    Action { name: &'static str }, // !!! Cannot use this as part of the game state.
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct PlayerController {
    player_id: usize,
    desired_facing: GameReal,
    last_inputs: Inputs,
}
impl PlayerController {
    pub fn new(player_id: usize) -> PlayerController {
        PlayerController {
            player_id: player_id,
            desired_facing: fixed!(0),
            last_inputs: Inputs {
            	movement: Vec2::new(fixed!(0), fixed!(0)),
            	facing: Vec2::new(fixed!(0), fixed!(0)),
            	roll: false,
            	attack: false,
            	attack_dumb: false,
                attack_special: false,
            	parry: false,
            	block: false,
                dbg_buttons: vec![],
            },
        }
    }
}
impl PlayerController {
    fn control(&mut self, entity: &Entity, _entities: &Vec<Entity>, inputs: &BTreeMap<u64, Inputs>) -> Command {
        inputs.get(&(self.player_id as u64)).map(|i| self.last_inputs = i.clone());
        let ref inputs = self.last_inputs;
    	if inputs.facing.norm() > fixed!(3) / fixed!(10) {
    		self.desired_facing = vec_angle(inputs.facing);
    	}
        let movement = Command::Idle { facing: self.desired_facing, dir: inputs.movement, block: inputs.block };
        if entity.stats.stamina > fixed!(0) {
			if inputs.roll {
                Command::Roll { dir: inputs.movement.normalize() }
			} else if let Some(name) =
                if inputs.attack { Some("light") }
                else if inputs.attack_dumb { Some("heavy") }
                else if inputs.parry { Some("parry") }
                else if inputs.attack_special { Some("special") }
                else { None }
            {
                if entity.weapon.attacks.contains_key(name) {
                    Command::Action { name: name }
                } else {
                    movement
                }
			} else {
				movement
			}
		} else {
			movement
		}
    }
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
enum AIPlan {
    PsycheThemOut { stare_angle: GameReal },
    Mobilize { desired_facing: GameReal, vel: Vec2 },
    Attack,
}
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub enum AIStrategy {
    Basic,
    Circle,
}
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct BasicAI {
    pub target: Option<u32>,
    pending: Vec<AIPlan>,
    pub strategy: AIStrategy,
    seed: u32,
}
impl BasicAI {
    pub fn new(seed: u32) -> BasicAI {
        BasicAI {
            target: None,
            pending: vec![],
            strategy: AIStrategy::Basic,
            seed: seed,
        }
    }
}
impl BasicAI {
    fn control(&mut self, entity: &Entity, entities: &Vec<Entity>, _inputs: &BTreeMap<u64, Inputs>) -> Command {
        use self::AIPlan::*;
        // Get a new target if necessary
        let target = self.target
            .and_then(|id| entities.iter().find(move |x| x.id == id && (x.position - entity.position).norm() < fixed!(1000)))
            .or_else(|| {
                let e = entities.iter().find(|x| x.id != entity.id && (x.position - entity.position).norm() < fixed!(1000));
                self.target = e.map(|e| e.id);
                e
            });
        let plan = if let Some(target) = target {
            match self.strategy {
                AIStrategy::Basic => {
                    let dir = target.position - entity.position;
                    let facing = vec_angle(dir.normalize());
                    if dir.norm() > fixed!(150) {
                        Mobilize { desired_facing: facing, vel: dir.normalize() }
                	} else if dir.norm() < fixed!(60) {
                        Mobilize { desired_facing: facing, vel: -dir.normalize() }
                	} else {
                        let mut rng = XorShiftRng::from_seed([self.seed, 0, 0, 0]);
                        self.seed = rng.next_u32();
                        if rng.gen_weighted_bool(120) {
                            Attack
                        } else {
                            PsycheThemOut { stare_angle: facing }
                        }
                	}
                },
                AIStrategy::Circle => {
                    let dir = target.position - entity.position;
                    let facing = vec_angle(dir.normalize());
                    if dir.norm() > fixed!(200) {
                        Mobilize { desired_facing: facing, vel: dir.normalize() }
                	} else if dir.norm() < fixed!(60) {
                        Mobilize { desired_facing: facing, vel: -dir.normalize() }
                	} else {
                        let mut rng = XorShiftRng::from_seed([self.seed, 0, 0, 0]);
                        self.seed = rng.next_u32();
                        if rng.gen_weighted_bool(120) {
                            Attack
                        } else {
                            Mobilize { desired_facing: facing, vel: vec_cross_z(dir.normalize()) }
                        }
                	}
                }
            }
        } else {
            PsycheThemOut { stare_angle: self.pending.iter().rev().flat_map(|x| match x {
                &PsycheThemOut { stare_angle } => Some(stare_angle),
                &Attack => None,
                &Mobilize { desired_facing, vel } => Some(desired_facing),
            }).next().unwrap_or(entity.facing) }
        };
        self.pending.insert(0, plan);

        let plan_to_execute = if self.pending.len() > 15 {
            self.pending.pop().unwrap()
        } else {
            PsycheThemOut { stare_angle: fixed!(0) }
        };
        match plan_to_execute {
            PsycheThemOut { stare_angle } => Command::Idle { facing: stare_angle, dir: Vec2::new(fixed!(0), fixed!(0)), block: false },
            Attack => Command::Action { name: "light" },
            Mobilize { desired_facing, vel } => Command::Idle { facing: desired_facing, dir: vel, block: false },
        }
    }
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct DisplayHold {
    pub state: DisplayHoldState,
}
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub enum DisplayHoldState {
    Hold { display: GameReal, frames: u32 },
    Drain { display: GameReal, rate: GameReal },
    Hide,
}
impl DisplayHold {
    pub fn hold(&mut self, at_least: GameReal) {
        use self::DisplayHoldState::*;
        self.state = DisplayHoldState::Hold {
            display: match self.state {
                Hold { display, .. } => display.max_(at_least),
                Drain { .. } => at_least,
                Hide => at_least,
            },
            frames: 30,
        };
    }
    pub fn update(&mut self, target: GameReal) {
        use self::DisplayHoldState::*;
        self.state = match self.state {
            Hold { display, frames } => {
                if display <= target {
                    Hide
                } else {
                    let nframes = frames - 1;
                    if nframes == 0 {
                        Drain { display: display, rate: ((target - display) / fixed!(20)).min_(fixed!(-1)) }
                    } else {
                        Hold { display: display, frames: nframes }
                    }
                }
            },
            Drain { display, rate } => {
                let ndisplay = display + rate;
                if ndisplay <= target || ndisplay <= fixed!(0) {
                    Hide
                } else {
                    Drain { display: ndisplay, rate: rate }
                }
            },
            Hide => Hide,
        };
    }
    pub fn query(&self) -> Option<GameReal> {
        use self::DisplayHoldState::*;
        match self.state {
            Hold { display, .. } => Some(display),
            Drain { display, .. } => Some(display),
            Hide => None,
        }
    }
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct EntityStats {
	pub move_speed: GameReal,
	pub blocking_move_speed: GameReal,
	pub turn_rate: GameReal,

	pub stamina: GameReal,
	pub stamina_max: GameReal,
    pub stamina_display: DisplayHold,
	pub stamina_regen: GameReal,
	pub blocking_stamina_regen: GameReal,

	pub health: GameReal,
    pub health_display: DisplayHold,
	pub health_max: GameReal,

    pub poise: GameReal,
}

#[derive(Debug, RustcDecodable, RustcEncodable)]
pub struct Attack {
    pub stamina_cost: GameReal,
    pub base_damage: GameReal,
    pub attack_anim: String,
}
#[derive(Debug, RustcDecodable, RustcEncodable)]
pub struct WeaponFist {
    pub idle_position: Vec2,
    pub radius: GameReal,
    pub attach_angle: Option<GameReal>,
}
#[derive(Debug, RustcDecodable, RustcEncodable)]
pub struct Weapon {
    pub fists: Vec<WeaponFist>,
    pub attacks: HashMap<String, Attack>,
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct Entity {
	pub id: u32,
	pub position: Vec2,
    pub push_vel: Vec2,
	pub radius: GameReal,

	pub stats: EntityStats,

	pub facing: GameReal,
	pub state: EntityState,
    pub weapon: Resource<Weapon>,
    pub fists: Vec<Hitcircle>,
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable, PartialEq, Eq)]
pub enum HitcircleState {
    Attack,
    Parry,
    Charge,
    Followup,
}
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct Hitcircle {
    pub position: Vec2,
    pub radius: GameReal,
    pub damage: GameReal,
    pub hits: Vec<u32>,
    pub state: HitcircleState,
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub enum ProjectileType {
    Normal,
    Piercing,
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct Projectile {
    pub id: u32,
    pub velocity: Vec2,
    pub owning_entity_id: u32,
    pub ttl: u32,
    pub kind: ProjectileType,
    pub hitcircle: Hitcircle,
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub enum EntityState {
	Idle { vel: Vec2, desired_facing: GameReal, blocking: bool },
	Roll { ttl: u32, dir: Vec2 },
    Action { mutator: attacks::ActionInterpreter, state: ActionState },
	Staggered { ttl: u32 },
}
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct ActionState {
    pub velocity: Vec2,
    pub facing: GameReal,
}

pub const BLOCK_SPREAD: GameReal = PI;

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct Inputs {
	pub movement: Vec2,
	pub facing: Vec2,
	pub roll: bool,
	pub attack: bool,
	pub attack_dumb: bool,
    pub attack_special: bool,
	pub parry: bool,
	pub block: bool,
    pub dbg_buttons: Vec<bool>,
}


impl Entity {
	fn is_idle(&self) -> bool {
		match self.state {
			EntityState::Idle { .. } => true,
			_ => false,
		}
	}
}

pub fn idle_fists(origin: Vec2, facing: GameReal, weapon: &Weapon) -> Vec<Hitcircle> {
    let orientation = vec_from_angle(facing);
    let cross_orientation = vec_cross_z(orientation);
    let oriented = |p: Vec2| origin + orientation * p.x + cross_orientation * p.y;
    weapon.fists.iter().map(|f| Hitcircle {
        state: HitcircleState::Followup,
        position: oriented(f.idle_position),
        radius: f.radius,
        damage: fixed!(0),
        hits: vec![],
    }).collect()
}

fn update_fists(entity: &mut Entity) {
    match entity.state {
        EntityState::Staggered { ttl, .. } => {
            entity.fists = entity.fists.iter().zip(idle_fists(entity.position, entity.facing, &entity.weapon).into_iter()).map(|(current_fist, mut idle_fist)| {
                let d = idle_fist.position - current_fist.position;
                let t = d / fixed!(ttl as i32);
                idle_fist.position = current_fist.position + t;
                idle_fist
            }).collect();
        },
        EntityState::Idle { .. } | EntityState::Roll { .. } => {
            entity.fists = idle_fists(entity.position, entity.facing, &entity.weapon);
        },
        EntityState::Action { .. } => {
            // Do nothing
        }
    }
}

fn advance_entity_state(entity: &mut Entity) -> Option<Projectile> {
    let mut projectile = None;
	if match entity.state {
		EntityState::Roll { ref mut ttl, .. } => {
			*ttl -= 1;
			*ttl == 0
		},
		EntityState::Staggered { ref mut ttl, .. } => {
			*ttl -= 1;
			*ttl == 0
		},
        EntityState::Action { ref mut mutator, ref mut state } => {
            let proj = mutator.mutate(state, &mut entity.fists, entity.position);
            if let Some(Some(attacks::ProjectileSpawn(pos, vel))) = proj {
                projectile = Some(Projectile {
                    id: 0,
                    hitcircle: Hitcircle {
                        position: pos,
                        radius: fixed!(10),
                        damage: fixed!(1),
                        hits: vec![],
                        state: HitcircleState::Attack,
                    },
                    velocity: vel,
                    owning_entity_id: entity.id,
                    ttl: 120,
                    kind: ProjectileType::Normal,
                });
            }
            proj.is_none()
        },
		_ => false
	} {
		entity.state = EntityState::Idle { desired_facing: entity.facing, vel: Vec2::new(fixed!(0), fixed!(0)), blocking: false };
	}
    projectile
}

fn apply_entity_input(entity: &mut Entity, command: Command) -> Vec<CommitEffect> {
    let mut effects: Vec<CommitEffect> = vec![];
    if entity.is_idle() {
        match command {
            Command::Idle { facing, dir, block } => {
                entity.state = EntityState::Idle { desired_facing: facing, vel: dir, blocking: block }
            },
            Command::Roll { dir } => {
                if entity.stats.stamina > fixed!(0) {
                    entity.stats.stamina_display.hold(entity.stats.stamina);
                    entity.stats.stamina = entity.stats.stamina - fixed!(25);
                    entity.state = EntityState::Roll { ttl: 25, dir: dir };
                    let entity_position = entity.position;
                    effects.push(Box::new(move |presentation| {
                        use fixvec::FromGame;
                        presentation.play_sound("assets/roll.ogg", entity_position.from_game());
                    }));
                }
            },
            Command::Action { name } => {
                if entity.stats.stamina > fixed!(0) {
                    entity.stats.stamina_display.hold(entity.stats.stamina);
                    let attack = entity.weapon.attacks.get(name).unwrap();
                    entity.stats.stamina = entity.stats.stamina - attack.stamina_cost;
                    let mut mutator = ActionInterpreter::new(entity.facing, Loadable::load(&attack.attack_anim), &entity.weapon, attack.base_damage);
                    let mut state = ActionState {
                        velocity: Vec2::new(fixed!(0), fixed!(0)),
                        facing: entity.facing,
                    };
                    // TODO: Update blender animations and export script so this isn't necessary:
                    for fist in entity.fists.iter_mut() {
                        fist.state = HitcircleState::Charge;
                        fist.hits = vec![];
                    }
                    let entity_position = entity.position;
                    effects.push(Box::new(move |presentation| {
                        use fixvec::FromGame;
                        presentation.play_sound("assets/whiff.ogg", entity_position.from_game());
                    }));
                    mutator.mutate(&mut state, &mut entity.fists, entity.position);
                    entity.state = EntityState::Action {
                        mutator: mutator,
                        state: state,
                    };
                }
            },
        }
    }
    match entity.state {
        EntityState::Idle { ref mut blocking, .. } => {
            *blocking = *blocking && entity.stats.stamina > fixed!(0);
        },
        _ => { },
    }
    effects
}

fn movement_pass(v: &mut Vec<Entity>) {
	for entity in v.iter_mut() {
		match entity.state {
			EntityState::Idle { vel, blocking, desired_facing } => {
				let turn_rate = entity.stats.turn_rate;
				let rotate = min_angle_diff(desired_facing, entity.facing).max_(-turn_rate).min_(turn_rate);
				entity.facing = normalized_angle(entity.facing + rotate);

				if entity.stats.stamina < fixed!(0) {
					let stamina_normal = entity.stats.stamina_regen;
					let stamina_slow = stamina_normal * (fixed!(1) / fixed!(2));
					entity.stats.stamina = entity.stats.stamina + stamina_slow;
					if entity.stats.stamina > fixed!(0) {
						entity.stats.stamina = entity.stats.stamina * stamina_normal / stamina_slow;
					}
				} else {
					entity.stats.stamina = entity.stats.stamina + if blocking {
						entity.stats.blocking_stamina_regen
					} else {
						entity.stats.stamina_regen
					};
				}
				entity.stats.stamina = entity.stats.stamina.min_(entity.stats.stamina_max);
			},
            EntityState::Action { ref state, .. } => {
                entity.facing = normalized_angle(state.facing);
            },
			_ => { },
		}
        let friction_mag = fixed!(3) / fixed!(10);
        let norm = CmpNorm(entity.push_vel);
        if norm < friction_mag {
            entity.push_vel = Vec2::new(fixed!(0), fixed!(0));
        } else {
            let v = entity.push_vel.normalize();
            entity.push_vel = entity.push_vel - v * friction_mag;
        }
	}
}

fn get_velocity(entity: &Entity) -> Vec2 {
    let base = match entity.state {
        EntityState::Idle { vel, blocking, .. } => {
            vel * if !blocking {
                entity.stats.move_speed
            } else {
                entity.stats.blocking_move_speed
            }
        },
        EntityState::Roll { ttl, dir } => {
            dir * fixed!(10) * if ttl > 10 {
                fixed!(1)
            } else {
                fixed!(ttl) / fixed!(10)
            }
        },
        EntityState::Action { ref state, .. } => {
            state.velocity
        },
        _ => Vec2::new(fixed!(0), fixed!(0)),
    };
    base + entity.push_vel
}

fn circle_overlap(c1: Vec2, r1: GameReal, c2: Vec2, r2: GameReal) -> bool {
	let d = c2 - c1;
	let r = r1 + r2;
    d.norm() <= r
}

#[derive(Debug, Clone)]
struct HitcircleOwner {
    entity_id: u32,
    source: HitcircleSource,
}
#[derive(Debug, Clone)]
enum HitcircleSource {
    Projectile { projectile_id: u32 },
    Fist { fist_index: usize },
}

struct ApplyCombat(Box<Fn(&mut [Entity], &mut Vec<Projectile>, &mut Vec<CommitEffect>)>);

fn combat_pass(hitcircles: &[(HitcircleOwner, &Hitcircle)], v: &[Entity]) -> Vec<ApplyCombat> {
	let mut results: Vec<ApplyCombat> = vec![];

	for &(ref offending_owner, offender) in hitcircles.iter() {
        if offender.state != HitcircleState::Attack { continue; }

        // First check for a parry
        let mut parried = false;
        for &(ref defending_owner, defender) in hitcircles.iter() {
            if offending_owner.entity_id == defending_owner.entity_id { continue; }
            if defender.state == HitcircleState::Parry {
                if circle_overlap(offender.position, offender.radius, defender.position, defender.radius) {
                    if !parried {
                        let dir = (offender.position - defender.position).normalize();
                        let offender_position = offender.position;
                        let offender_radius = offender.radius;
                        let offending_owner = offending_owner.clone();
                        results.push(ApplyCombat(Box::new(move |entities: &mut [Entity], projectiles: &mut Vec<Projectile>, effects: &mut Vec<CommitEffect>| {
                            effects.push(presentation::parry(dir, offender_position, offender_radius));

                            match offending_owner.source {
                                HitcircleSource::Fist { .. } => {
                                    get_entity(entities, offending_owner.entity_id).state = EntityState::Staggered { ttl: 120 };
                                },
                                HitcircleSource::Projectile { projectile_id } => {
                                    projectiles.retain(|x| x.id != projectile_id);
                                },
                            }
                        })));
                        parried = true;
                    }
                }
            }
        }

        if parried { continue; }

        // Not parried, find everyone we hit
        for defender in v.iter() {
            if offending_owner.entity_id == defender.id { continue; }
            if offender.hits.iter().find(|&&hit_id| hit_id == defender.id).is_some() { continue; }
            if circle_overlap(offender.position, offender.radius, defender.position, defender.radius) {
                let angle = vec_angle(offender.position - defender.position);
                let blocked = match defender.state {
                    EntityState::Idle { blocking: true, .. } => min_angle_diff(angle, defender.facing) < BLOCK_SPREAD / fixed!(2),
                    _ => false,
                };
                let dir = (offender.position - defender.position).normalize();
                let owner = offending_owner.clone();
                let defending_id = defender.id;
                let damage = offender.damage;
                let offender_position = offender.position;
                let offender_radius = offender.radius;
                results.push(ApplyCombat(Box::new(move |entities, projectiles, effects| {
                    effects.push(presentation::hit(blocked, dir, offender_position, offender_radius));

                    match owner.source {
                        HitcircleSource::Fist { fist_index } => {
                            if let Entity { ref mut fists, state: EntityState::Action { .. }, .. } = *get_entity(entities, owner.entity_id) {
                                if let Some(fist) = fists.get_mut(fist_index) {
                                    fist.hits.push(defending_id);
                                }
                            }
                        },
                        HitcircleSource::Projectile { projectile_id } => {
                            projectiles.iter_mut().find(|x| x.id == projectile_id).map(|p| p.hitcircle.hits.push(defending_id));
                        },
                    }
                    let ref mut e = get_entity(entities, defending_id);
                    if blocked {
                        e.stats.stamina_display.hold(e.stats.stamina);
                        e.stats.stamina = e.stats.stamina - damage;
                    } else {
                        e.stats.health_display.hold(e.stats.health);
        				e.stats.health = e.stats.health - damage;
                        if e.stats.poise <= damage {
                            let current_ttl = match e.state { EntityState::Staggered { ttl } => ttl, _ => 0 };
                            e.state = EntityState::Staggered { ttl: max(current_ttl, 60) };
                            e.push_vel = e.push_vel + dir * fixed!(-5);
                        }
                    }
                })));
            }
        }
	}

	results
}

fn attack_wall_pass(lines: &[collision::Line], hitcircles: &[(HitcircleOwner, &Hitcircle)]) -> Vec<ApplyCombat> {
    let mut results = vec![];
    for &(ref offending_owner, offender) in hitcircles.iter() {
        for line in lines.iter() {
            match collision::overlapping_point_line(&collision::Circle { center: offender.position, radius: offender.radius }, line) {
                Some(p) => {
                    let dir = offender.position - p;
                    let owner = offending_owner.clone();
                    let entity_id = owner.entity_id;
                    let offender_position = offender.position;
                    let offender_radius = offender.radius;
                    results.push(ApplyCombat(Box::new(move |entities, projectiles, effects| {
                        effects.push(presentation::attack_wall(dir, offender_position, offender_radius));
                        match owner.source {
                            HitcircleSource::Fist { .. } => {
                                get_entity(entities, entity_id).state = EntityState::Staggered { ttl: 30 };
                            },
                            HitcircleSource::Projectile { projectile_id } => {
                                projectiles.retain(|x| x.id != projectile_id);
                            },
                        }
                    })));
                    break;
                },
                _ => (),
            }
        }
    }
    results
}

pub fn get_entity(v: &mut [Entity], id: u32) -> &mut Entity {
	v.iter_mut().find(|x| x.id == id).unwrap()
}

// projectiles.retain(|x| projectiles_to_remove.iter().find(|&&y| y == x.id).is_none());

fn move_entities(v: &mut Vec<Entity>, lines: &Vec<collision::Line>) {
    let velocities = v.iter().map(get_velocity).collect::<Vec<_>>();
    let mut colliders = velocities.into_iter().zip(v.iter()).map(|(vel, entity)| {
        collision::Collider {
            circle: collision::Circle {
                center: entity.position,
                radius: entity.radius,
            },
            velocity: vel
        }
    }).collect();
    collision::move_colliders(&mut colliders, lines);
    for (collider, entity) in colliders.into_iter().zip(v.iter_mut()) {
        entity.position = collider.circle.center;
    }
}

fn move_projectiles(v: &mut Vec<Projectile>) {
    for p in v.iter_mut() {
        p.hitcircle.position = p.hitcircle.position + p.velocity;
    }
}

fn advance_projectiles(v: &mut Vec<Projectile>) {
    for projectile in v.iter_mut() {
        projectile.ttl = projectile.ttl.saturating_sub(1);
    }
    v.retain(|p| {
        match p.kind {
            ProjectileType::Normal => p.ttl > 0 && p.hitcircle.hits.len() == 0,
            ProjectileType::Piercing => p.ttl > 0,
        }
    });
}

fn spawn_entity(id: u32) -> Entity {
    let mut e = Entity {
        id: id,
        position: Vec2::new(fixed!(0), fixed!(0)),
        push_vel: Vec2::new(fixed!(0), fixed!(0)),
        radius: fixed!(30),
        stats: EntityStats {
            move_speed: fixed!(3),
            blocking_move_speed: fixed!(3) / fixed!(2),
            turn_rate: fixed!(2) / fixed!(10),

            stamina: fixed!(100),
            stamina_display: DisplayHold { state: DisplayHoldState::Hide },
            stamina_max: fixed!(100),
            stamina_regen: fixed!(1) / fixed!(2),
            blocking_stamina_regen: fixed!(1) / fixed!(10),

            health: fixed!(100),
            health_display: DisplayHold { state: DisplayHoldState::Hide },
            health_max: fixed!(100),

            poise: fixed!(40),
        },
        facing: fixed!(0),
        state: EntityState::Idle { desired_facing: fixed!(0), vel: Vec2::new(fixed!(0), fixed!(0)), blocking: false },
        weapon: Loadable::load("data/test.weapon.json"),
        fists: vec![],
    };
    e.fists = idle_fists(e.position, e.facing, &e.weapon);
    e
}

use std::iter::FromIterator;
pub fn update_state(game: &mut GameState, player_commands: &BTreeMap<u64, Vec<network::Command>>) -> Vec<CommitEffect> {
    let mut effects = vec![];
    for player in player_commands.iter().flat_map(|(&id, commands)| commands.iter().flat_map(move |c| match c { &network::Command::RemovePlayer => Some(id), _ => None })) {
        for (entity, _) in game.entities.iter_mut().zip(game.controllers.iter()).filter(|&(_, ref controller)| controller.get_owner() == Some(player as usize)) {
            entity.stats.health = fixed!(0);
        }
    }
    let player_inputs = BTreeMap::from_iter(player_commands.iter().flat_map(|(&id, commands)| {
        commands.iter().filter_map(|c| match c {
            &network::Command::SetInputs(ref inputs) => Some((id, inputs.clone())),
             _ => None
         }).last()
    }));

    for inputs in player_inputs.iter() {
        if *inputs.1.dbg_buttons.get(6).unwrap_or(&false) {
            let id = game.next_projectile_id;
            game.next_projectile_id += 1;
            game.projectiles.push(Projectile {
                id: id,
                hitcircle: Hitcircle {
                    position: Vec2::new(fixed!(0), fixed!(0)),
                    radius: fixed!(20),
                    damage: fixed!(5),
                    hits: vec![],
                    state: HitcircleState::Attack,
                },
                velocity: Vec2::new(fixed!(20), fixed!(0)),
                owning_entity_id: 0,
                ttl: 180,
                kind: ProjectileType::Normal,
            });
        }
        if *inputs.1.dbg_buttons.get(3).unwrap_or(&false) {
            horde::spawn(1, game);
        }
    }

    for (&player_id, inputs) in player_inputs.iter() {
        if *inputs.dbg_buttons.get(7).unwrap_or(&false) {
            if !game.controllers.iter().any(|c| c.get_owner() == Some(player_id as usize)) {
                let id = game.next_entity_id;
                game.next_entity_id += 1;
                game.entities.push(spawn_entity(id));
                game.controllers.push(Controller::Player(PlayerController::new(player_id as usize)));
            }
        }
    }

    /* for player in player_commands.iter().flat_map(|(&id, commands)| commands.iter().flat_map(move |c| match c { &network::Command::Spawn => Some(id), _ => None })) {
        let id = game.next_entity_id;
        game.next_entity_id += 1;
        game.entities.push(spawn_entity(id));
        game.controllers.push(Controller::Player(PlayerController::new(player as usize)));
    } */

    for entity in game.entities.iter_mut() {
        if let Some(mut projectile) = advance_entity_state(entity) {
            projectile.id = game.next_projectile_id;
            game.next_projectile_id += 1;
            game.projectiles.push(projectile);
        }
    }
    let commands = {
        let ref entities = game.entities;
        game.controllers.iter_mut().zip(game.entities.iter()).map(|(controller, entity)| {
            controller.control(entity, entities, &player_inputs)
        }).collect::<Vec<Command>>()
    };
    for (entity, command) in game.entities.iter_mut().zip(commands.into_iter()) {
        effects.extend(apply_entity_input(entity, command));
    }
    movement_pass(&mut game.entities);
    move_entities(&mut game.entities, &game.lines);
    move_projectiles(&mut game.projectiles);
    let combat_results = {
        let hitcircles: Vec<_> = game.entities.iter()
            .filter(|entity| match entity.state { EntityState::Action { .. } => true, _ => false })
            .flat_map(|entity| {
                entity.fists.iter().enumerate().map(move |(index, fist)| {
                    (HitcircleOwner { entity_id: entity.id, source: HitcircleSource::Fist { fist_index: index } }, fist)
                })
            })
            .chain(game.projectiles.iter().map(|projectile|
                (HitcircleOwner { entity_id: projectile.owning_entity_id, source: HitcircleSource::Projectile { projectile_id: projectile.id } }, &projectile.hitcircle)
            ))
            .collect();
        let mut results = combat_pass(&hitcircles, &game.entities);
        let results_b = attack_wall_pass(&game.lines, &hitcircles);
        results.extend(results_b);
        results
    };
    for r in combat_results {
        r.0(&mut game.entities[..], &mut game.projectiles, &mut effects);
    }
    effects.extend(game.entities.iter().filter(|e| e.stats.health <= fixed!(0)).map(|e| presentation::death(e.position, e.radius)));
    advance_projectiles(&mut game.projectiles);

    for entity in game.entities.iter_mut() {
        entity.stats.stamina_display.update(entity.stats.stamina);
        entity.stats.health_display.update(entity.stats.health);
    }

    {
        let mut i = 0;
        let ref entities = game.entities;
        game.controllers.retain(|_| {
            let b = entities[i].stats.health > fixed!(0);
            i += 1;
            b
        });
    }
    game.entities.retain(|e| e.stats.health > fixed!(0));

    horde::update(game);

    for entity in game.entities.iter_mut() { update_fists(entity); }

    effects
}