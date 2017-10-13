use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::iter::FromIterator;
use std::cmp::{min, max};
use game;
use network;

pub struct Engine {
	canonical_frame: u64,
	preserved_frame: u64,

    state_base_frame: u64,
    // [0,len) => [state_base_frame, ..)
	states: VecDeque<(Vec<game::CommitEffect>, game::GameState)>,

    // [0,len) => [canonical_frame + 1, ..)
	inputs: VecDeque<BTreeMap<u64, Vec<network::Command>>>,

	info_discarded_predictions: u32,
}
impl Engine {
    pub fn new(frame: u64, state: game::GameState) -> Engine {
        Engine {
			canonical_frame: frame,
            preserved_frame: frame,
            state_base_frame: frame,
            inputs: VecDeque::new(),
			states: VecDeque::from_iter(vec![(vec![], state)].into_iter()),
			info_discarded_predictions: 0,
        }
    }
	pub fn dbg_get_canonical_frame(&self) -> u64 {
		self.canonical_frame
	}
	pub fn dbg_get_earliest_frame(&self) -> u64 {
		self.state_base_frame
	}
    fn first_input_frame(&self) -> u64 {
        self.canonical_frame + 1
    }
    fn last_input_frame(&self) -> u64 {
        self.first_input_frame() + self.inputs.len() as u64 - 1
    }
    fn first_state_frame(&self) -> u64 {
        self.state_base_frame
    }
    fn last_state_frame(&self) -> u64 {
        self.first_state_frame() + self.states.len() as u64 - 1
    }
    fn discard_old_states(&mut self) {
        // Discard old states
        let earliest_state_needed = min(self.canonical_frame, self.preserved_frame);
        while self.first_state_frame() < earliest_state_needed {
            self.state_base_frame += 1;
            self.states.pop_front();
        }
        assert!(self.canonical_frame - self.state_base_frame < self.states.len() as u64);
    }
	pub fn pending_inputs(&self) -> Vec<(u64, u64, network::Command)> {
		self.inputs.iter().enumerate()
			.flat_map(|(frame_offset, frame_commands)|
				frame_commands.iter().flat_map(move |(&player, commands)|
					commands.iter().map(move |command| (self.first_input_frame() + frame_offset as u64, player, command.clone())))).collect()
	}
    pub fn add_input(&mut self, frame: u64, player: u64, command: network::Command) {
        assert!(frame >= self.first_input_frame());
        while self.last_input_frame() < frame {
            self.inputs.push_back(BTreeMap::new());
        }
        let insert_index = (frame - self.first_input_frame()) as usize;
        self.inputs.get_mut(insert_index).unwrap().entry(player).or_insert_with(|| vec![]).push(command);

        // Discard predictions from frame onwards
        while self.last_state_frame() >= frame {
			self.info_discarded_predictions += 1;
            self.states.pop_back();
        }
    }
	pub fn poll_discarded_predictions(&mut self) -> u32 {
		let r = self.info_discarded_predictions;
		self.info_discarded_predictions = 0;
		r
	}
    pub fn commit(&mut self, frame: u64) -> Vec<Vec<game::CommitEffect>> {
        assert!(frame > self.canonical_frame);
        self.simulate_to(frame);

		let effects = (self.canonical_frame + 1..frame + 1).map(|f| {
			let index = (f - self.first_state_frame()) as usize;
			::std::mem::replace(&mut self.states.get_mut(index).unwrap().0, vec![])
		}).collect();

        // Discard inputs for frames <= frame
        while self.first_input_frame() <= frame {
            self.canonical_frame += 1;
            self.inputs.pop_front();
        }
        self.discard_old_states();

        assert!(self.state_base_frame <= frame);
		effects
    }
    pub fn set_preserved_frame(&mut self, frame: u64) {
        self.preserved_frame = frame;
        self.discard_old_states();
    }
    pub fn simulate_to(&mut self, frame: u64) {
        assert!(frame >= self.state_base_frame);
        loop {
            let f = self.last_state_frame();
            if f >= frame { break; }
            let mut ns = self.states.get((f - self.first_state_frame()) as usize).unwrap().1.clone();
            let es = match self.inputs.get((f + 1 - self.first_input_frame()) as usize) {
                Some(inputs) => game::update_state(&mut ns, inputs),
                None => game::update_state(&mut ns, &BTreeMap::new()),
            };
            self.states.push_back((es, ns));
        }
        assert!(self.canonical_frame - self.state_base_frame < self.states.len() as u64);
    }
    pub fn get_closest_state(&self, frame: u64) -> &game::GameState {
        let index =
            if frame < self.state_base_frame { 0 }
            else if frame >= self.last_state_frame() { self.states.len() - 1 }
            else { (frame - self.first_state_frame()) as usize };
        &self.states.get(index).unwrap().1
    }
}