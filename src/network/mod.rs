pub mod server;

use std::collections::BTreeMap;
use game::Inputs;

struct InputConn {
    // players: Vec<u64>,
    last_incoming_commit: u64,
}
struct OutputConn {
    last_outgoing_commit: u64,
}

#[derive(Debug, Clone, RustcEncodable, RustcDecodable)]
pub enum Command {
    SetInputs(Inputs),
    Spawn,
    AddPlayer,
    RemovePlayer,
}

#[derive(Debug, Clone, RustcEncodable, RustcDecodable)]
pub enum Message {
    Command {
        frame: u64,
        player: u64,
        command: Command,
    },
    Commit {
        frame: u64,
    }
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, RustcEncodable, RustcDecodable)]
pub enum PlayerModification {
    Add,
    Remove,
}

struct Player {
    owner: u64,
    is_finished: bool,
    last_command: u64,
}

pub struct Hub {
    inputs: BTreeMap<u64, InputConn>,
    outputs: BTreeMap<u64, OutputConn>,

    players: BTreeMap<u64, Player>,

    next_player_id: u64,
    pending_players: Vec<(u64, u64, PlayerModification)>, // (frame, by_player, command)
}

pub type OutputMessages = Vec<(u64, Message)>;

impl Hub {
    pub fn get_all_players(&self) -> Vec<u64> {
        self.players.iter().map(|(&pid, _)| pid).collect()
    }
    pub fn get_players(&self, id: u64) -> Vec<u64> {
        self.players.iter().filter(|&(_, player)| player.owner == id).map(|(&pid, _)| pid).collect()
    }
    pub fn get_next_player_id(&self) -> u64 {
        self.next_player_id
    }
    pub fn get_canonical_frame(&self) -> u64 {
        self.inputs.iter()
            .map(|(_, p)| p.last_incoming_commit)
            .min().unwrap()
    }
    fn update_canonical_frame(&mut self) -> OutputMessages {
        let mut commits_to_send = vec![];
        for (&out_id, out_peer) in self.outputs.iter_mut() {
            let out_commit = self.inputs.iter()
                .filter(|&(&id, _)| id != out_id)
                .map(|(_, p)| p.last_incoming_commit)
                .min();
            match out_commit {
                Some(c) if c > out_peer.last_outgoing_commit => {
                    commits_to_send.push((out_id, c));
                    out_peer.last_outgoing_commit = c;
                },
                _ => { }
            }
        }
        let canonical_frame = self.get_canonical_frame();
        {
            let mut players_to_add: Vec<_> = self.pending_players.iter().filter(|&&(frame, _, _)| frame <= canonical_frame).collect();
            players_to_add.sort();
            for &&(_, player, ref c) in players_to_add.iter() {
                match *c {
                    PlayerModification::Add => {
                        // A player cannot send any commands until they are officially committed to a peer
                        let new_player_id = self.next_player_id;
                        self.next_player_id += 1;
                        let owner = self.players[&player].owner;
                        self.players.insert(new_player_id, Player {
                            owner: owner,
                            is_finished: false,
                            last_command: canonical_frame + 1,
                        });
                    },
                    PlayerModification::Remove => {
                        self.players.remove(&player);
                    },
                }
            }
        }
        self.pending_players.retain(|&(frame, _, _)| frame > canonical_frame);
        commits_to_send.iter().map(|&(i, c)| (i, Message::Commit { frame: c })).collect()
    }
    fn commit(&mut self, peer_id: u64, frame: u64) -> OutputMessages {
        {
            let peer = self.inputs.get_mut(&peer_id).unwrap();
            assert!(frame > peer.last_incoming_commit);
            peer.last_incoming_commit = frame;
        }
        self.update_canonical_frame()
    }
    fn add_command(&mut self, peer_id: u64, frame: u64, player: u64, command: Command) -> OutputMessages {
        {
            let peer = &self.inputs[&peer_id];
            let player = &self.players[&player];
            assert!(frame > peer.last_incoming_commit);
            assert!(player.owner == peer_id);
            assert!(frame >= player.last_command);
            assert!(!player.is_finished);
        }
        self.players.get_mut(&player).unwrap().last_command = frame;
        match command {
            Command::AddPlayer => {
                self.pending_players.push((frame, player, PlayerModification::Add));
            },
            Command::RemovePlayer => {
                self.pending_players.push((frame, player, PlayerModification::Remove));
                self.players.get_mut(&player).unwrap().is_finished = true;
            },
            _ => { }
        }
        self.outputs.iter()
            .filter(|&(&id, _)| id != peer_id)
            .map(|(&id, _)| (id, Message::Command { frame: frame, player: player, command: command.clone() }))
            .collect()
    }

    pub fn process_message(&mut self, peer_id: u64, message: Message) -> OutputMessages {
        assert!(self.inputs.contains_key(&peer_id));
        match message {
            Message::Commit { frame } => self.commit(peer_id, frame),
            Message::Command { frame, player, command } => self.add_command(peer_id, frame, player, command),
        }
    }

    pub fn add_input_conn(&mut self, peer_id: u64, frame: u64, players: Vec<u64>) {
        self.inputs.insert(peer_id, InputConn {
            last_incoming_commit: frame,
        });
        for player in players.iter() {
            self.players.insert(*player, Player {
                owner: peer_id,
                is_finished: false,
                last_command: frame + 1,
            });
        }
    }
    pub fn add_output_conn(&mut self, peer_id: u64, frame: u64) {
        self.outputs.insert(peer_id, OutputConn {
            last_outgoing_commit: frame,
        });
    }
    pub fn remove_input_conn(&mut self, peer_id: u64, transfer_players_to: u64) -> Vec<u64> {
        self.inputs.remove(&peer_id).unwrap();
        self.players.iter_mut().filter(|&(_, ref player)| player.owner == peer_id).map(|(&pid, player)| {
            player.owner = transfer_players_to;
            pid
        }).collect()
    }
    pub fn remove_output_conn(&mut self, peer_id: u64) {
        self.outputs.remove(&peer_id);
    }
    // TODO: Find a way to make these two dbg_* functions unnecessary
    pub fn dbg_player_last_frame(&self, player_id: u64) -> u64 {
        self.players.get(&player_id).unwrap().last_command
    }
    pub fn dbg_remove_local_player(&mut self, peer_id: u64, player_id: u64) {
        let player = self.players.remove(&player_id).unwrap();
        assert!(player.owner == peer_id);
    }

    pub fn new(next_player_id: u64) -> Hub {
        Hub {
            next_player_id: next_player_id,
            pending_players: vec![],
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            players: BTreeMap::new(),
        }
    }
}

/*
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use super::{Network, Hub, Command, NetworkMessage, InputConn, OutputConn};
    use std::sync::mpsc::{Sender, Receiver, channel};
    use na::Vec2;
    use game::Inputs;

    fn default_inputs() -> Inputs {
        Inputs {
        	movement: Vec2::new(fixed!(0), fixed!(0)),
        	facing: Vec2::new(fixed!(0), fixed!(0)),
        	roll: false,
        	attack: false,
        	attack_dumb: false,
            attack_special: false,
        	parry: false,
        	block: false,
            dbg_buttons: vec![],
        }
    }

    #[test]
    fn connection_test() {
        let (joy_sender, obs_receiver, mut network) = Network::local();

        let cmd_message = NetworkMessage::AddCommand { frame: 1, player: 0, command: Command::Game(default_inputs()) };
        joy_sender.send(cmd_message.clone());
        network.poll_inputs();
        assert!(match obs_receiver.try_recv() { Ok(NetworkMessage::AddCommand { .. }) => true, _ => false });

        let cmt_message = NetworkMessage::Commit { frame: 1 };
        joy_sender.send(cmt_message.clone());
        network.poll_inputs();
        assert!(match obs_receiver.try_recv() { Ok(NetworkMessage::Commit { frame: 1 }) => true, _ => false });
    }
}
*/