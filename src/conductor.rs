use game;
use network;
use engine;
use std::sync::mpsc::{self, Sender, Receiver};
use std::collections::BTreeMap;
use std::cmp::{max};
use rustc_serialize::{Decodable, Encodable};
use msgpack::{self, Encoder, Decoder};

const NUM_INTERP_FRAMES: u64 = 10;

#[derive(Debug, Clone, RustcEncodable, RustcDecodable)]
pub enum NetworkMessage {
    Basic(network::Message),
    InitialState { state: game::GameState, players: Vec<u64>, next_player_id: u64, frame: u64 },
    RequestJoin,
    JoinAs { frame: u64, player: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NodeState {
	Observing,
	Joining,
	Playing,
}

struct NetConnection {
	sender: Sender<Vec<u8>>,
	receiver: Receiver<Vec<u8>>,
	state: NodeState,
}

const LOCAL_OUTPUT: u64 = 0;
const LOCAL_INPUT: u64 = 1;
pub struct Conductor {
	engine: engine::Engine,
	hub: network::Hub,
	active_frame: u64,

	state: NodeState,
	net_connections: BTreeMap<u64, NetConnection>,
	next_conn_id: u64,

    players_to_remove: Vec<u64>,

    need_spawn: bool,
    want_drop_frame: bool,

    buffered_frames: u64,
}

#[derive(Debug)]
enum MessageReceiveError<T> {
    Malformed(msgpack::decode::Error),
    Channel(T),
}
use std::convert::From;
impl<T> From<msgpack::decode::Error> for MessageReceiveError<T> {
    fn from(err: msgpack::decode::Error) -> MessageReceiveError<T> {
        MessageReceiveError::Malformed(err)
    }
}
impl From<mpsc::RecvError> for MessageReceiveError<mpsc::RecvError> {
    fn from(err: mpsc::RecvError) -> MessageReceiveError<mpsc::RecvError> {
        MessageReceiveError::Channel(err)
    }
}
impl From<mpsc::TryRecvError> for MessageReceiveError<mpsc::TryRecvError> {
    fn from(err: mpsc::TryRecvError) -> MessageReceiveError<mpsc::TryRecvError> {
        MessageReceiveError::Channel(err)
    }
}

fn receive_message<T: Decodable>(receiver: &Receiver<Vec<u8>>) -> Result<T, MessageReceiveError<mpsc::RecvError>> {
    let bytes = try!(receiver.recv());
    let t = try!(Decodable::decode(&mut Decoder::new(&bytes[..])));
    Ok(t)
}
fn try_receive_message<T: Decodable>(receiver: &Receiver<Vec<u8>>) -> Result<T, MessageReceiveError<mpsc::TryRecvError>> {
    let bytes = try!(receiver.try_recv());
    let t = try!(Decodable::decode(&mut Decoder::new(&bytes[..])));
    Ok(t)
}
fn send_message<T: Encodable>(sender: &Sender<Vec<u8>>, t: T) -> Result<(), mpsc::SendError<T>> {
    let mut bytes = vec![];
    t.encode(&mut Encoder::new(&mut bytes)).unwrap();
    sender.send(bytes).map_err(|_| mpsc::SendError(t))
}

impl Conductor {
	pub fn local(state: game::GameState) -> Conductor {
        println!("Hosting");
		let mut hub = network::Hub::new(2);
		hub.add_input_conn(LOCAL_INPUT, 0, vec![0]);
		hub.add_output_conn(LOCAL_OUTPUT, 0);
		let engine = engine::Engine::new(0, state);
		Conductor {
			engine: engine,
			hub: hub,
			active_frame: 1,

			state: NodeState::Playing,
			net_connections: BTreeMap::new(),
			next_conn_id: 2,

            players_to_remove: vec![],

            need_spawn: true,
            want_drop_frame: false,

            buffered_frames: 0,
		}
	}
    pub fn remote(conn: network::server::NewConnection) -> Conductor {
        println!("Observing");
        if let NetworkMessage::InitialState { state, players, next_player_id, frame } = receive_message(&conn.receiver).unwrap() {
            let mut hub = network::Hub::new(next_player_id);
            let mut net_connections = BTreeMap::new();
            net_connections.insert(2, NetConnection {
                receiver: conn.receiver,
                sender: conn.sender,
                state: NodeState::Playing,
            });
            hub.add_input_conn(2, frame, players);
            hub.add_output_conn(LOCAL_OUTPUT, frame);
            let engine = engine::Engine::new(frame, state);

            Conductor {
                engine: engine,
                hub: hub,
                active_frame: frame + 1,
                state: NodeState::Observing,
                net_connections: net_connections,
                next_conn_id: 3,
                players_to_remove: vec![],
                need_spawn: false,
                want_drop_frame: false,
                buffered_frames: 0,
            }
        } else {
            panic!("OH FUCK");
        }
    }
    pub fn join(&mut self) {
        if self.state == NodeState::Observing {
            println!("Joining");
            let host = self.net_connections.iter().find(|&(_, c)| c.state == NodeState::Playing).unwrap();
            send_message(&host.1.sender, NetworkMessage::RequestJoin).unwrap();
            self.state = NodeState::Joining;
        } else {
            panic!("Can't join if not observing");
        }
    }
	fn send_to_engine(&mut self, message: network::Message) -> Vec<Vec<game::CommitEffect>> {
		match message {
			network::Message::Command { frame, player, command } => {
                self.engine.add_input(frame, player, command);
                vec![]
			},
			network::Message::Commit { frame } => {
				self.engine.commit(frame)
			}
		}
	}
    pub fn dbg_drop_frame(&mut self) {
        self.want_drop_frame = true;
    }
    pub fn set_num_buffered_frames(&mut self, n: u64) {
        self.buffered_frames = n;
    }
    pub fn get_buffered_frame(&self) -> u64 {
        self.active_frame.saturating_sub(self.buffered_frames)
    }
    pub fn get_local_player(&self) -> Option<u64> {
        self.hub.get_players(LOCAL_INPUT).get(0).map(|&id| id)
    }
    pub fn get_active_frame(&self) -> u64 {
        self.active_frame
    }
    pub fn simulate_to(&mut self, frame: u64) {
        self.engine.simulate_to(frame);
    }
    pub fn get_state(&self, frame: u64) -> &game::GameState {
        self.engine.get_closest_state(frame)
    }
    pub fn dbg_frames(&self) -> String {
        format!("{}, {}, {}, Active {}, Engine {}, Hub {}",
            self.buffered_frames,
            self.get_buffered_frame().saturating_sub(2) as i64 - self.engine.dbg_get_earliest_frame() as i64,
            self.want_drop_frame,
            self.active_frame,
            self.engine.dbg_get_canonical_frame(),
            self.hub.get_canonical_frame())
    }
    pub fn dbg_second(&mut self) -> String {
        format!("Discarded {} predictions", self.engine.poll_discarded_predictions())
    }
    pub fn add_net_conn(&mut self, conn: network::server::NewConnection) {
        println!("Adding Observer {}", self.next_conn_id);
        let frame = self.engine.dbg_get_canonical_frame();
        // Send initial state
        let init = NetworkMessage::InitialState {
            state: self.engine.get_closest_state(frame).clone(),
            players: self.hub.get_all_players(),
            next_player_id: self.hub.get_next_player_id(),
            frame: frame
        };
        send_message(&conn.sender, init).unwrap();
        // Send all pending commands (inputs, add/remove player)
        for (frame, player, command) in self.engine.pending_inputs().into_iter() {
            send_message(&conn.sender, NetworkMessage::Basic(network::Message::Command { frame: frame, player: player, command: command })).unwrap();
        }
        // Now it should be caught up for observing
        self.net_connections.insert(self.next_conn_id, NetConnection {
            receiver: conn.receiver,
            sender: conn.sender,
            state: NodeState::Observing,
        });
        self.hub.add_output_conn(self.next_conn_id, frame);
        self.next_conn_id += 1;
    }
	pub fn advance(&mut self, joystick_input: game::Inputs, dbg_timings: &mut ::DbgNetworkTimings) -> Vec<Vec<game::CommitEffect>> {
    	let mut pending_outputs = vec![];
        if self.want_drop_frame {
            self.want_drop_frame = false;
        } else {
    		if self.state == NodeState::Playing {
                let local_players = self.hub.get_players(LOCAL_INPUT);
                if local_players.len() == 0 { panic!("NO LOCAL PLAYERS"); }
                if self.need_spawn {
                    pending_outputs.extend(self.hub.process_message(LOCAL_INPUT, network::Message::Command { frame: self.active_frame, player: local_players[0], command: network::Command::Spawn }));
                    self.need_spawn = false;
                }
                pending_outputs.extend(self.hub.process_message(LOCAL_INPUT, network::Message::Command {
                    frame: self.active_frame,
                    player: local_players[0],
                    command: network::Command::SetInputs(joystick_input)
                }));
                if self.players_to_remove.len() > 0 {
                    let to_remove = self.players_to_remove.clone();
                    self.players_to_remove.clear();
                    pending_outputs.extend(to_remove.into_iter().flat_map(|player_id| {
                        println!("Removing player {}", player_id);
                        let frame = max(self.active_frame, self.hub.dbg_player_last_frame(player_id));
                        let r = self.hub.process_message(LOCAL_INPUT, network::Message::Command { frame: frame, player: player_id, command: network::Command::RemovePlayer });
                        self.hub.dbg_remove_local_player(LOCAL_INPUT, player_id);
                        r
                    }));
                }
                pending_outputs.extend(self.hub.process_message(LOCAL_INPUT, network::Message::Commit { frame: self.active_frame }));
            }
        	self.active_frame += 1;
        	self.engine.set_preserved_frame(self.active_frame.saturating_sub(self.buffered_frames + NUM_INTERP_FRAMES));
        }

        dbg_timings.add_value(LOCAL_OUTPUT, self.get_buffered_frame() as i64 - self.active_frame as i64);
		let mut closed = vec![];
		for (&id, conn) in self.net_connections.iter_mut() {
            loop {
				match try_receive_message::<NetworkMessage>(&conn.receiver) {
                    Ok(in_m) => {
                        match (&conn.state, in_m) {
                            (&NodeState::Playing, NetworkMessage::Basic(network::Message::Command { frame, player, ref command })) => {
                                pending_outputs.extend(self.hub.process_message(id, network::Message::Command { frame: frame, player: player, command: command.clone() }));
                            },
                            (&NodeState::Playing, NetworkMessage::Basic(network::Message::Commit { frame })) => {
                                pending_outputs.extend(self.hub.process_message(id, network::Message::Commit { frame: frame }));

                                // crude time sync
                                // TODO: replace this with something not stupid
                                dbg_timings.add_value(id, frame as i64 - self.active_frame as i64);
                                if self.active_frame > frame + 10 {
                                    self.want_drop_frame = true;
                                }
                            },
                            (&NodeState::Observing, NetworkMessage::RequestJoin) if self.state == NodeState::Playing => {
                                conn.state = NodeState::Joining;
                                // TODO: Kick off adding the player
                                println!("Adding player for {}", id);
                                let local_player = self.hub.get_players(LOCAL_INPUT)[0];
                                pending_outputs.extend(self.hub.process_message(LOCAL_INPUT, network::Message::Command {
                                    frame: self.active_frame, // Hardcoded crap
                                    player: local_player,
                                    command: network::Command::AddPlayer
                                }));
                            },
                            (&NodeState::Playing, NetworkMessage::JoinAs { frame, player }) if self.state == NodeState::Joining => {
                                println!("Joining as {} from {}", id, player);
                                self.active_frame = max(self.active_frame, frame + 1);
                                self.hub.add_input_conn(LOCAL_INPUT, frame, vec![player]);
                                self.hub.add_output_conn(id, frame);
                                self.state = NodeState::Playing;
                                self.need_spawn = true;
                                println!("Local players: {:?}", self.hub.get_players(LOCAL_INPUT));
                            },
                            _ => { panic!("THE FUCK IS THIS?"); }
                        }
                    },
                    Err(MessageReceiveError::Channel(mpsc::TryRecvError::Empty)) => { break; },
                    Err(MessageReceiveError::Channel(mpsc::TryRecvError::Disconnected)) | Err(MessageReceiveError::Malformed(_)) => {
                        closed.push(id);
                        break;
                    }
				}
            }
		}
        dbg_timings.add_value(LOCAL_INPUT, self.hub.get_canonical_frame() as i64 - self.active_frame as i64);

		let mut pending_engine_outputs = vec![];
		for (out_id, out_m) in pending_outputs.into_iter() {
			if out_id == LOCAL_OUTPUT {
				pending_engine_outputs.push(out_m);
			} else {
				let r = send_message(&self.net_connections[&out_id].sender, NetworkMessage::Basic(out_m));
				if r.is_err() {
					closed.push(out_id);
				}
			}
		}
        let mut effects = vec![];
		for engine_output in pending_engine_outputs.into_iter() {
			let es = self.send_to_engine(engine_output);
            effects.extend(es.into_iter());
		}

        // Hacky player giving
        if self.state == NodeState::Playing {
            let players = self.hub.get_players(LOCAL_INPUT);
            if players.len() > 1 {
                for ((&id, conn), player) in self.net_connections.iter_mut().filter(|&(_, ref c)| c.state == NodeState::Joining).zip(players.into_iter().skip(1)) {
                    println!("Giving player {} to {}", player, id);
                    if send_message(&conn.sender, NetworkMessage::JoinAs { frame: self.active_frame, player: player }).is_err() {
                        closed.push(id);
                    }
                    self.hub.add_input_conn(id, self.active_frame, vec![player]);
                    conn.state = NodeState::Playing;
                }
            }
        }
		for closed_id in closed.into_iter() {
			self.remove_net_conn(closed_id);
		}
        effects
	}
	fn remove_net_conn(&mut self, id: u64) {
        println!("Disconnecting {}", id);
		let conn = self.net_connections.remove(&id).unwrap();
		match conn.state {
			NodeState::Observing | NodeState::Joining => {
				self.hub.remove_output_conn(id);
			},
			NodeState::Playing => {
				self.hub.remove_output_conn(id);
				self.players_to_remove.extend(self.hub.remove_input_conn(id, LOCAL_INPUT));
			}
		}
	}
}