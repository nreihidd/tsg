use std::io::{Read, Write};
use std::thread;
use std::net::{TcpListener, TcpStream};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::sync::mpsc::{Sender, Receiver, channel};
pub struct NewConnection {
    pub sender: Sender<Vec<u8>>,
    pub receiver: Receiver<Vec<u8>>,
}

pub fn stream_to_channels(mut stream: TcpStream) -> NewConnection {
    let (in_snd, in_rcv) = channel::<Vec<u8>>();
    let (out_snd, out_rcv) = channel::<Vec<u8>>();
    let mut stream2 = stream.try_clone().unwrap();
    // Read
    thread::spawn(move || {
        loop {
            let r = stream.read_u32::<LittleEndian>().ok().and_then(|msg_size| {
                let mut msg_bytes = vec![0; msg_size as usize];
                stream.read_exact(&mut msg_bytes).ok().and_then(|()| {
                    in_snd.send(msg_bytes).ok()
                })
            });
            if r.is_none() { break; }
        }
    });
    // Write
    thread::spawn(move || {
        for msg_bytes in out_rcv.iter() {
            let r = stream2.write_u32::<LittleEndian>(msg_bytes.len() as u32).ok().and_then(|()| {
                stream2.write_all(&msg_bytes).ok()
            });
            if r.is_none() { break; }
        }
    });
    NewConnection {
        sender: out_snd,
        receiver: in_rcv,
    }
}
pub fn server() -> Receiver<NewConnection> {
    let (snd, rcv) = channel();
    thread::spawn(move || {
        let listener = TcpListener::bind("0.0.0.0:1650").unwrap();
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    snd.send(stream_to_channels(stream)).unwrap();
                }
                Err(_) => { /* connection failed */ }
            }
        }
    });
    rcv
}