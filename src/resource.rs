use std;
use std::rc::Rc;
use rustc_serialize::{Encoder, Decoder, Decodable, Encodable};
pub trait Loadable: Decodable {
    fn load(path: &str) -> Resource<Self>;
}

#[derive(Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
struct ResourceId {
    path: String,
    // TODO: This hash doesn't solve most resource loading desyncs, only when a Resource happens to
    //       get serialized as part of the game state.  It might be better to do away with this
    //       specific hash check (on the contents of the file, which may or may not hold only game-
    //       relevant data) and instead periodically hash and compare the game state across clients.
    //       Maybe hash the whole ./data folder and only allow loading resources from there?  Maybe
    //       load the whole folder into one Resource so they get their hashes compared?
    //       Really the problem is that this hash won't prevent two .attack.json files being different
    //       so long as that attack isn't in progress when the second client joins, because once they
    //       both go to load the file separately but in lock step, there's no way currently for them
    //       to compare the results of loading a resource.  Resources are assumed to be as identical
    //       and deterministic as the executable for all clients.
    hash: Vec<u8>,
}

#[derive(Debug)]
struct ResourceInner<T> {
    data: T,
    id: ResourceId,
}
#[derive(Debug)]
pub struct Resource<T> {
    _p: Rc<ResourceInner<T>>,
}

impl<T> Resource<T> {
    pub fn new(d: T, path: String, hash: Vec<u8>) -> Resource<T> {
        Resource {
            _p: Rc::new(ResourceInner {
                data: d,
                id: ResourceId {
                    path: path,
                    hash: hash,
                },
            })
        }
    }
}

impl<T> Clone for Resource<T> {
    fn clone(&self) -> Self {
        Resource { _p: self._p.clone() }
    }
}

impl<T> std::ops::Deref for Resource<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self._p.data
    }
}
impl<T> Encodable for Resource<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        self._p.id.encode(s)
    }
}
impl<T: Loadable> Decodable for Resource<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
        let id: ResourceId = try!(Decodable::decode(d));
        let t = Loadable::load(&id.path);
        if t._p.id == id { Ok(t) } else { Err(d.error("Mismatched hash")) }
    }
}

pub trait Cacheable: Sized {
    fn load_for_cache(bytes: Vec<u8>) -> Self;
}
macro_rules! implement_cache {
    ($i:ident, $t:ty) => {
        thread_local! {
            static $i: RefCell<HashMap<String, Resource<$t>>> = RefCell::new(HashMap::new())
        }

        impl Loadable for $t {
            fn load(path: &str) -> Resource<Self> {
                use std::fs::File;
                use std::io::Read;
                use sha1;

                fn is_path_legal(path: &str) -> bool {
                    use std::path::{PathBuf, Component};
                    use std::convert::From;
                    for component in PathBuf::from(path).components() {
                        match component {
                            Component::CurDir | Component::Normal(_) => (),
                            _ => return false,
                        }
                    }
                    true
                }

                $i.with(|mcell| {
                    let mut m = mcell.borrow_mut();
            		let has = m.contains_key(path);
            		if !has {
                        if !is_path_legal(path) {
                            panic!("Resource references illegal path: {}", path);
                        }
                        let mut bytes = vec![];
                        let mut file = File::open(path).unwrap();
                        file.read_to_end(&mut bytes).unwrap();
                        let mut hasher = sha1::Sha1::new();
                        hasher.update(&bytes[..]);
                        let hash = hasher.digest();
            			let t = Resource::<$t>::new(
                            Cacheable::load_for_cache(bytes),
                            path.to_string(),
                            hash,
                        );
            			m.insert(path.to_string(), t.clone());
            			t
            		} else {
            			m.get(path).unwrap().clone()
            		}
                })
            }
        }
    }
}
