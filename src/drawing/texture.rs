use std;
use std::rc::Rc;
use glium::backend::Context;
use glium;
use drawing::{load_image_srgb};

// TODO: These should be SrgbTexture2d's.
pub struct TextureCache {
	facade: Rc<Context>,
	cache: std::collections::HashMap<&'static str, glium::texture::SrgbTexture2d>,
}
impl TextureCache {
	pub fn new(facade: &Rc<Context>) -> TextureCache {
		TextureCache {
			facade: facade.clone(),
			cache: std::collections::HashMap::new(),
		}
	}
	pub fn get(&mut self, path: &'static str) -> &glium::texture::SrgbTexture2d {
		if !self.cache.contains_key(path) {
			self.cache.insert(path, glium::texture::SrgbTexture2d::new(&self.facade, load_image_srgb(path)).unwrap());
		}
		self.cache.get(path).unwrap()
	}
}