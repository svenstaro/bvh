#![feature(plugin)]
#![feature(test)]

#![plugin(clippy)]

#[cfg(test)]
extern crate test;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

extern crate nalgebra;

pub mod aabb;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
