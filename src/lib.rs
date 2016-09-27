#![feature(plugin)]
#![feature(test)]

#![plugin(clippy)]

#[cfg(test)]
extern crate test;
#[cfg(test)]
extern crate rand;
#[cfg(test)]
#[macro_use]
extern crate quickcheck;

pub extern crate nalgebra;

pub const EPSILON: f32 = 0.00001;

pub mod aabb;
pub mod ray;
