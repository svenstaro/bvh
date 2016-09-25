#![feature(plugin)]
#![feature(test)]

#![plugin(clippy)]

#[cfg(test)]
extern crate test;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

pub extern crate nalgebra;

pub mod aabb;
