// Copyright 2018 Jeffery Xiao, 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! This module contains an implementation of the Trivial Group, and
//! an "extractor" function for using it as the group associated with the
//! Counted Skiplist.  Using the Trivial Group as the associated group
//! is functionally the same as having *no* associated group, so it serves
//! as a good default.

use alga::general::{AbstractMagma, Additive, AdditiveGroup, Identity, Inverse};
use num_traits::identities::Zero;
use std::ops::{Add, Neg, AddAssign, SubAssign};

/// Trivial Group with a single element.
/// It is represented using an empty (size 0) struct.
/// 
/// # Examples
///
/// ``` 
/// use counted_skiplist::group_util::TrivialGroup;
///
/// let element = TrivialGroup{};
/// assert_eq!(element, element + element);
/// assert_eq!(element, - element);
/// ```
#[derive(Alga, Clone, Copy, PartialEq, Debug)]
#[alga_traits(Group(Additive))]
pub struct TrivialGroup;

impl AbstractMagma<Additive> for TrivialGroup {
    fn operate(&self, _right: &Self) -> Self {
        TrivialGroup {}
    }
}

impl Identity<Additive> for TrivialGroup {
    fn identity() -> Self {
        TrivialGroup {}
    }
}

impl Inverse<Additive> for TrivialGroup {
    fn inverse(&self) -> Self {
        TrivialGroup {}
    }
}

impl Zero for TrivialGroup {
    fn zero() -> Self {
        TrivialGroup::identity()
    }
    fn is_zero(&self) -> bool {
        true
    }
}

impl Add for TrivialGroup {
    type Output = TrivialGroup;
    fn add(self, _other: TrivialGroup) -> Self::Output {
        self
    }
}

impl Neg for TrivialGroup {
    type Output = TrivialGroup;
    fn neg(self) -> TrivialGroup {
        self
    }
}

impl AddAssign for TrivialGroup {
    fn add_assign(&mut self, _other : TrivialGroup) {
    }
}

impl SubAssign for TrivialGroup {
    fn sub_assign(&mut self, _other :TrivialGroup) {
    }
}

/// A default "Extractor" which maps a pair of any type
/// to the trivial group.
/// 
/// An "extractor" is used to map the elements inserted into
/// a 
#[derive(Clone, Copy)]
pub struct DefaultExtractor<S: AdditiveGroup + Copy> {
    phantom: std::marker::PhantomData<S>,
}

impl<S: AdditiveGroup + Copy> DefaultExtractor<S> {
    /// Create a new DefaultExtractor.`
    pub fn new() -> DefaultExtractor<S> {
        DefaultExtractor {
            phantom: std::marker::PhantomData
        }
    }
}
impl<T, S: AdditiveGroup + Copy> FnOnce<(&T, &T)> for DefaultExtractor<S> {
    type Output = S;

    extern "rust-call" fn call_once(self, _args: (&T, &T)) -> Self::Output {
        S::identity()
    }
}

impl<T, S: AdditiveGroup + Copy> FnMut<(&T, &T)> for DefaultExtractor<S> {
    extern "rust-call" fn call_mut(&mut self, _args: (&T, &T)) -> Self::Output {
        S::identity()
    }
}

impl<T, S: AdditiveGroup + Copy> Fn<(&T, &T)> for DefaultExtractor<S> {
    extern "rust-call" fn call(&self, _args: (&T, &T)) -> Self::Output {
        S::identity()
    }
}
