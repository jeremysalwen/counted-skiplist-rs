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


//! An implmentation of a Counted Skiplist, optimized for solving Traveling Salesman Problems.
//! 
//! This is the Skiplist equivalent of a [Counted B-Tree](http://msinfo.info/msi/cdsisis/basico/countedb-trees.htm)
//!
//! The code is based on the [extended-collections-rs](https://github.com/jeffrey-xiao/extended-collections-rs) skiplist implementation
//! 
//! Like a Skiplist, it provides O(1) previous/next lookup, and O(Log N) average case insertion,
//! deletion, lookup by index, splitting and joining.  On top of these basic operations, the *counted*
//! skiplist also has a value associated with every element, and provides O(Log N) functions for
//! calculating the sum of those values between any two endpoints.
//! 
//! This implementation actually allows the value associated with each element to depend on the
//! element to its right (this comes from the TSP use case, where the element is a city, and the
//! derived value is the distance to the next city.)  This implementation is designed to work 
//! even with noncommutative "sums" (i.e. it will work with any [Group](https://en.wikipedia.org/wiki/Group_(mathematics))
//! 
//! This implementation also supports custom allocators.

#![warn(missing_docs)]
#![feature(allocator_api, alloc_layout_extra, fn_traits, unboxed_closures)]

extern crate alga;
#[macro_use]
extern crate alga_derive;   
extern crate num_traits;

pub mod group_util;

use alga::general::{AdditiveGroup};
use rand::Rng;
use rand::XorShiftRng;
use std::alloc::{Alloc, Global, Layout};
use std::fmt::Debug;
use std::fmt::Error;
use std::fmt::Formatter;
use std::mem;
use std::ops::{Add, Index, IndexMut};
use std::ptr;


type DefaultGroup = i32;

#[repr(C)]
#[derive(Copy, Clone)]
struct Layer<T, S: AdditiveGroup + Copy = DefaultGroup> {
    sum: S,
    next: *mut Node<T, S>,
    next_distance: usize,
    prev: *mut Node<T, S>,
}

#[repr(C)]
struct Node<T, S: AdditiveGroup + Copy = DefaultGroup> {
    links_len: usize,
    value: T,
    links: [Layer<T, S>; 0],
}

/// A Pointer to an element of a CountedSkipList,
/// invalidated iff that element is removed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Finger<T, S: AdditiveGroup + Copy = DefaultGroup>(*mut Node<T, S>);

const MAX_HEIGHT: usize = 32;

impl<T, S: AdditiveGroup + Copy> Node<T, S> {
    pub fn new(value: T, links_len: usize, alloc: &mut Alloc) -> *mut Self {
        let ptr = unsafe { Self::allocate(links_len, alloc) };
        unsafe {
            ptr::write(&mut (*ptr).value, value);
        }
        ptr
    }

    pub fn get_pointer(&self, height: usize) -> &Layer<T, S> {
        unsafe { self.links.get_unchecked(height) }
    }

    pub fn get_pointer_mut(&mut self, height: usize) -> &mut Layer<T, S> {
        unsafe { self.links.get_unchecked_mut(height) }
    }

    fn layout(links_len: usize) -> Layout {
        Layout::new::<Self>()
            .extend(Layout::array::<Layer<T, S>>(2 * links_len).unwrap())
            .unwrap()
            .0
    }

    unsafe fn allocate(links_len: usize, alloc: &mut Alloc) -> *mut Self {
        let ptr: *mut Node<T, S> = alloc
            .alloc_zeroed(Self::layout(links_len))
            .unwrap()
            .cast()
            .as_ptr();
        ptr::write(&mut (*ptr).links_len, links_len);
        // We populate the sum with 0 by default.
        for height in 0..links_len {
            ptr::write(&mut (*ptr).get_pointer_mut(height).sum, S::identity());
        }
        ptr
    }

    unsafe fn deallocate(ptr: *mut Self, alloc: &mut Alloc) {
        alloc.dealloc(
            std::ptr::NonNull::new_unchecked(ptr).cast(),
            Self::layout((*ptr).links_len),
        );
    }

    unsafe fn free(ptr: *mut Self, alloc: &mut Alloc) {
        for i in 0..(*ptr).links_len {
            ptr::drop_in_place(&mut (*ptr).get_pointer_mut(i).sum);
        }
        ptr::drop_in_place(&mut (*ptr).value);
        Self::deallocate(ptr, alloc);
    }

    unsafe fn extract_value(ptr: *mut Self, alloc: &mut Alloc) -> T {
        for i in 0..(*ptr).links_len {
            ptr::drop_in_place(&mut (*ptr).get_pointer_mut(i).sum);
        }
        let value = mem::replace(&mut (*ptr).value, std::mem::uninitialized());
        Self::deallocate(ptr, alloc);
        value
    }
}

unsafe fn link<T, S: AdditiveGroup + Copy>(
    first: *mut Node<T, S>,
    second: *mut Node<T, S>,
    height: usize,
    distance: usize,
) {
    let mut first_link = (*first).get_pointer_mut(height);
    first_link.next = second;
    first_link.next_distance = distance;

    if !second.is_null() {
        let mut second_link = (*second).get_pointer_mut(height);
        second_link.prev = first;
    }
}

unsafe fn swap_link<T, S: AdditiveGroup + Copy>(
    first: *mut Node<T, S>,
    second: *mut Node<T, S>,
    height: usize,
) {
    let first_link = (*first).get_pointer_mut(height);
    let second_link = (*second).get_pointer_mut(height);
    let first_next = first_link.next;
    let first_distance = first_link.next_distance;
    let second_next = second_link.next;
    let second_distance = second_link.next_distance;
    link(first, second_next, height, second_distance);
    link(second, first_next, height, first_distance);
}

/// A Counted SkipList
///
/// A skiplist is a probabilistic data structure that allows for binary search tree operations by
/// maintaining a linked hierarchy of subsequences. The first subsequence is essentially a sorted
/// linked list of all the elements that it contains. Each successive subsequence contains
/// approximately half the elements of the previous subsequence. Using the sparser subsequences,
/// elements can be skipped and searching, insertion, and deletion of keys can be done in
/// approximately logarithm time.
///
/// Each link in this skiplist store the width of the link. The width is defined as the number of
/// bottom layer links being traversed by each of the higher layer links. This augmentation allows
/// the list to get, remove, and insert at an arbitrary index in `O(log N)` time.
/// 
/// Additionally the *counted* skip list has a value associated with every element, and provides
/// `O(Log N)` time lookup of the sums of those values across any given range.  This implementation
/// supports associated values which can be derived from an element and it's right-neighbor,
/// and supports summing any values with a `Group` structure (i.e. they have a 0 element, can be 
/// added together (not neccessarily commutatively), and have an inverse).
/// 
///
/// # Examples
///
/// ```
/// use counted_skiplist::CountedSkipList;
///
/// let mut list = CountedSkipList::new();
/// list.insert(0, 1);
/// list.push_back(2);
/// list.push_front(3);
///
/// assert_eq!(list.get(0), Some(&3));
/// assert_eq!(list.get(3), None);
/// assert_eq!(list.len(), 3);
///
/// *list.get_mut(0).unwrap() += 1;
/// assert_eq!(list.pop_front(), 4);
/// assert_eq!(list.pop_back(), 2);
/// ```
pub struct CountedSkipList<
    T,
    S: AdditiveGroup + Copy = DefaultGroup,
    F: Fn(&T, &T) -> S + Copy = group_util::DefaultExtractor<S>,
    A: Alloc = Global,
> {
    head: *mut Node<T, S>,
    len: usize,
    rng: XorShiftRng,
    key_func: F,
    alloc: A,
}

impl<T> CountedSkipList<T, DefaultGroup, group_util::DefaultExtractor<DefaultGroup>, Global> {
    /// Constructs a new, empty `CountedSkipList<T,S>` in the default allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let list: CountedSkipList<u32, i32, _> = CountedSkipList::new_with_group(|&x, &_y| x as i32);
    /// ```

    pub fn new() -> Self {
        Self::new_with_group(group_util::DefaultExtractor::<DefaultGroup>::new())
    }
}

impl<T, S: AdditiveGroup + Copy, F: Fn(&T, &T) -> S + Copy> CountedSkipList<T, S, F, Global> {
    /// Constructs a new, empty `CountedSkipList<T,S>` in the default allocator.
    /// 
    ///
    /// # Examples
    ///
    /// ``` 
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let list: CountedSkipList<u32> = CountedSkipList::new();
    /// ```
    pub fn new_with_group(key_func: F) -> Self {
        Self::new_with_group_in(key_func, Global {})
    }
}
impl<T, S: AdditiveGroup + Copy, F: Fn(&T, &T) -> S + Copy, A: Alloc> CountedSkipList<T, S, F, A> {
    /// Constructs a new, empty `CountedSkipList<T,S>` in a given allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let alloc = std::alloc::System {};
    /// let mut list = CountedSkipList::new_with_group_in(|&a, &b| 0, alloc);
    /// list.push_back(5);
    /// ```
    pub fn new_with_group_in(key_func: F, mut alloc: A) -> Self {
        CountedSkipList {
            head: unsafe { Node::allocate(MAX_HEIGHT + 1, &mut alloc) },
            len: 0,
            rng: XorShiftRng::new_unseeded(),
            key_func,
            alloc,
        }
    }

    fn gen_random_height(&mut self) -> usize {
        self.rng.next_u32().leading_zeros() as usize
    }

    fn build_prev_nodes_cache(
        &self,
        mut index: usize,
    ) -> [(*mut Node<T, S>, usize, S); MAX_HEIGHT + 1] {
        assert!(index <= self.len);
        let mut curr_node = self.head;
        let mut last_nodes = [(self.head, 0, S::identity()); MAX_HEIGHT + 1];
        unsafe {
            for height in (0..=MAX_HEIGHT).rev() {
                let mut next_link = (*curr_node).get_pointer_mut(height);
                while !next_link.next.is_null() && next_link.next_distance < index {
                    last_nodes[height].1 += next_link.next_distance;
                    last_nodes[height].2 += next_link.sum;
                    index -= next_link.next_distance;

                    curr_node = next_link.next;
                    next_link = (*curr_node).get_pointer_mut(height);
                }
                last_nodes[height].0 = curr_node;
            }
        }
        last_nodes
    }

    /// Inserts a value into the list at a particular index, shifting elements one position to the
    /// right if needed.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// list.insert(0, 1);
    /// list.insert(0, 2);
    /// assert_eq!(list.get(0), Some(&2));
    /// assert_eq!(list.get(1), Some(&1));
    /// ```
    pub fn insert(&mut self, index: usize, value: T) -> Finger<T, S> {
        assert!(index <= self.len);

        self.len += 1;
        let new_height = self.gen_random_height();
        let new_node = Node::new(value, new_height + 1, &mut self.alloc);
        let mut last_nodes = self.build_prev_nodes_cache(index);

        unsafe {
            for height in 0..=new_height {
                let last_node = last_nodes[height].0;
                let last_node_link = (*last_node).get_pointer_mut(height);
                link(
                    &mut *new_node,
                    &mut *last_node_link.next,
                    height,
                    1, // Good default since it will work for layer 0.
                );
                link(
                    &mut *last_node,
                    &mut *new_node,
                    height,
                    last_node_link.next_distance,
                );
            }

            // Since key_fun looks at two nodes, when we insert a node, we need to
            // recalculate its value for both the inserted node, and the node previous to it.
            // In this section we calculate those values, and set the sums correctly for the layer 0
            // of the list.  We also update the last_nodes datastructures to include the updated
            // values of the previous-to-inserted node.
            let new_node_count = {
                let base_layer = (*new_node).get_pointer_mut(0);

                // TODO: This hack says if you insert at the last position, just treat it as if
                // you had a second copy of the same value for computing the key_func.
                let next_val = base_layer
                    .next
                    .as_ref()
                    .map_or(&(*new_node).value, |n| &n.value);
                let new_node_count = (self.key_func)(&(*new_node).value, next_val);
                base_layer.sum = new_node_count;

                // If we are inserting at the first position, we can skip this.
                if base_layer.prev != self.head {
                    let prev_base_layer = (*base_layer.prev).get_pointer_mut(0);

                    let prev_node_count =
                        (self.key_func)(&(*base_layer.prev).value, &(*new_node).value);

                    base_layer.sum = new_node_count;
                    prev_base_layer.sum = prev_node_count;

                    last_nodes[0].1 += 1;
                    last_nodes[0].2 += prev_node_count;
                }

                new_node_count
            };

            for i in 1..=MAX_HEIGHT {
                last_nodes[i].1 += last_nodes[i - 1].1;
                last_nodes[i].2 += last_nodes[i - 1].2;

                let right_distance =
                    1 + (*last_nodes[i].0).get_pointer(i).next_distance - last_nodes[i - 1].1;
                // Note that order matters here, since the group is not necessarily commutative.
                let right_nodes = new_node_count
                    + (last_nodes[i - 1].2).inverse()
                    + (*last_nodes[i].0).get_pointer(i).sum;

                let last_node_link = (*last_nodes[i].0).get_pointer_mut(i);
                if i <= new_height {
                    let new_node_link = (*new_node).get_pointer_mut(i);

                    last_node_link.next_distance = last_nodes[i - 1].1;
                    last_node_link.sum = last_nodes[i - 1].2;

                    new_node_link.next_distance = right_distance;
                    new_node_link.sum = right_nodes;
                } else {
                    last_node_link.next_distance += 1;
                    last_node_link.sum = last_nodes[i - 1].2 + right_nodes;
                }
            }
            Finger(new_node)
        }
    }

    /// Removes a value at a particular index from the list. Returns the value at the index.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// list.insert(0, 1);
    /// assert_eq!(list.remove(0), 1);
    /// ```
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len);

        let mut last_nodes = self.build_prev_nodes_cache(index);

        unsafe {
            let node_to_remove = (*last_nodes[0].0).get_pointer_mut(0).next;
            let following_node = (*node_to_remove).get_pointer(0).next;

            let value_removed = (*node_to_remove).get_pointer(0).sum;

            let left_value_removed = (*last_nodes[0].0).get_pointer_mut(0).sum;
            // Since the key_func looks at two nodes, we have to recompute the
            // value for the previous node.
            let left_value_added = if last_nodes[0].0 != self.head {
                let next_val = following_node
                    .as_ref()
                    .map_or(&(*last_nodes[0].0).value, |n| &n.value);
                (self.key_func)(&(*last_nodes[0].0).value, next_val)
            } else {
                S::identity()
            };

            let next_distance = if last_nodes[0].0 != self.head { 1 } else { 0 };
            link(last_nodes[0].0, following_node, 0, next_distance);

            for i in 1..=MAX_HEIGHT {
                last_nodes[i].1 += last_nodes[i - 1].1;
                last_nodes[i].2 += last_nodes[i - 1].2;

                let last_node_link = (*last_nodes[i].0).get_pointer_mut(i);
                let removed_node_link = (*node_to_remove).get_pointer_mut(i);
                if i < (*node_to_remove).links_len {
                    let distance = last_nodes[i - 1].1
                        + next_distance
                        + (*node_to_remove).get_pointer(i).next_distance
                        - 1;
                    last_node_link.sum = last_nodes[i - 1].2
                        + left_value_added
                        + value_removed.inverse()
                        + (*node_to_remove).get_pointer(i).sum;
                    link(last_nodes[i].0, removed_node_link.next, i, distance);
                } else {
                    last_node_link.next_distance -= 1;
                    // Note that order matters here, since the group is not necessarily commutative.
                    last_node_link.sum = last_nodes[i - 1].2
                        + left_value_added
                        + (last_nodes[i - 1].2 + left_value_removed + value_removed).inverse()
                        + last_node_link.sum;
                }
            }
            self.len -= 1;

            // Frees the node
            Node::extract_value(node_to_remove, &mut self.alloc)
        }
    }

    /// Splits a list at a given node with the default allocator
    pub unsafe fn split_at_finger(&mut self, finger: Finger<T, S>) -> CountedSkipList<T, S, F> {
        self.split_at_finger_in(finger, Global {})
    }

    /// Splits a list at a given Node with a given allocator
    ///
    /// The node must be an element of this skiplist.
    pub unsafe fn split_at_finger_in<B: Alloc>(
        &mut self,
        finger: Finger<T, S>,
        alloc: B,
    ) -> CountedSkipList<T, S, F, B> {
        let mut curr_node = finger.0;
        let mut newlist = CountedSkipList::<T, S, F, B>::new_with_group_in(self.key_func, alloc);
        let mut distance_from_end = 1;
        let mut sum_from_end = (self.key_func)(&(*curr_node).value, &(*curr_node).value);
        for height in 0..=MAX_HEIGHT {
            let (parent, distance, sum) = self.parent_at_height(curr_node, height);
            curr_node = parent;
            distance_from_end += distance;
            sum_from_end = sum + sum_from_end;

            // Split current level, order of link() calls and .sum assignments important here.
            let curr_link = (*curr_node).get_pointer_mut(height);
            link(
                newlist.head,
                curr_link.next,
                height,
                curr_link.next_distance - distance_from_end,
            );
            (*newlist.head).get_pointer_mut(height).sum = sum_from_end.inverse() + curr_link.sum;
            link(curr_node, std::ptr::null_mut(), height, distance_from_end);
            (*curr_node).get_pointer_mut(height).sum = sum_from_end;
        }
        while curr_node != self.head {
            curr_node = (*curr_node).get_pointer_mut(MAX_HEIGHT).prev;
            distance_from_end += (*curr_node).get_pointer_mut(MAX_HEIGHT).next_distance;
        }

        newlist.len = self.len - distance_from_end;
        self.len = distance_from_end;
        newlist
    }

    /// Inserts a value at the front of the list.
    /// 
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// list.push_front(1);
    /// list.push_front(2);
    /// assert_eq!(list.get(0), Some(&2));
    /// ```
    pub fn push_front(&mut self, value: T) -> Finger<T,S> {
        self.insert(0, value)
    }

    /// Inserts a value at the back of the list.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// list.push_back(1);
    /// list.push_back(2);
    /// assert_eq!(list.get(0), Some(&1));
    /// ```
    pub fn push_back(&mut self, value: T) -> Finger<T,S> {
        let index = self.len();
        self.insert(index, value)
    }

    /// Removes a value at the front of the list.
    ///
    /// # Panics
    ///
    /// Panics if list is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// list.push_back(1);
    /// list.push_back(2);
    /// assert_eq!(list.pop_front(), 1);
    /// ```
    pub fn pop_front(&mut self) -> T {
        self.remove(0)
    }

    /// Removes a value at the back of the list.
    ///
    /// # Panics
    ///
    /// Panics if list is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// list.push_back(1);
    /// list.push_back(2);
    /// assert_eq!(list.pop_back(), 2);
    /// ```
    pub fn pop_back(&mut self) -> T {
        let index = self.len() - 1;
        self.remove(index)
    }

    /// Returns a mutable reference to the value at a particular index. Returns `None` if the
    /// index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// list.insert(0, 1);
    /// *list.get_mut(0).unwrap() = 2;
    /// assert_eq!(list.get(0), Some(&2));
    /// ```
    pub fn get(&self, index: usize) -> Option<&T> {
        unsafe { self.get_finger(index).map(|n| &(*n.0).value) }
    }

    /// Returns a mutable reference to the value at a particular index. Returns `None` if the
    /// index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// list.insert(0, 1);
    /// *list.get_mut(0).unwrap() = 2;
    /// assert_eq!(list.get(0), Some(&2));
    /// ```
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        unsafe { self.get_finger(index).map(|n| &mut (*n.0).value) }
    }

    /// Returns a finger reference to the node at a particular index. Returns `None` if the
    /// index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// let finger = list.insert(0, 1);
    /// assert_eq!(list.get_finger(0), Some(finger));
    /// ```
    pub fn get_finger(&self, mut index: usize) -> Option<Finger<T, S>> {
        let mut curr_height = MAX_HEIGHT;
        let mut curr_node = self.head;

        unsafe {
            loop {
                let mut next_link = (*curr_node).get_pointer(curr_height);
                while !next_link.next.is_null() && next_link.next_distance <= index {
                    index -= next_link.next_distance;
                    let next_next_link = (*next_link.next).get_pointer(curr_height);

                    curr_node = mem::replace(&mut next_link, next_next_link).next;

                    if index == 0 {
                        return Some(Finger(curr_node));
                    }
                }

                if curr_height == 0 {
                    return None;
                }

                curr_height -= 1;
            }
        }
    }

    /// Finds the first node to the left of a given node that
    /// has a height greater than or equal to the given height,
    /// summing up the total distance and sum traversed.
    ///
    /// If the node is already at the given height, it will just
    /// return the given node.
    unsafe fn parent_at_height(
        &self,
        node: *mut Node<T, S>,
        height: usize,
    ) -> (*mut Node<T, S>, usize, S) {
        assert!(height <= MAX_HEIGHT);

        let mut curr_node = node;
        let mut distance_traversed = 0;
        let mut sum_traversed = S::identity();

        if node.is_null() {
            return (node, distance_traversed, sum_traversed);
        }

        while (*curr_node).links_len <= height {
            curr_node = (*curr_node).get_pointer(height - 1).prev;
            let curr_link = (*curr_node).get_pointer(height - 1);
            distance_traversed += curr_link.next_distance;
            sum_traversed = curr_link.sum + sum_traversed;
        }
        (curr_node, distance_traversed, sum_traversed)
    }

    /// Finds the first node to the right of a given node that
    /// has a height greater than or equal to the given height,
    /// summing up the total distance and sum traversed.
    ///    
    /// If the node is already at the given height, it will just
    /// return the given node.
    #[allow(dead_code)]
    unsafe fn right_parent_at_height(
        &self,
        node: *mut Node<T, S>,
        height: usize,
    ) -> (*mut Node<T, S>, usize, S) {
        assert!(height <= MAX_HEIGHT);
        let mut curr_node = node;
        let mut distance_traversed = 0;
        let mut sum_traversed = S::identity();
        while !curr_node.is_null() && (*curr_node).links_len <= height {
            let curr_link = (*curr_node).get_pointer(height - 1);
            curr_node = curr_link.prev;
            distance_traversed += curr_link.next_distance;
            sum_traversed = curr_link.sum + sum_traversed;
        }
        (curr_node, distance_traversed, sum_traversed)
    }

    /// Finds the first node to the left of a given node that
    /// has a greater height, summing up the total distance and
    /// sum traversed.
    #[allow(dead_code)]
    unsafe fn parent(&self, node: *mut Node<T, S>) -> (*mut Node<T, S>, usize, S) {
        self.parent_at_height(node, (*node).links_len)
    }

    /// Finds the first node to the left of a given node that
    /// has a greater height, summing up the total distance and
    /// sum traversed.
    #[allow(dead_code)]
    unsafe fn right_parent(&self, node: *mut Node<T, S>) -> (*mut Node<T, S>, usize, S) {
        self.parent_at_height(node, (*node).links_len)
    }

    /// Returns the distance and sum between two fingers.
    ///
    /// i.e. the sum on the half open interval [a,b).
    /// It will return a negative sum if b is before a
    /// in the list.
    ///
    /// Undefined behavior if a or b is not a part of
    /// this list.
    pub unsafe fn finger_difference(&self, a: Finger<T, S>, b: Finger<T, S>) -> (i64, S) {
        // The algorithm we use is we walk left from a and b until we reach
        // a node at the next height, repeating up to MAX_HEIGHT, or the nodes
        // coincide.  If the nodes coincide, then we can tell whether a is
        // before or after b based on the number of steps taken at the previous
        // level.  If we reach MAX_HEIGHT without meeting, we just do a linear
        // search at the MAX_HEIGHt level.
        let mut curr_a = a.0;
        let mut curr_b = b.0;
        let mut a_distance: i64 = 0;
        let mut b_distance: i64 = 0;
        let mut a_sum = S::identity();
        let mut b_sum = S::identity();

        if curr_a == curr_b {
            return (0, S::identity());
        }

        for height in 1..=MAX_HEIGHT {
            let (new_a, a_distance_step, a_sum_step) = self.parent_at_height(curr_a, height);
            let (new_b, b_distance_step, b_sum_step) = self.parent_at_height(curr_b, height);
            curr_a = new_a;
            curr_b = new_b;
            a_distance += a_distance_step as i64;
            b_distance += b_distance_step as i64;
            a_sum = a_sum_step + a_sum;
            b_sum = b_sum_step + b_sum;
            if curr_a == curr_b {
                return (b_distance - a_distance, a_sum.inverse() + b_sum);
            }
        }

        // If we reach here, we have got to the top level, without finding a
        // common parent.  We will search to the right starting at node a
        // and if that fails, we will search to the right starting at node b.

        // We store the state once we reached the top, in case our search from
        // node a fails.
        let (top_a, top_a_distance, top_a_sum) = (curr_a, a_distance, a_sum);

        loop {
            if curr_a.is_null() {
                break;
            } else if curr_a == curr_b {
                return (b_distance - a_distance, a_sum.inverse() + b_sum);
            }
            let link = (*curr_a).get_pointer(MAX_HEIGHT);
            curr_a = link.next;
            a_distance -= link.next_distance as i64;
            a_sum = link.sum.inverse() + a_sum;
        }

        // If we reach here, then that means a is to the right of b, so
        // we now search right from b.

        loop {
            if curr_b.is_null() {
                break;
            } else if top_a == curr_b {
                return (b_distance - top_a_distance, top_a_sum.inverse() + b_sum);
            }
            let link = (*curr_b).get_pointer(MAX_HEIGHT);
            curr_b = link.next;
            b_distance -= link.next_distance as i64;
            b_sum = link.sum.inverse() + b_sum;
        }
        panic!("Cannot take difference between a and b which are not from the same list.")
    }

    /// Returns the number of elements in the list.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// list.insert(0, 1);
    /// assert_eq!(list.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the list is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let list: CountedSkipList<u32> = CountedSkipList::new();
    /// assert!(list.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clears the list, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// list.insert(0, 1);
    /// list.insert(1, 2);
    /// list.clear();
    /// assert_eq!(list.is_empty(), true);
    /// ```
    pub fn clear(&mut self) {
        self.len = 0;
        unsafe {
            let mut curr_node = (*self.head).get_pointer(0).next;
            while !curr_node.is_null() {
                let next_node = (*curr_node).get_pointer(0).next;
                Node::free(mem::replace(&mut curr_node, next_node), &mut self.alloc);
            }
            for height in 0..=MAX_HEIGHT {
                *(*self.head).links.get_unchecked_mut(height) = Layer {
                    sum: S::identity(),
                    next: std::ptr::null_mut(),
                    next_distance: 0,
                    prev: std::ptr::null_mut(),
                }
            }
        }
    }

    /// Returns an iterator over the list.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// list.insert(0, 1);
    /// list.insert(1, 2);
    ///
    /// let mut iterator = list.iter();
    /// assert_eq!(iterator.next(), Some(&1));
    /// assert_eq!(iterator.next(), Some(&2));
    /// assert_eq!(iterator.next(), None);
    /// ```
    pub fn iter(&self) -> CountedSkipListIter<'_, T, S> {
        unsafe {
            CountedSkipListIter {
                current: &(*self.head).get_pointer(0).next,
            }
        }
    }

    /// Returns a mutable iterator over the list.
    ///
    /// # Examples
    ///
    /// ```
    /// use counted_skiplist::CountedSkipList;
    ///
    /// let mut list = CountedSkipList::new();
    /// list.insert(0, 1);
    /// list.insert(1, 2);
    ///
    /// for value in &mut list {
    ///     *value += 1;
    /// }
    ///
    /// let mut iterator = list.iter();
    /// assert_eq!(iterator.next(), Some(&2));
    /// assert_eq!(iterator.next(), Some(&3));
    /// assert_eq!(iterator.next(), None);
    /// ```
    pub fn iter_mut(&mut self) -> CountedSkipListIterMut<'_, T, S> {
        unsafe {
            CountedSkipListIterMut {
                current: &mut (*self.head).get_pointer_mut(0).next,
            }
        }
    }
}

impl<T, S: AdditiveGroup + Copy, F: Fn(&T, &T) -> S + Copy, A: Alloc> Drop for CountedSkipList<T, S, F, A> {
    fn drop(&mut self) {
        unsafe {
            let next_node = (*self.head).get_pointer(0).next;
            Node::free(mem::replace(&mut self.head, next_node), &mut self.alloc);
            while !self.head.is_null() {
                let next_node = (*self.head).get_pointer(0).next;
                Node::free(mem::replace(&mut self.head, next_node), &mut self.alloc);
            }
        }
    }
}

impl<T, S: AdditiveGroup + Copy, F: Fn(&T, &T) -> S + Copy, A: Alloc + Default> IntoIterator
    for CountedSkipList<T, S, F, A>
{
    type IntoIter = CountedSkipListIntoIter<T, S, A>;
    type Item = T;

    fn into_iter(mut self) -> Self::IntoIter {
        unsafe {
            let alloc = std::mem::replace(&mut self.alloc, A::default());
            let ret = Self::IntoIter {
                current: (*self.head).links.get_unchecked_mut(0).next,
                alloc,
            };
            ptr::write_bytes((*self.head).links.get_unchecked_mut(0), 0, MAX_HEIGHT + 1);
            ret
        }
    }
}

impl<'a, T, S: AdditiveGroup + Copy, F: Fn(&T, &T) -> S + Copy, A: Alloc> IntoIterator
    for &'a CountedSkipList<T, S, F, A>
where
    T: 'a,
{
    type IntoIter = CountedSkipListIter<'a, T, S>;
    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, S: AdditiveGroup + Copy, F: Fn(&T, &T) -> S + Copy, A: Alloc> IntoIterator
    for &'a mut CountedSkipList<T, S, F, A>
where
    T: 'a,
{
    type IntoIter = CountedSkipListIterMut<'a, T, S>;
    type Item = &'a mut T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// An owning iterator for `CountedSkipList<T,S>`.
///
/// This iterator traverses the elements of the list and yields owned entries.
pub struct CountedSkipListIntoIter<T, S: AdditiveGroup + Copy, A: Alloc> {
    current: *mut Node<T, S>,
    alloc: A,
}

impl<T, S: AdditiveGroup + Copy, A: Alloc> Iterator for CountedSkipListIntoIter<T, S, A> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            None
        } else {
            unsafe {
                let ret = ptr::read(&(*self.current).value);
                let next_node = (*self.current).get_pointer(0).next;
                Node::deallocate(mem::replace(&mut self.current, next_node), &mut self.alloc);
                Some(ret)
            }
        }
    }
}

impl<T, S: AdditiveGroup + Copy, A: Alloc> Drop for CountedSkipListIntoIter<T, S, A> {
    fn drop(&mut self) {
        unsafe {
            while !self.current.is_null() {
                ptr::drop_in_place(&mut (*self.current).value);
                let next_node = (*self.current).get_pointer(0).next;
                Node::deallocate(mem::replace(&mut self.current, next_node), &mut self.alloc);
            }
        }
    }
}

/// An iterator for `CountedSkipList<T,S>`.
///
/// This iterator traverses the elements of the list in-order and yields immutable references.
pub struct CountedSkipListIter<'a, T, S: AdditiveGroup + Copy> {
    current: &'a *mut Node<T, S>,
}

impl<'a, T, S: AdditiveGroup + Copy> Iterator for CountedSkipListIter<'a, T, S>
where
    T: 'a,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            None
        } else {
            unsafe {
                let ret = &(**self.current).value;
                let next_node = &(**self.current).get_pointer(0).next;
                mem::replace(&mut self.current, next_node);
                Some(ret)
            }
        }
    }
}

/// A mutable iterator for `CountedSkipList<T,S>`.
///
/// This iterator traverses the elements of the list in-order and yields mutable references.
pub struct CountedSkipListIterMut<'a, T, S: AdditiveGroup + Copy> {
    current: &'a mut *mut Node<T, S>,
}

impl<'a, T, S: AdditiveGroup + Copy> Iterator for CountedSkipListIterMut<'a, T, S>
where
    T: 'a,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            None
        } else {
            unsafe {
                let ret = &mut (**self.current).value;
                let next_node = &mut (**self.current).get_pointer_mut(0).next;
                mem::replace(&mut self.current, next_node);
                Some(ret)
            }
        }
    }
}

impl<T> Default for CountedSkipList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, S: AdditiveGroup + Copy, F: Fn(&T, &T) -> S + Copy, A: Alloc> Add for CountedSkipList<T, S, F, A> {
    type Output = CountedSkipList<T, S, F, A>;

    fn add(mut self, other: CountedSkipList<T, S, F, A>) -> CountedSkipList<T, S, F, A> {
        self.len += other.len();

        let mut curr_nodes = [self.head; MAX_HEIGHT + 1];
        unsafe {
            let mut curr_height = MAX_HEIGHT;
            let mut curr_node = self.head;
            while !curr_node.is_null() {
                while (*curr_node).get_pointer(curr_height).next.is_null() {
                    curr_nodes[curr_height] = curr_node;
                    if curr_height == 0 {
                        break;
                    }
                    curr_height -= 1;
                }
                curr_node = (*curr_node).get_pointer(curr_height).next;
            }

            for (i, curr_node) in curr_nodes.iter_mut().enumerate().take(MAX_HEIGHT + 1) {
                let other_link = (*other.head).get_pointer_mut(i);
                let distance =
                    (**curr_node).get_pointer_mut(i).next_distance + other_link.next_distance;
                let sum = (**curr_node).get_pointer_mut(i).sum + other_link.sum;
                swap_link(*curr_node, other.head, i);
                (**curr_node).get_pointer_mut(i).next_distance = distance;
                (**curr_node).get_pointer_mut(i).sum = sum;
            }
        }
        self
    }
}

impl<T, S: AdditiveGroup + Copy, F: Fn(&T, &T) -> S + Copy, A: Alloc> Index<usize>
    for CountedSkipList<T, S, F, A>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("Error: index out of bounds.")
    }
}

impl<T, S: AdditiveGroup + Copy, F: Fn(&T, &T) -> S + Copy, A: Alloc> IndexMut<usize>
    for CountedSkipList<T, S, F, A>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("Error: index out of bounds.")
    }
}

#[derive(Debug)]
struct LinkRep<'a, T: Debug, S: Debug> {
    prev: Option<&'a T>,
    next: Option<&'a T>,
    next_distance: usize,
    sum: S,
}

impl<T: Debug + Clone + Default, S: AdditiveGroup + Copy + Debug, F: Fn(&T, &T) -> S + Copy, A: Alloc> Debug
    for CountedSkipList<T, S, F, A>
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        unsafe {
            let mut curr_node = (*self.head).get_pointer(0).next;
            let mut actual = vec![];
            while !curr_node.is_null() {
                actual.push(&(*curr_node).value);
                let next_link = (*curr_node).get_pointer(0);
                curr_node = next_link.next;
            }
            curr_node = (*self.head).get_pointer(0).next;
            writeln!(f, "[")?;
            while !curr_node.is_null() {
                let links =
                    std::slice::from_raw_parts((*curr_node).get_pointer(0), (*curr_node).links_len);
                let links_rep: Vec<_> = links
                    .iter()
                    .map(|l| {
                        LinkRep {
                            prev: l.prev.as_ref().map(|n| &n.value),
                            next: l.next.as_ref().map(|n| &n.value),
                            next_distance: l.next_distance,
                            sum: l.sum,
                        }
                    })
                    .collect();
                let mut builder = f.debug_struct("Node");
                let _ = builder.field("value", &(*curr_node).value);
                let _ = builder.field("links", &links_rep);
                builder.finish()?;
                writeln!(f, ",")?;
                let next_link = (*curr_node).get_pointer(0);
                curr_node = next_link.next;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}



    /// Checks the internal validity of a CountedSkipList.
    pub fn check_valid<T, S: AdditiveGroup + Copy, F: Fn(&T, &T) -> S + Copy, A: Alloc>(
        list: &CountedSkipList<T, S, F, A>,
    ) where
        T: PartialEq + Debug + Clone + Default,
        S: Debug
    {
        //println!("check_valid()");
        //println!("{:#?}", list);
        unsafe {
            let mut length = 0;
            let mut expected_distances : Vec<usize> = (0..=MAX_HEIGHT).map(|i| (*list.head).get_pointer(i).next_distance).collect();
            let mut expected_sums : Vec<S> =  (0..=MAX_HEIGHT).map(|i| (*list.head).get_pointer(i).sum).collect();
            let mut distances = [0; MAX_HEIGHT+1];
            let mut sums = [S::identity(); MAX_HEIGHT+1];

            assert_eq!((*list.head).get_pointer(0).next_distance, 0);

            let mut curr_node = (*list.head).get_pointer(0).next;
            while !curr_node.is_null() {
                let next_node = (*curr_node).get_pointer(0).next;

                let next_value = &next_node.as_ref().unwrap_or(&*curr_node).value;
                let sum = (list.key_func)(&(*curr_node).value, next_value);

                let height = (*curr_node).links_len;
                for i in 0..height {
                    let skip_node = (*curr_node).get_pointer(i).next;
                    if !skip_node.is_null() {
                        assert_eq!((*skip_node).get_pointer(i).prev, curr_node);
                    }
                    assert_eq!(distances[i], expected_distances[i]);
                    assert_eq!(sums[i], expected_sums[i]);
                    
                    let link = (*curr_node).get_pointer(i);
                    expected_distances[i] = link.next_distance;
                    expected_sums[i] = link.sum;

                    distances[i] = 1;
                    sums[i] = sum;
                }
                for i in height..=MAX_HEIGHT {
                        distances[i] += 1;
                        sums[i] += sum;
                }

                length +=1;
                curr_node = next_node;
            }
            for i in 0..=MAX_HEIGHT {
                assert_eq!(distances[i], expected_distances[i]);
                assert_eq!(sums[i], expected_sums[i]);
            }
            assert_eq!(list.len, length);
        }
    }

    #[test]
    fn test_len_empty() {
        let list: CountedSkipList<u32> = CountedSkipList::new();
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_is_empty() {
        let list: CountedSkipList<u32> = CountedSkipList::new();
        check_valid(&list);
        assert!(list.is_empty());
    }

    #[test]
    fn test_insert() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        check_valid(&list);
        list.insert(0, 1);

        check_valid(&list);
        assert_eq!(list.get(0), Some(&1));
    }

    #[test]
    fn test_insert_order() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.insert(0, 2);
        list.insert(0, 3);
        check_valid(&list);
        assert_eq!(list.iter().collect::<Vec<&u32>>(), vec![&3, &2, &1],);
    }

    #[test]
    fn test_insert_order2() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.insert(0, 2);
        check_valid(&list);
        assert_eq!(list.iter().collect::<Vec<&u32>>(), vec![&2, &1],);
    }

    #[test]
    fn test_insert_order3() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.insert(0, 2);
        list.insert(1, 3);
        check_valid(&list);
        assert_eq!(list.iter().collect::<Vec<&u32>>(), vec![&2, &3, &1],);
    }

    #[test]
    fn test_remove() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        let ret = list.remove(0);

        check_valid(&list);
        assert_eq!(list.get(0), None);
        assert_eq!(ret, 1);
    }

    #[test]
    fn test_remove_two() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.insert(0, 2);
        let ret = list.remove(1);

        check_valid(&list);
        assert_eq!(list.get(0), Some(&2));
        assert_eq!(ret, 1);
    }



    #[test]
    fn test_remove_fuzzed() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.insert(0, 2);

        check_valid(&list);
        let ret = list.remove(1);
        check_valid(&list);
        assert_eq!(ret, 1);
    }

    #[test]
    fn test_remove_fuzzed2() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.insert(0, 2);
        list.insert(2, 3);
        list.insert(0, 5);

        check_valid(&list);
        let ret = list.remove(0);
        check_valid(&list);
        assert_eq!(ret, 5);
    }

    #[test]
    fn test_get_mut() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        {
            let value = list.get_mut(0);
            *value.unwrap() = 3;
        }
        assert_eq!(list.get(0), Some(&3));
    }

    #[test]
    fn test_push_front() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.push_front(2);

        check_valid(&list);
        assert_eq!(list.get(0), Some(&2));
    }

    #[test]
    fn test_push_back() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.push_back(2);

        check_valid(&list);
        assert_eq!(list.get(1), Some(&2));
    }

    #[test]
    fn test_pop_front() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.insert(1, 2);

        check_valid(&list);
        assert_eq!(list.pop_front(), 1);
    }

    #[test]
    fn test_pop_back() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.insert(1, 2);
        assert_eq!(list.pop_back(), 2);
    }

    #[test]
    fn test_add() {
        let mut n: CountedSkipList<u32> = CountedSkipList::new();
        n.insert(0, 1);
        n.insert(0, 2);
        n.insert(1, 3);

        let mut m: CountedSkipList<u32> = CountedSkipList::new();
        m.insert(0, 4);
        m.insert(0, 5);
        m.insert(1, 6);

        check_valid(&n);
        check_valid(&m);

        let res = n + m;

        check_valid(&res);
        assert_eq!(
            res.iter().collect::<Vec<&u32>>(),
            vec![&2, &3, &1, &5, &6, &4],
        );
        assert_eq!(res.len(), 6);
    }

    #[test]
    fn test_split_at_finger() {
        let mut l: CountedSkipList<u32> = CountedSkipList::new();
        l.insert(0, 1);
        l.insert(0, 2);
        let finger = l.insert(0, 3);
        l.insert(0, 4);

        check_valid(&l);
        let m = unsafe { l.split_at_finger(finger) };
        check_valid(&m);
        check_valid(&l);
    }

    #[test]
    fn test_add_fuzzed() {
        let mut n: CountedSkipList<u32> = CountedSkipList::new();
        n.insert(0, 1);
        n.insert(0, 2);

        let mut m: CountedSkipList<u32> = CountedSkipList::new();
        m.insert(0, 4);
        m.insert(0, 5);
        m.remove(1);

        check_valid(&n);
        check_valid(&m);

        let res = m + n;

        check_valid(&res);
        assert_eq!(res.iter().collect::<Vec<&u32>>(), vec![&5, &2, &1],);
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn test_split_fuzzed() {
        let mut m: CountedSkipList<u32> = CountedSkipList::new();
        let finger = m.insert(0, 1);

        let n = unsafe { m.split_at_finger(finger) };
        check_valid(&n);

        m.insert(0, 2);
        check_valid(&m);

        let res = m + n;
        check_valid(&res);
    }

    #[test]
    fn test_split_fuzzed2() {
        let mut m: CountedSkipList<u32> = CountedSkipList::new();
        m.insert(0, 1);
        let finger = m.insert(0, 2);

        let n = unsafe { m.split_at_finger(finger) };
        check_valid(&m);
        check_valid(&n);

        let res = m + n;
        check_valid(&res);
    }

    #[test]
    fn test_split_fuzzed3() {
        let mut m: CountedSkipList<u32> = CountedSkipList::new();
        m.insert(0, 1);
        m.insert(0, 2);
        m.insert(0, 3);
        let finger = m.insert(2, 4);
        check_valid(&m);
        let mut n = unsafe { m.split_at_finger(finger) };
        check_valid(&n);
        n.insert(1, 5);
        check_valid(&n);
    }

    #[test]
    fn test_finger_difference() {
        let mut m = CountedSkipList::new_with_group(|&x, &_y| x as i32);
        m.insert(0, 1);
        let finger1 = m.insert(1, 2);
        m.insert(2, 3);
        let finger2 = m.insert(3, 4);

        let (distance, _) = unsafe { m.finger_difference(finger1, finger2) };
        let (rev_distance, _) = unsafe { m.finger_difference(finger2, finger1) };
        assert_eq!(distance, 2);
        assert_eq!(rev_distance, -2);
    }

    #[test]
    fn test_subsum() {
        let mut list = CountedSkipList::new_with_group(|&x, &_y| x as i32);
         list.push_back(1);
        let finger1 = list.push_back(2);
        list.push_back(3);
        list.push_back(4);
        let finger2 =list.push_back(5);
        list.push_back(6);
        let (_distance, sum) = unsafe { list.finger_difference(finger1, finger2) };
        assert_eq!(sum, 2+3+4);
    }


    #[test]
    fn test_subsum_backwards() {
        let mut m  = CountedSkipList::new_with_group(|&x, &_y| x as i32);
        let finger1 = m.insert(0, 1);
        let finger2 = m.insert(1, 2);
        check_valid(&m);
        let (distance, sum) = unsafe { m.finger_difference(finger2, finger1) };
        assert_eq!(distance, -1);
        assert_eq!(sum, -1);
    }

    #[test]
    fn test_into_iter() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.insert(0, 2);
        list.insert(1, 3);

        check_valid(&list);
        assert_eq!(list.into_iter().collect::<Vec<u32>>(), vec![2, 3, 1]);
    }

    #[test]
    fn test_iter() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.insert(0, 2);
        list.insert(1, 3);

        check_valid(&list);
        assert_eq!(list.iter().collect::<Vec<&u32>>(), vec![&2, &3, &1]);
    }

    #[test]
    fn test_iter_mut() {
        let mut list: CountedSkipList<u32> = CountedSkipList::new();
        list.insert(0, 1);
        list.insert(0, 2);
        list.insert(1, 3);

        for value in &mut list {
            *value += 1;
        }

        check_valid(&list);
        assert_eq!(list.iter().collect::<Vec<&u32>>(), vec![&3, &4, &2]);
    }