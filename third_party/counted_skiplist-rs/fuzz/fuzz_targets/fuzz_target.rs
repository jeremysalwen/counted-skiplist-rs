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

#![no_main]
#[macro_use]
extern crate libfuzzer_sys;
#[macro_use]
extern crate log;
extern crate arbitrary;
extern crate counted_skiplist;

use arbitrary::*;
use counted_skiplist::TspList;
use counted_skiplist::Finger;

fn fuzztest(data: &[u8]) -> Result<(), <arbitrary::FiniteBuffer<'_> as Unstructured>::Error> {
    let _ = env_logger::try_init();
    let mut buff = FiniteBuffer::new(data, 4048).unwrap();
    let closure = |&x: &_, &_y: &_| x as i32;
    let mut list = TspList::new_with_monoid(closure);
    let mut list2 = TspList::new_with_monoid(closure);
    let mut model = Vec::<u8>::new();
    let mut model2 = Vec::<u8>::new();
    let mut finger1 :Option<Finger<u8>> = None;
    let mut finger2 : Option<Finger<u8>> = None;
    let mut modelfinger1 : Option<usize> = None;
    let mut modelfinger2 : Option<usize> = None;

    let iterations = buff.container_size()?;
    let mut val =0;
    for _ in 0..iterations {
        let size: u8 = model.len() as u8;
        let op: u8 = Arbitrary::arbitrary(&mut buff)?;
        match op % 8 {
            0 => {
                let ind8: u8 = Arbitrary::arbitrary(&mut buff)?;

                let keepfinger:u8 = Arbitrary::arbitrary(&mut buff)?;
                let ind = ind8 % (size + 1);
                debug!("insert {} {} (kept finger {})", ind, val, keepfinger != 0);
                let fngr = list.insert(ind.into(), val);
                model.insert(ind.into(), val);
                val = val + 1;
                
                if keepfinger != 0 {
                    finger2 = finger1;
                    finger1 = Some(fngr);
                    modelfinger2 = modelfinger1;
                    modelfinger1 = Some(ind as usize);
                } else {
                    if let Some(mf) = modelfinger1 {
                        if mf >= ind as usize {
                            modelfinger1 = Some(mf + 1);
                        }
                    }
                    if let Some(mf) = modelfinger2 {
                        if mf >= ind as usize {
                            modelfinger2 = Some(mf + 1);
                        }
                    }
                }
            },
            1 => {
                if size > 0 {
                    let ind8: u8 = Arbitrary::arbitrary(&mut buff)?;
                    let ind = ind8 % size;
                    debug!("get {}", ind);
                    assert_eq!(
                        *list.get(ind.into()).unwrap(),
                        model[ind as usize]
                    );
                }
            },
            2 => {
                if size > 0 {
                    let ind8: u8 = Arbitrary::arbitrary(&mut buff)?;
                    let ind = ind8 % size;
                    debug!("remove {}", ind);
                    assert_eq!(list.remove(ind.into()), model.remove(ind.into()));

                    if let Some(mf) = modelfinger1 {
                        if ind as usize == mf {
                            finger1 = None;
                            modelfinger1 = None;
                        } else if mf > ind as usize { 
                           modelfinger1 = Some(mf - 1);
                        }
                    }

                    if let Some(mf) = modelfinger2 {
                        if ind as usize == mf {
                            finger2 = None;
                            modelfinger2 = None;
                        } else if mf > ind as usize { 
                            modelfinger2 = Some(mf - 1);
                        }
                    }
                }
            }
            3 => {
                debug!("swap");
                std::mem::swap(&mut list, &mut list2);
                std::mem::swap(&mut model, &mut model2);

                finger1 = None;
                finger2 = None;
                modelfinger1 = None;
                modelfinger2 = None;
            }
            4 => {
                debug!("append");
                list = list + std::mem::replace(&mut list2, TspList::new_with_monoid(closure));
                model.append(&mut model2);
            }
            5 => {
                debug!("clear");
                list.clear();
                model.clear();
                finger1 = None;
                finger2 = None;

                modelfinger1 = None;
                modelfinger2 = None;
            }
            6 => {
                if let Some(mf) = modelfinger1 {
                    debug!("split");
                    list2 = unsafe {list.split_at_finger(finger1.unwrap())};
                    model2 = model.split_off(mf+1);
                    finger1 = None;
                    finger2 = None;
                    modelfinger1 = None;
                    modelfinger2 = None;
                }
            }
            7 => {
                if size > 0 {
                    let ind18: u8 = Arbitrary::arbitrary(&mut buff)?;
                    let ind1 = ind18 % size;
                    let ind28: u8 = Arbitrary::arbitrary(&mut buff)?;
                    let ind2 = ind28 % size;
                    debug!("Compare finger {} {}", ind1, ind2);
                    let finger1 = list.get_finger(ind1.into()).unwrap();
                    let finger2 = list.get_finger(ind2.into()).unwrap();
                    let (distance, sum) = unsafe {list.finger_difference(finger1,finger2)};
                    assert_eq!(distance,
                               ind2 as i64 - ind1 as i64);
                    let mut modelsum =  model[std::cmp::min(ind1,ind2) as usize..std::cmp::max(ind1, ind2) as  usize].iter().map(|&n| n as i32).sum();

                    if ind1 > ind2 {
                        modelsum *= -1;
                    };
                    assert_eq!(sum, modelsum);
                }
            }
            _ => panic!("Invalid op!"),
        }
            counted_skiplist::check_valid(&list);
    }
    Ok(())
}

fuzz_target!(|data: &[u8]| {
    fuzztest(data).ok();
});
