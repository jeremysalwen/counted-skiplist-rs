# Counted-Skiplist

This is an implmentation of a Counted Skiplist (the Skiplist equivalent of a [Counted B-Tree](http://msinfo.info/msi/cdsisis/basico/countedb-trees.htm)), optimized for solving Traveling Salesman Problems.


The code is based on the [extended-collections-rs](https://github.com/jeffrey-xiao/extended-collections-rs) skiplist implementation.

The actual cargo package is under `third_party/counted-skiplist-rs` due to Google's open source packaging requirements.

This is not an officially supported Google product.

## Usage

Add this to your `Cargo.toml`:
```toml
[dependencies]
counted-skiplist = "*"
```
and this to your crate root:
```rust
extern crate counted_skiplist;
```

## License

`counted-skiplist-rs` is licensed under the terms of the Apache
License (Version 2.0).

See [LICENSE](LICENSE) for more details.
