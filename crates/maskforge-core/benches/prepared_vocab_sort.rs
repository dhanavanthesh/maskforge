//! Compares indirect and prefix-decorated vocabulary sort strategies.

use std::hint::black_box;
use std::time::Instant;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_byte(&mut self) -> u8 {
        (self.next_u64() & 0xff) as u8
    }

    fn range(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

fn synthetic_tokens(count: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = Rng(seed);
    let stems: Vec<Vec<u8>> = (0..64)
        .map(|i| {
            let len = 3 + (i % 6);
            (0..len).map(|_| rng.next_byte() & 0x7f).collect()
        })
        .collect();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if rng.range(4) == 0 {
            let len = 1 + rng.range(2);
            out.push((0..len).map(|_| rng.next_byte()).collect());
        } else {
            let mut t = stems[rng.range(stems.len())].clone();
            let suffix_len = rng.range(5);
            for _ in 0..suffix_len {
                t.push(rng.next_byte() & 0x7f);
            }
            out.push(t);
        }
    }
    out
}

fn sort_indirect(tokens: &[Vec<u8>]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..tokens.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| tokens[a as usize].cmp(&tokens[b as usize]));
    order
}

#[derive(Copy, Clone)]
struct PrefixKey {
    prefix: u64,
    index: u32,
}

fn lexical_prefix(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = bytes.len().min(8);
    buf[..n].copy_from_slice(&bytes[..n]);
    u64::from_be_bytes(buf)
}

fn sort_decorated_prefix(tokens: &[Vec<u8>]) -> Vec<u32> {
    let mut order: Vec<PrefixKey> = (0..tokens.len() as u32)
        .map(|i| PrefixKey {
            prefix: lexical_prefix(&tokens[i as usize]),
            index: i,
        })
        .collect();
    order.sort_unstable_by(|a, b| {
        a.prefix
            .cmp(&b.prefix)
            .then_with(|| tokens[a.index as usize].cmp(&tokens[b.index as usize]))
    });
    order.into_iter().map(|k| k.index).collect()
}

fn median_us(mut samples: Vec<u128>) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2] as f64 / 1000.0
}

fn time_variant(tokens: &[Vec<u8>], iters: usize, sort_fn: impl Fn(&[Vec<u8>]) -> Vec<u32>) -> f64 {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let order = sort_fn(black_box(tokens));
        samples.push(t0.elapsed().as_nanos());
        black_box(order);
    }
    median_us(samples)
}

fn main() {
    println!("== PREPARED-VOCABULARY SORT COMPARATOR (reported, not gated) ==");
    println!("synthetic BPE-scale vocabularies; median of 10 sorts per variant per scale");
    let scales: [(&str, usize, usize); 3] = [
        ("gpt2 (~50k)", 50_257, 10),
        ("qwen2.5 (~152k)", 151_936, 10),
        ("bloom (~251k)", 250_880, 10),
    ];
    for (tag, count, iters) in scales {
        let tokens = synthetic_tokens(count, 0xD1B5_4A32_D192_ED03 ^ count as u64);
        let indirect_us = time_variant(&tokens, iters, sort_indirect);
        let decorated_us = time_variant(&tokens, iters, sort_decorated_prefix);
        let delta_pct = (decorated_us - indirect_us) / indirect_us * 100.0;
        println!(
            "{tag:>18}: indirect={indirect_us:>9.1}us  decorated_prefix={decorated_us:>9.1}us  \
             delta={delta_pct:+.1}%"
        );

        let a = sort_indirect(&tokens);
        let b = sort_decorated_prefix(&tokens);
        let a_keys: Vec<&[u8]> = a.iter().map(|&i| tokens[i as usize].as_slice()).collect();
        let b_keys: Vec<&[u8]> = b.iter().map(|&i| tokens[i as usize].as_slice()).collect();
        assert_eq!(
            a_keys, b_keys,
            "{tag}: decorated-prefix order diverged from indirect order"
        );
    }
    println!("== END ==");
}
