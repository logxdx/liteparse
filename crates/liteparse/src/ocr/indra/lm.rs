//! Token n-gram language model and CTC prefix beam search for lines the
//! recognizer is unsure of (port of Indra's `lm.py`).
//!
//! score = log P_ctc + alpha * log P_lm + beta * tokens. `lm*.npz` holds a
//! backoff table per history length n: sorted keys of (history, token) pairs
//! with log-probabilities, and sorted history keys with backoff weights. A key
//! packs tokens `bits` bits each (0 = line start, t + 1 = token t).

use std::collections::HashMap;

use super::npz;
use crate::error::LiteParseError;

const BITS: u32 = 11;
const TAG_SHIFT: u32 = 61;

pub struct NgramLm {
    order: usize,
    vocab_size: usize,
    bits: u32,
    k: Vec<Vec<u64>>,
    v: Vec<Vec<f32>>,
    hk: Vec<Vec<u64>>,
    hb: Vec<Vec<f32>>,
    max_logp: f64,
}

/// Sort a key/value table by key (the files are sorted; this only guards).
fn sorted(keys: Vec<u64>, vals: Vec<f32>) -> (Vec<u64>, Vec<f32>) {
    if keys.windows(2).all(|w| w[0] <= w[1]) {
        return (keys, vals);
    }
    let mut pairs: Vec<(u64, f32)> = keys.into_iter().zip(vals).collect();
    pairs.sort_by_key(|p| p.0);
    pairs.into_iter().unzip()
}

fn find(keys: &[u64], key: u64) -> Option<usize> {
    keys.binary_search(&key).ok()
}

impl NgramLm {
    pub fn from_npz(bytes: &[u8]) -> Result<Self, LiteParseError> {
        let z = npz::read(bytes)?;
        let get = |name: &str| {
            z.get(name)
                .ok_or_else(|| LiteParseError::Ocr(format!("indra lm .npz: no {name}")))
        };
        let order = get("order")?.scalar_int()? as usize;
        let vocab_size = get("vocab_size")?.scalar_int()? as usize;
        let bits = match z.get("bits") {
            Some(b) => b.scalar_int()? as u32,
            None => BITS,
        };
        if order == 0 || order as u32 * bits > TAG_SHIFT {
            return Err(LiteParseError::Ocr(format!(
                "indra lm: order {order} x {bits} bits do not fit a key"
            )));
        }
        let (mut k, mut v, mut hk, mut hb) = (vec![], vec![], vec![], vec![]);
        for n in 0..order {
            let (a, b) = sorted(
                get(&format!("k{n}"))?.u64s()?,
                get(&format!("v{n}"))?.f32s()?,
            );
            k.push(a);
            v.push(b);
            let (a, b) = sorted(
                get(&format!("hk{n}"))?.u64s()?,
                get(&format!("hb{n}"))?.f32s()?,
            );
            hk.push(a);
            hb.push(b);
        }
        let top = v
            .iter()
            .flatten()
            .map(|&x| x as f64)
            .fold(-(vocab_size as f64).ln(), f64::max);
        let bo: f64 = hb
            .iter()
            .filter(|h| !h.is_empty())
            .map(|h| {
                h.iter()
                    .map(|&x| x as f64)
                    .fold(f64::NEG_INFINITY, f64::max)
                    .max(0.0)
            })
            .sum();
        Ok(Self {
            order,
            vocab_size,
            bits,
            k,
            v,
            hk,
            hb,
            max_logp: top + bo,
        })
    }

    /// log P(token | history) for a key: the last `order - 1` history tokens
    /// then the token, packed. Uses the longest history the model has seen,
    /// backing off through the shorter ones.
    pub fn logp_key(&self, full: u64) -> f64 {
        let b = self.bits;
        let ct = full & ((1u64 << b) - 1);
        // Backoff weight per history length (0 when unseen), and m: the
        // longest history length seen with every shorter one seen too.
        let mut hb = vec![0f64; self.order];
        let mut m = 0;
        let mut counting = true;
        for (n, (slot, (keys, vals))) in hb.iter_mut().zip(self.hk.iter().zip(&self.hb)).enumerate()
        {
            let mask = (1u64 << (b as usize * n)) - 1;
            match find(keys, (full >> b) & mask) {
                Some(i) => {
                    *slot = vals[i] as f64;
                    if n > 0 && counting {
                        m = n;
                    }
                }
                None if n > 0 => counting = false,
                None => {}
            }
        }
        let mut chosen = None;
        for n in (0..=m).rev() {
            let mask = (1u64 << (b as usize * n)) - 1;
            let pq = (((full >> b) & mask) << b) | ct;
            if let Some(i) = find(&self.k[n], pq) {
                chosen = Some((n, self.v[n][i] as f64));
                break;
            }
        }
        match chosen {
            Some((c, p)) => hb[c + 1..=m].iter().sum::<f64>() + p,
            None => hb[..=m].iter().sum::<f64>() - (self.vocab_size as f64).ln(),
        }
    }
}

fn lse(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[derive(Clone)]
struct Beam {
    b: f64,
    nb: f64,
    s: f64,
    steps: Vec<usize>,
    hk: u64,
}

/// Insertion-ordered prefix -> beam map (Python dict semantics).
#[derive(Default)]
struct Beams {
    items: Vec<(Vec<i64>, Beam)>,
    index: HashMap<Vec<i64>, usize>,
}

impl Beams {
    fn add(&mut self, prefix: Vec<i64>, b: f64, nb: f64, s: f64, steps: &[usize], hk: u64) {
        if let Some(&i) = self.index.get(&prefix) {
            let e = &mut self.items[i].1;
            e.b = lse(e.b, b);
            e.nb = lse(e.nb, nb);
        } else {
            self.index.insert(prefix.clone(), self.items.len());
            self.items.push((
                prefix,
                Beam {
                    b,
                    nb,
                    s,
                    steps: steps.to_vec(),
                    hk,
                },
            ));
        }
    }
}

/// Settings of one recognizer's beam search.
#[derive(Debug, Clone, Copy)]
pub struct BeamParams {
    pub alpha: f64,
    pub beta: f64,
    pub beam: usize,
    pub cand_floor: f64,
}

/// CTC prefix beam search over a line's steps. `blank`: `[T]` blank
/// log-probabilities; `topv` / `topi`: `[T, k]` the k best classes per step and
/// their log-probabilities. Returns (token ids, the step each token was first
/// emitted at).
pub fn beam_search(
    blank: &[f32],
    topv: &[f32],
    topi: &[i64],
    k: usize,
    lm: &NgramLm,
    p: BeamParams,
) -> (Vec<i64>, Vec<usize>) {
    const BLANK_SKIP: f64 = 0.999;
    let bits = lm.bits;
    let hmask = (1u64 << (bits as usize * (lm.order - 1))) - 1;
    let log_skip = BLANK_SKIP.ln();
    let log_floor = p.cand_floor.ln();
    let reach = p.beta + p.alpha * lm.max_logp;
    let score = |e: &Beam| lse(e.b, e.nb) + e.s;

    let mut beams = Beams::default();
    beams.add(Vec::new(), 0.0, f64::NEG_INFINITY, 0.0, &[], 0);
    for t in 0..blank.len() {
        let b0 = blank[t] as f64;
        if b0 > log_skip {
            for (_, e) in &mut beams.items {
                e.b = lse(e.b, e.nb) + b0;
                e.nb = f64::NEG_INFINITY;
            }
            continue;
        }
        let cands: Vec<(i64, f64)> = (0..k)
            .map(|j| (topi[t * k + j], topv[t * k + j] as f64))
            .filter(|&(c, v)| c != 0 && v >= log_floor)
            .collect();
        let mut nxt = Beams::default();
        // Every beam's own entries first (blank, and a repeat that collapses).
        for (prefix, e) in &beams.items {
            nxt.add(
                prefix.clone(),
                lse(e.b, e.nb) + b0,
                f64::NEG_INFINITY,
                e.s,
                &e.steps,
                e.hk,
            );
            let last = prefix.last().copied();
            for &(c, lp) in &cands {
                if Some(c) == last {
                    nxt.add(
                        prefix.clone(),
                        f64::NEG_INFINITY,
                        e.nb + lp,
                        e.s,
                        &e.steps,
                        e.hk,
                    );
                }
            }
        }
        let floor = if beams.items.len() >= p.beam {
            nxt.items
                .iter()
                .map(|(_, e)| score(e))
                .fold(f64::INFINITY, f64::min)
        } else {
            f64::NEG_INFINITY
        };
        let mut ext: Vec<(usize, i64, f64)> = Vec::new(); // (beam index, class, log p)
        for (bi, (prefix, e)) in beams.items.iter().enumerate() {
            let tot = lse(e.b, e.nb);
            let last = prefix.last().copied();
            for &(c, lp) in &cands {
                // After a blank, the same class is a new token.
                let new_from = if Some(c) == last { e.b + lp } else { tot + lp };
                if new_from == f64::NEG_INFINITY {
                    continue;
                }
                let mut q = prefix.clone();
                q.push(c);
                if nxt.index.contains_key(&q) {
                    nxt.add(q, f64::NEG_INFINITY, new_from, e.s, &e.steps, e.hk);
                } else if new_from + e.s + reach >= floor - 1e-9 {
                    ext.push((bi, c, new_from));
                }
            }
        }
        for (bi, c, new_from) in ext {
            let (prefix, e) = &beams.items[bi];
            let key = (e.hk << bits) | (c as u64 + 1);
            let lmp = lm.logp_key(key);
            let mut q = prefix.clone();
            q.push(c);
            let mut steps = e.steps.clone();
            steps.push(t);
            nxt.add(
                q,
                f64::NEG_INFINITY,
                new_from,
                e.s + p.alpha * lmp + p.beta,
                &steps,
                key & hmask,
            );
        }
        let mut items = nxt.items;
        items.sort_by(|a, b| score(&b.1).total_cmp(&score(&a.1))); // stable
        items.truncate(p.beam);
        beams = Beams {
            index: items
                .iter()
                .enumerate()
                .map(|(i, (q, _))| (q.clone(), i))
                .collect(),
            items,
        };
    }
    let mut best = 0;
    for (i, (_, e)) in beams.items.iter().enumerate() {
        if score(e) > score(&beams.items[best].1) {
            best = i;
        }
    }
    let (prefix, e) = &beams.items[best];
    (prefix.clone(), e.steps.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bigram LM over tokens 1..=2 (bits 11): P(2 | 1) high, P(1 | 1) low.
    fn tiny() -> NgramLm {
        let pack = |ts: &[u64]| ts.iter().fold(0u64, |k, &t| (k << 11) | t);
        NgramLm {
            order: 2,
            vocab_size: 3,
            bits: 11,
            // unigrams: token t is packed as t + 1
            k: vec![vec![2, 3], vec![pack(&[2, 2]), pack(&[2, 3])]],
            v: vec![vec![-1.0, -1.0], vec![-3.0, -0.1]],
            hk: vec![vec![0], vec![2, 3]],
            hb: vec![vec![0.0], vec![-0.5, -0.7]],
            max_logp: -0.1,
        }
    }

    #[test]
    fn backoff_lookup() {
        let lm = tiny();
        let key = |h: u64, t: u64| (h << 11) | t;
        // Seen bigram: its own log-probability.
        assert!((lm.logp_key(key(2, 3)) + 0.1).abs() < 1e-6);
        // History 2 (token 1) seen, (1, 1) seen too.
        assert!((lm.logp_key(key(2, 2)) + 3.0).abs() < 1e-6);
        // History 3 (token 2) seen but (2, 2) not: backoff -0.7 + unigram -1.0.
        assert!((lm.logp_key(key(3, 3)) + 1.7).abs() < 1e-6);
        // Unseen history: unigram only.
        assert!((lm.logp_key(key(9, 2)) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn beam_search_reads_a_clear_line() {
        let lm = tiny();
        // Steps: class 1, blank, class 2. Log-probs: the chosen class near 0.
        let k = 2;
        let blank = [-5.0f32, -0.01, -5.0];
        let topi = [1i64, 2, 0, 1, 2, 1];
        let topv = [-0.01f32, -5.0, -0.01, -5.0, -0.01, -5.0];
        let p = BeamParams {
            alpha: 0.5,
            beta: 1.5,
            beam: 8,
            cand_floor: 3e-3,
        };
        let (ids, steps) = beam_search(&blank, &topv, &topi, k, &lm, p);
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(steps, vec![0, 2]);
    }
}
