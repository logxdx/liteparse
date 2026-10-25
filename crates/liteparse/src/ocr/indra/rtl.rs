//! Right-to-left text for a recognizer that reads every line left to right
//! (port of Indra's `rtl.py`). CTC emits characters in visual order; [`flip`]
//! turns visual order into logical order (and back: it is its own inverse).
//!
//! A small subset of the Unicode bidi algorithm: no embeddings or isolates,
//! neutrals resolved by runs. A cluster is a base character with its combining
//! marks; mirrored characters (brackets) at right-to-left level are mirrored.

use unicode_bidi::{BidiClass, bidi_class};
use unicode_normalization::char::is_combining_mark;

const MIRROR: [(char, char); 12] = [
    ('(', ')'),
    (')', '('),
    ('[', ']'),
    (']', '['),
    ('{', '}'),
    ('}', '{'),
    ('<', '>'),
    ('>', '<'),
    ('«', '»'),
    ('»', '«'),
    ('‹', '›'),
    ('›', '‹'),
];

pub fn is_rtl(c: char) -> bool {
    matches!(bidi_class(c), BidiClass::R | BidiClass::AL)
}

/// Digits (European and Arabic-Indic) read left to right inside RTL text.
fn is_ltr(c: char) -> bool {
    matches!(bidi_class(c), BidiClass::L | BidiClass::EN | BidiClass::AN)
}

pub fn has_rtl(text: &str) -> bool {
    text.chars().any(is_rtl)
}

/// The line reads right to left: more RTL letters than LTR letters.
pub fn rtl_base(text: &str) -> bool {
    let r = text.chars().filter(|&c| is_rtl(c)).count();
    let l = text
        .chars()
        .filter(|&c| bidi_class(c) == BidiClass::L)
        .count();
    r > l
}

fn clusters(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in text.chars() {
        match out.last_mut() {
            Some(last) if is_combining_mark(c) => last.push(c),
            _ => out.push(c.to_string()),
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    R,
    L,
    N,
}

fn kind(cl: &str) -> Kind {
    let c = cl.chars().next().unwrap_or(' ');
    if is_rtl(c) {
        Kind::R
    } else if is_ltr(c) {
        Kind::L
    } else {
        Kind::N
    }
}

/// `[start, end)` spans from a `strong` cluster to the last `strong` cluster
/// before a cluster of kind `stop` (or the end).
fn runs(kinds: &[Kind], strong: Kind, stop: Kind) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let (mut start, mut last) = (None, 0);
    for i in 0..=kinds.len() {
        let k = kinds.get(i).copied();
        if k == Some(strong) {
            start.get_or_insert(i);
            last = i;
        } else if (k.is_none() || k == Some(stop))
            && let Some(s) = start.take()
        {
            spans.push((s, last + 1));
        }
    }
    spans
}

fn mirror(cl: &str) -> String {
    let mut chars = cl.chars();
    let first = chars.next().unwrap_or(' ');
    let m = MIRROR
        .iter()
        .find(|(a, _)| *a == first)
        .map_or(first, |(_, b)| *b);
    std::iter::once(m).chain(chars).collect()
}

/// Logical order to visual order and back. `base_rtl`: the paragraph
/// direction when known; `None` decides by [`rtl_base`].
pub fn flip(text: &str, base_rtl: Option<bool>) -> String {
    if !has_rtl(text) {
        return text.to_owned();
    }
    let cls = clusters(text);
    let kinds: Vec<Kind> = cls.iter().map(|c| kind(c)).collect();
    if base_rtl.unwrap_or_else(|| rtl_base(text)) {
        // Units: LTR runs kept whole, every other cluster alone; reverse the units.
        let mut units: Vec<String> = Vec::new();
        let mut spans = runs(&kinds, Kind::L, Kind::R);
        spans.push((cls.len(), cls.len()));
        let mut i = 0;
        for (a, b) in spans {
            for cl in &cls[i..a] {
                units.push(mirror(cl));
            }
            if a < b {
                units.push(cls[a..b].concat());
            }
            i = b;
        }
        units.reverse();
        return units.concat();
    }
    // LTR base: reverse each RTL run in place.
    let mut out = cls.clone();
    for (a, b) in runs(&kinds, Kind::R, Kind::L) {
        for (k, cl) in cls[a..b].iter().rev().enumerate() {
            out[a + k] = mirror(cl);
        }
    }
    out.concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latin_is_unchanged() {
        assert_eq!(flip("hello (world)", None), "hello (world)");
    }

    #[test]
    fn rtl_line_reverses_units_and_keeps_ltr_runs() {
        // Visual "abc )ב א(" -> logical: Hebrew read right to left, brackets mirrored.
        let visual = "123 גבא";
        assert_eq!(flip(visual, None), "אבג 123");
        // Self-inverse.
        assert_eq!(flip(&flip(visual, None), None), visual);
        assert_eq!(flip("(אב)", None), "(בא)");
    }

    #[test]
    fn ltr_base_reverses_rtl_runs_only() {
        let s = "word בא word word";
        assert_eq!(flip(s, None), "word אב word word");
        // Values from indra_ocr.rtl.flip.
        assert_eq!(flip(s, Some(true)), "word word אב word");
        assert_eq!(flip("الكلب 42 (كبير)", None), "(ريبك) 42 بلكلا");
    }
}
