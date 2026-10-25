//! From recognized lines to output boxes (port of `OrtOCR.assemble` without
//! reading order, which is LiteParse's job): split lines at wide gaps and
//! column gutters, drop leader rules, join lone list markers to their items,
//! and optionally cut lines into word boxes.

use unicode_normalization::UnicodeNormalization;

use super::models::Vocab;
use super::rtl::{flip, has_rtl, is_rtl};

/// Cut a detected line at a gap wider than this many line heights.
pub const SPLIT_GAP: f32 = 3.0;
/// Cut at a smaller gap when other lines' edges line up with it (a gutter).
const GUTTER_GAP: f32 = 1.6;
/// Other line edges that must line up (within half a line height).
const GUTTER_EDGES: usize = 3;
/// A list marker and its item are at most this many line heights apart.
const MARKER_GAP: f32 = 2.5;
/// Token extent around a character centre, in line heights.
const TOKEN_PAD: f32 = 0.35;

const LEADER_CHARS: &str = ".·…‥_-–—=";
const DASHES: &str = "-–—";
const BULLETS: &str = "•◦▪▫●○■□‣⁃∙*";

/// A recognized token: a class id of the base recognizer, or a character of a
/// group recognizer.
#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Id(i64),
    Char(String),
}

/// A token and the page x of its CTC peak (in the frame that was read).
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub x: f32,
}

/// One recognized line: its text, confidence and tokens. A line under the
/// confidence floor has empty text and no tokens.
#[derive(Debug, Clone, Default)]
pub struct LineRead {
    pub text: String,
    pub conf: f32,
    pub toks: Vec<Token>,
}

/// A box `[x0, y0, x1, y1]` with its text and confidence.
#[derive(Debug, Clone, PartialEq)]
pub struct TextBox {
    pub bbox: [f32; 4],
    pub text: String,
    pub conf: f32,
}

pub struct Assembler<'a> {
    vocab: &'a Vocab,
    /// The page reads right to left (more RTL letters than others).
    page_rtl: bool,
}

impl<'a> Assembler<'a> {
    pub fn new(vocab: &'a Vocab, lines: &[LineRead]) -> Self {
        let letters: Vec<char> = lines
            .iter()
            .flat_map(|l| l.text.chars())
            .filter(|c| c.is_alphabetic())
            .collect();
        let n_rtl = letters.iter().filter(|&&c| is_rtl(c)).count();
        Self {
            vocab,
            page_rtl: n_rtl > letters.len() - n_rtl,
        }
    }

    fn is_space(&self, t: &Tok) -> bool {
        match t {
            Tok::Char(c) => c == " ",
            Tok::Id(i) => Some(*i) == self.vocab.space(),
        }
    }

    /// Text of tokens: class ids through the vocabulary; characters of a group
    /// recognizer joined, NFC-composed, and right-to-left runs put in logical
    /// order.
    pub fn decode(&self, toks: &[Token]) -> String {
        if matches!(
            toks.first(),
            Some(Token {
                tok: Tok::Char(_),
                ..
            })
        ) {
            let joined: String = toks
                .iter()
                .filter_map(|t| match &t.tok {
                    Tok::Char(c) => Some(c.as_str()),
                    Tok::Id(_) => None,
                })
                .collect();
            let text: String = joined
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .nfc()
                .collect();
            if has_rtl(&text) {
                return flip(&text, self.page_rtl.then_some(true));
            }
            return text;
        }
        let ids: Vec<i64> = toks
            .iter()
            .filter_map(|t| match t.tok {
                Tok::Id(i) => Some(i),
                Tok::Char(_) => None,
            })
            .collect();
        self.vocab.decode(&ids)
    }

    /// Cut a line where two neighbouring characters are more than `gap` line
    /// heights apart, or across one of the `at` x positions.
    pub fn split_line(
        &self,
        b: [f32; 4],
        toks: &[Token],
        gap: f32,
        at: &[f32],
    ) -> Vec<([f32; 4], String, Vec<Token>)> {
        let h = (b[3] - b[1]).max(1.0);
        let mut segs: Vec<Vec<Token>> = vec![Vec::new()];
        let mut prev: Option<f32> = None;
        for tk in toks {
            if !self.is_space(&tk.tok) {
                let x = tk.x;
                let cut =
                    prev.is_some_and(|p| x - p > gap * h || at.iter().any(|&c| p < c && c < x));
                if cut {
                    segs.push(Vec::new());
                }
                prev = Some(x);
            }
            segs.last_mut().unwrap().push(tk.clone());
        }
        if segs.len() == 1 {
            return vec![(b, self.decode(toks), toks.to_vec())];
        }
        segs.into_iter()
            .filter_map(|sg| {
                let text = self.decode(&sg).trim().to_owned();
                let xs: Vec<f32> = sg
                    .iter()
                    .filter(|t| !self.is_space(&t.tok))
                    .map(|t| t.x)
                    .collect();
                if text.is_empty() || xs.is_empty() {
                    return None;
                }
                Some((self.token_box(b, &xs, h), text, sg))
            })
            .collect()
    }

    /// The part of line box `b` spanned by characters at `xs`.
    fn token_box(&self, b: [f32; 4], xs: &[f32], h: f32) -> [f32; 4] {
        let lo = xs.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        [
            b[0].max(lo - TOKEN_PAD * h),
            b[1],
            b[2].min(hi + TOKEN_PAD * h),
            b[3],
        ]
    }

    /// Per line, the x positions of gaps on a column gutter: wider than
    /// `GUTTER_GAP` line heights, with at least `GUTTER_EDGES` other lines
    /// ending at its left side or starting at its right side.
    pub fn gutter_cuts(&self, boxes: &[[f32; 4]], lines: &[LineRead]) -> Vec<Vec<f32>> {
        boxes
            .iter()
            .zip(lines)
            .enumerate()
            .map(|(i, (b, line))| {
                let h = (b[3] - b[1]).max(1.0);
                let xs: Vec<f32> = line
                    .toks
                    .iter()
                    .filter(|t| !self.is_space(&t.tok))
                    .map(|t| t.x)
                    .collect();
                xs.windows(2)
                    .filter(|w| w[1] - w[0] > GUTTER_GAP * h)
                    .filter(|w| {
                        let (gl, gr) = (w[0] + TOKEN_PAD * h, w[1] - TOKEN_PAD * h);
                        let near = boxes
                            .iter()
                            .enumerate()
                            .filter(|&(j, o)| {
                                j != i
                                    && ((o[2] - gl).abs() < 0.5 * h || (o[0] - gr).abs() < 0.5 * h)
                            })
                            .count();
                        near >= GUTTER_EDGES
                    })
                    .map(|w| (w[0] + w[1]) / 2.0)
                    .collect()
            })
            .collect()
    }

    /// Lines without text of their own: leader rules are dropped, and a lone
    /// list marker joins the item that starts just after it on the same row.
    pub fn tidy_lines(&self, boxes: &mut Vec<[f32; 4]>, lines: &mut Vec<LineRead>) {
        let n = lines.len();
        let mut keep = vec![true; n];
        let h: Vec<f32> = boxes.iter().map(|b| b[3] - b[1]).collect();
        let w: Vec<f32> = boxes.iter().map(|b| b[2] - b[0]).collect();
        for (i, line) in lines.iter().enumerate() {
            let core: String = line.text.chars().filter(|&c| c != ' ').collect();
            if !core.is_empty() && core.chars().all(|c| LEADER_CHARS.contains(c)) {
                let wide = w[i] >= 3.0 * h[i];
                if core.chars().count() >= 3 || wide || !core.chars().any(|c| DASHES.contains(c)) {
                    keep[i] = false;
                }
            }
        }
        for i in 0..n {
            let t = lines[i].text.trim().to_owned();
            let is_bullet = t.chars().count() == 1 && BULLETS.contains(t.as_str());
            if !keep[i] || !(is_bullet || is_enumerator(&t)) {
                continue;
            }
            let mut best: Option<(usize, f32)> = None;
            for j in 0..n {
                if j == i || !keep[j] || lines[j].text.trim().is_empty() {
                    continue;
                }
                let (bi, bj) = (boxes[i], boxes[j]);
                let ov = bi[3].min(bj[3]) - bi[1].max(bj[1]);
                if ov < 0.5 * h[i].min(h[j]) {
                    continue;
                }
                let gap = if self.page_rtl {
                    bi[0] - bj[2]
                } else {
                    bj[0] - bi[2]
                };
                if -0.3 * h[i] <= gap
                    && gap <= MARKER_GAP * h[i].max(h[j])
                    && best.is_none_or(|(_, g)| gap < g)
                {
                    best = Some((j, gap));
                }
            }
            let Some((j, _)) = best else { continue };
            let marker_toks = lines[i].toks.clone();
            let item = &mut lines[j];
            let chars = matches!(
                item.toks.first(),
                Some(Token {
                    tok: Tok::Char(_),
                    ..
                })
            );
            let space = if chars {
                Some(Tok::Char(" ".into()))
            } else {
                self.vocab.space().map(Tok::Id)
            };
            // Left to right only, and only tokens of the same kind join.
            if let (false, Some(first), true, Some(sp)) = (
                self.page_rtl,
                marker_toks.first(),
                !item.toks.is_empty(),
                space,
            ) && matches!(first.tok, Tok::Char(_)) == chars
            {
                let x = (boxes[i][2] + boxes[j][0]) / 2.0;
                let mut toks = marker_toks.clone();
                toks.push(Token { tok: sp, x });
                toks.append(&mut item.toks);
                item.toks = toks;
            }
            item.text = format!("{t} {}", item.text.trim());
            boxes[j] = [
                boxes[i][0].min(boxes[j][0]),
                boxes[i][1].min(boxes[j][1]),
                boxes[i][2].max(boxes[j][2]),
                boxes[i][3].max(boxes[j][3]),
            ];
            keep[i] = false;
        }
        let mut k = keep.iter();
        boxes.retain(|_| *k.next().unwrap());
        let mut k = keep.iter();
        lines.retain(|_| *k.next().unwrap());
    }

    /// Lines -> split at wide gaps and gutters -> tidy. Returns the output
    /// boxes (lines with text only) and, per box, its tokens.
    pub fn assemble(&self, boxes: &[[f32; 4]], lines: &[LineRead]) -> Vec<(TextBox, Vec<Token>)> {
        let cuts = self.gutter_cuts(boxes, lines);
        let (mut nb, mut nl) = (Vec::new(), Vec::new());
        for ((b, line), at) in boxes.iter().zip(lines).zip(&cuts) {
            for (bx, text, toks) in self.split_line(*b, &line.toks, SPLIT_GAP, at) {
                nb.push(bx);
                nl.push(LineRead {
                    text,
                    conf: line.conf,
                    toks,
                });
            }
        }
        self.tidy_lines(&mut nb, &mut nl);
        nb.into_iter()
            .zip(nl)
            .filter(|(_, l)| !l.text.trim().is_empty())
            .map(|(bbox, l)| {
                (
                    TextBox {
                        bbox,
                        text: l.text.trim().to_owned(),
                        conf: l.conf,
                    },
                    l.toks,
                )
            })
            .collect()
    }

    /// Word boxes of a line: its tokens cut at space tokens. Each word spans
    /// half a character pitch beyond its first and last character centres (the
    /// pitch is the median distance between neighbouring characters of a word
    /// on the line), so neighbouring word boxes are about a space apart. A line
    /// without tokens (merged text only) stays one box.
    pub fn words(&self, line: &TextBox, toks: &[Token]) -> Vec<TextBox> {
        let h = (line.bbox[3] - line.bbox[1]).max(1.0);
        let mut steps: Vec<f32> = toks
            .windows(2)
            .filter(|w| !self.is_space(&w[0].tok) && !self.is_space(&w[1].tok))
            .map(|w| w[1].x - w[0].x)
            .filter(|d| *d > 0.0)
            .collect();
        steps.sort_by(f32::total_cmp);
        let half = steps
            .get(steps.len() / 2)
            .map_or(TOKEN_PAD * h, |m| 0.5 * m)
            .clamp(0.1 * h, 0.5 * h);
        let mut out = Vec::new();
        let mut seg: Vec<Token> = Vec::new();
        let flush = |seg: &mut Vec<Token>, out: &mut Vec<TextBox>| {
            if seg.is_empty() {
                return;
            }
            let text = self.decode(seg).trim().to_owned();
            if !text.is_empty() {
                let lo = seg.iter().map(|t| t.x).fold(f32::INFINITY, f32::min);
                let hi = seg.iter().map(|t| t.x).fold(f32::NEG_INFINITY, f32::max);
                let b = line.bbox;
                out.push(TextBox {
                    bbox: [b[0].max(lo - half), b[1], b[2].min(hi + half), b[3]],
                    text,
                    conf: line.conf,
                });
            }
            seg.clear();
        };
        for tk in toks {
            if self.is_space(&tk.tok) {
                flush(&mut seg, &mut out);
            } else {
                seg.push(tk.clone());
            }
        }
        flush(&mut seg, &mut out);
        if out.is_empty() {
            out.push(line.clone());
        }
        out
    }
}

/// A numbered list marker: `1.`, `12)`, `a.`, `B)`, `(3)`, `(b)`, `(iv)`.
fn is_enumerator(t: &str) -> bool {
    let c: Vec<char> = t.chars().collect();
    let digits = |s: &[char]| (1..=3).contains(&s.len()) && s.iter().all(|c| c.is_numeric());
    let letter = |s: &[char]| s.len() == 1 && s[0].is_ascii_alphabetic();
    let roman =
        |s: &[char]| (1..=5).contains(&s.len()) && s.iter().all(|c| "ivxlcIVXLC".contains(*c));
    if let [body @ .., last] = c.as_slice()
        && (*last == '.' || *last == ')')
        && (digits(body) || letter(body))
    {
        return true;
    }
    if let ['(', body @ .., ')'] = c.as_slice() {
        return digits(body) || letter(body) || roman(body);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocab {
        Vocab::new(["a", "b", " ", "•", "1", "."].map(String::from).to_vec())
    }

    fn ids(v: &[(i64, f32)]) -> Vec<Token> {
        v.iter()
            .map(|&(i, x)| Token { tok: Tok::Id(i), x })
            .collect()
    }

    #[test]
    fn enumerators() {
        for t in ["1.", "12)", "a.", "B)", "(3)", "(b)", "(iv)"] {
            assert!(is_enumerator(t), "{t}");
        }
        for t in ["1234.", "ab.", "(abc)", "1", "-"] {
            assert!(!is_enumerator(t), "{t}");
        }
    }

    #[test]
    fn splits_at_wide_gap() {
        let v = vocab();
        let a = Assembler::new(&v, &[]);
        let b = [0.0, 0.0, 300.0, 10.0]; // h = 10
        // "ab" ... 50 px gap (> 3 h) ... "ba"
        let toks = ids(&[(1, 5.0), (2, 12.0), (2, 62.0), (1, 70.0)]);
        let parts = a.split_line(b, &toks, SPLIT_GAP, &[]);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].1, "ab");
        assert_eq!(parts[1].1, "ba");
        assert!((parts[0].0[0] - 1.5).abs() < 1e-4 && (parts[0].0[2] - 15.5).abs() < 1e-4);
        // A narrow gap stays one line.
        let toks = ids(&[(1, 5.0), (3, 10.0), (2, 15.0)]);
        assert_eq!(a.split_line(b, &toks, SPLIT_GAP, &[]).len(), 1);
    }

    #[test]
    fn marker_joins_item_and_leaders_drop() {
        let v = vocab();
        let a = Assembler::new(&v, &[]);
        let mut boxes = vec![
            [0.0, 0.0, 8.0, 10.0],    // "•"
            [12.0, 0.0, 80.0, 10.0],  // "ab"
            [0.0, 20.0, 200.0, 30.0], // "..." leader rule
        ];
        let line = |t: &str, toks: Vec<Token>| LineRead {
            text: t.into(),
            conf: 0.9,
            toks,
        };
        let mut lines = vec![
            line("•", ids(&[(4, 4.0)])),
            line("ab", ids(&[(1, 15.0), (2, 22.0)])),
            line("...", ids(&[(6, 10.0)])),
        ];
        a.tidy_lines(&mut boxes, &mut lines);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "• ab");
        assert_eq!(boxes[0], [0.0, 0.0, 80.0, 10.0]);
        assert_eq!(a.decode(&lines[0].toks), "• ab");
    }

    #[test]
    fn words_cut_at_spaces() {
        let v = vocab();
        let a = Assembler::new(&v, &[]);
        let line = TextBox {
            bbox: [0.0, 0.0, 100.0, 10.0],
            text: "ab ba".into(),
            conf: 0.9,
        };
        let toks = ids(&[(1, 5.0), (2, 12.0), (3, 20.0), (2, 30.0), (1, 37.0)]);
        let words = a.words(&line, &toks);
        assert_eq!(
            words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
            vec!["ab", "ba"]
        );
        assert!(words[0].bbox[2] < words[1].bbox[0]);
    }

    #[test]
    fn char_tokens_decode_rtl_in_logical_order() {
        let v = vocab();
        let lines = [LineRead {
            text: "גבא".into(),
            conf: 0.9,
            toks: vec![],
        }];
        let a = Assembler::new(&v, &lines);
        let toks: Vec<Token> = ["ג", "ב", "א"]
            .iter()
            .enumerate()
            .map(|(i, c)| Token {
                tok: Tok::Char(c.to_string()),
                x: i as f32,
            })
            .collect();
        assert_eq!(a.decode(&toks), "אבג");
    }
}
