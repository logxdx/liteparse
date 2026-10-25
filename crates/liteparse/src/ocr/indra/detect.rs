//! Line detection: `det.onnx` input preparation and output decoding
//! (port of `det_post.page_to_input` / `decode` and `OrtOCR.detect`).

use super::imgproc::{Gray, components4, resize_area, resize_linear};

const CORE_X_SHRINK: f32 = 0.15;
const OUT_STRIDE: usize = 2;
const INPUT_MULTIPLE: usize = 32;

/// Median line height (input px) under which a second, larger pass runs.
const ADAPTIVE_MIN_H: f32 = 9.0;
const ADAPTIVE_TARGET_H: f32 = 12.0;
const ADAPTIVE_MAX_SIDE: usize = 2048;
const ADAPTIVE_MIN_LINES: usize = 8;

/// A detected text line: ink box in page pixels and detector score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
}

/// Detector input: ink map (0 = white) padded to a multiple of 32, with its
/// `[height, width]` and the page-to-input scale.
pub struct DetInput {
    pub data: Vec<f32>,
    pub height: usize,
    pub width: usize,
    pub scale: f32,
}

/// Resize so the long side is `long_side`, pad with white, return the ink map.
pub fn page_to_input(gray: &Gray, long_side: usize) -> DetInput {
    let (h, w) = (gray.height, gray.width);
    let scale = long_side as f32 / h.max(w) as f32;
    let nh = ((h as f32 * scale).round() as usize).max(1);
    let nw = ((w as f32 * scale).round() as usize).max(1);
    let resized;
    let page: &[u8] = if (nh, nw) != (h, w) {
        resized = if scale < 1.0 {
            resize_area(gray, nw, nh)
        } else {
            resize_linear(gray, nw, nh)
        };
        &resized
    } else {
        gray.data
    };
    let ph = nh.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let pw = nw.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let mut data = vec![0f32; ph * pw];
    for y in 0..nh {
        let src = &page[y * nw..(y + 1) * nw];
        let dst = &mut data[y * pw..y * pw + nw];
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = 1.0 - s as f32 / 255.0;
        }
    }
    DetInput {
        data,
        height: ph,
        width: pw,
        scale,
    }
}

/// Decode a `[3, h, w]` detector output (core logit, top distance, bottom
/// distance) into line boxes in input pixels.
pub fn decode(out: &[f32], h: usize, w: usize) -> Vec<LineBox> {
    const THRESH: f32 = 0.5;
    const MIN_AREA: usize = 2;
    let plane = h * w;
    let prob: Vec<f32> = out[..plane]
        .iter()
        .map(|&l| 1.0 / (1.0 + (-l).exp()))
        .collect();
    let mask: Vec<bool> = prob.iter().map(|&p| p > THRESH).collect();
    let (labels, stats) = components4(&mask, w, h);
    if stats.is_empty() {
        return Vec::new();
    }
    let n = stats.len();
    let (mut top, mut bot, mut score) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    let mut cnt = vec![0f64; n];
    for (i, &lab) in labels.iter().enumerate() {
        if lab == 0 {
            continue;
        }
        let k = lab as usize - 1;
        let ys = ((i / w) as f32 + 0.5) * OUT_STRIDE as f32;
        top[k] += (ys - out[plane + i]) as f64;
        bot[k] += (ys + out[2 * plane + i]) as f64;
        score[k] += prob[i] as f64;
        cnt[k] += 1.0;
    }
    let s = OUT_STRIDE as f32;
    stats
        .iter()
        .enumerate()
        .filter(|(_, c)| c.area >= MIN_AREA)
        .map(|(k, c)| {
            let (y0, y1) = ((top[k] / cnt[k]) as f32, (bot[k] / cnt[k]) as f32);
            let lh = (y1 - y0).max(1.0);
            let ex = (CORE_X_SHRINK * lh).min(0.5 * lh);
            LineBox {
                x0: c.left as f32 * s - ex,
                y0,
                x1: (c.left + c.width) as f32 * s + ex,
                y1,
                score: (score[k] / cnt[k]) as f32,
            }
        })
        .collect()
}

/// Long side for the adaptive second pass, or `None` when the first pass is
/// fine. `boxes` are in input pixels.
pub fn adaptive_long_side(boxes: &[LineBox], long_side: usize) -> Option<usize> {
    if boxes.len() < ADAPTIVE_MIN_LINES {
        return None;
    }
    let mut hs: Vec<f32> = boxes.iter().map(|b| b.y1 - b.y0).collect();
    hs.sort_by(|a, b| a.total_cmp(b));
    let m = hs.len();
    let med = if m % 2 == 1 {
        hs[m / 2]
    } else {
        0.5 * (hs[m / 2 - 1] + hs[m / 2])
    };
    if med >= ADAPTIVE_MIN_H {
        return None;
    }
    let want = (long_side as f32 * ADAPTIVE_TARGET_H / med.max(1.0)).min(ADAPTIVE_MAX_SIDE as f32);
    let ls = want as usize / INPUT_MULTIPLE * INPUT_MULTIPLE;
    (ls > long_side).then_some(ls)
}

/// Map input-pixel boxes back to page pixels and clip to the page.
pub fn to_page(boxes: &mut [LineBox], scale: f32, width: usize, height: usize) {
    let (w, h) = (width as f32, height as f32);
    for b in boxes {
        b.x0 = (b.x0 / scale).clamp(0.0, w);
        b.x1 = (b.x1 / scale).clamp(0.0, w);
        b.y0 = (b.y0 / scale).clamp(0.0, h);
        b.y1 = (b.y1 / scale).clamp(0.0, h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_is_padded_ink_map() {
        // 40x20 page at long_side 40: no resize, padded to 64x32.
        let data = vec![255u8; 40 * 20];
        let inp = page_to_input(
            &Gray {
                data: &data,
                width: 40,
                height: 20,
            },
            40,
        );
        assert_eq!((inp.height, inp.width), (32, 64));
        assert_eq!(inp.data[0], 0.0); // white page is zero ink
        assert_eq!(inp.data[63], 0.0); // padding is zero too
    }

    #[test]
    fn decode_finds_one_line() {
        // 8x16 output: a core band at rows 3..5, cols 4..10 of a 16-wide map.
        let (h, w) = (8usize, 16usize);
        let mut out = vec![-10f32; 3 * h * w];
        for y in 3..5 {
            for x in 4..10 {
                out[y * w + x] = 10.0;
                out[h * w + y * w + x] = 4.0; // top distance, input px
                out[2 * h * w + y * w + x] = 4.0; // bottom distance
            }
        }
        let boxes = decode(&out, h, w);
        assert_eq!(boxes.len(), 1);
        let b = boxes[0];
        // mean centre y = ((3.5 + 4.5) / 2) * 2 = 8; box 8 +- 4.
        assert!(
            (b.y0 - 4.0).abs() < 1e-4 && (b.y1 - 12.0).abs() < 1e-4,
            "{b:?}"
        );
        // x spans cols 4..10 (px 8..20) grown by 0.15 * 8 = 1.2.
        assert!(
            (b.x0 - 6.8).abs() < 1e-3 && (b.x1 - 21.2).abs() < 1e-3,
            "{b:?}"
        );
    }

    #[test]
    fn adaptive_only_for_small_text() {
        let line = |h: f32| LineBox {
            x0: 0.0,
            y0: 0.0,
            x1: 50.0,
            y1: h,
            score: 1.0,
        };
        let small: Vec<_> = (0..8).map(|_| line(6.0)).collect();
        assert_eq!(adaptive_long_side(&small, 1024), Some(2048));
        let big: Vec<_> = (0..8).map(|_| line(14.0)).collect();
        assert_eq!(adaptive_long_side(&big, 1024), None);
        assert_eq!(adaptive_long_side(&small[..7], 1024), None);
    }
}
