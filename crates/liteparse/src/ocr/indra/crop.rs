//! Recognizer input: line crops, dark-line detection and width batching
//! (port of `rec_preproc`).

use super::detect::LineBox;
use super::imgproc::{Gray, percentiles, resize_area, resize_linear};

pub const LINE_HEIGHT: usize = 32;
pub const PAD_Y: f32 = 0.2;
pub const PAD_X: f32 = 0.3;
pub const WIDTH_MULTIPLE: usize = 8;
const MAX_CROP_WIDTH: usize = 4096;

/// Light text on a dark ground: the bright tail of the line box is longer than
/// the dark one, so the line is read inverted.
pub fn is_dark_line(gray: &Gray, b: &LineBox) -> bool {
    let (h, w) = (gray.height, gray.width);
    let y0 = (b.y0 as isize).max(0) as usize;
    let y1 = ((b.y1 as isize + 1).max(0) as usize).min(h);
    let x0 = (b.x0 as isize).max(0) as usize;
    let x1 = ((b.x1 as isize + 1).max(0) as usize).min(w);
    if y1 <= y0 || x1 <= x0 {
        return false;
    }
    let mut region = Vec::with_capacity((y1 - y0) * (x1 - x0));
    for y in y0..y1 {
        region.extend_from_slice(&gray.data[y * w + x0..y * w + x1]);
    }
    let [lo, med, hi] = percentiles(&region, [0.02, 0.5, 0.98]);
    hi - med > 2.0 * (med - lo) + 8.0
}

/// Ink-map crop of one line, `[window * LINE_HEIGHT, width]` row-major.
/// `window > 1` keeps the text scale and adds the page above and below.
pub fn crop_line(gray: &Gray, b: &LineBox, window: usize, dark: bool) -> (Vec<f32>, usize) {
    let w = window as f32;
    let pad_y = (w * (1.0 + 2.0 * PAD_Y) - 1.0) / 2.0;
    let height = LINE_HEIGHT * window;

    let h = (b.y1 - b.y0).max(1.0);
    let (x0, x1) = (b.x0 - PAD_X * h, b.x1 + PAD_X * h);
    let (y0, y1) = (b.y0 - pad_y * h, b.y1 + pad_y * h);
    let (ix0, iy0) = (x0.floor() as isize, y0.floor() as isize);
    let (ix1, iy1) = (x1.ceil() as isize, y1.ceil() as isize);
    let cw = (ix1 - ix0).max(1) as usize;
    let ch = (iy1 - iy0).max(1) as usize;
    let (pw, ph) = (gray.width as isize, gray.height as isize);
    let (sx0, sy0) = (ix0.max(0), iy0.max(0));
    let (sx1, sy1) = (ix1.min(pw), iy1.min(ph));

    let mut crop = vec![if dark { 0u8 } else { 255u8 }; cw * ch];
    if sx1 > sx0 && sy1 > sy0 {
        let n = (sx1 - sx0) as usize;
        for sy in sy0..sy1 {
            let src = &gray.data[(sy * pw + sx0) as usize..][..n];
            let dy = (sy - iy0) as usize;
            let dx = (sx0 - ix0) as usize;
            crop[dy * cw + dx..dy * cw + dx + n].copy_from_slice(src);
        }
    }
    let out_w = ((cw as f32 * height as f32 / ch as f32).round() as usize)
        .clamp(WIDTH_MULTIPLE, MAX_CROP_WIDTH);
    let src = Gray {
        data: &crop,
        width: cw,
        height: ch,
    };
    let resized = if ch > height {
        resize_area(&src, out_w, height)
    } else {
        resize_linear(&src, out_w, height)
    };
    let ink = resized
        .iter()
        .map(|&v| {
            let v = v as f32 / 255.0;
            if dark { v } else { 1.0 - v }
        })
        .collect();
    (ink, out_w)
}

/// Width of the 1x line crop (`window = 1`) without building it; batching and
/// the CTC step-to-page mapping use this width, as in the reference.
pub fn line_crop_width(b: &LineBox) -> usize {
    let h = (b.y1 - b.y0).max(1.0);
    let cw = ((b.x1 + PAD_X * h).ceil() as isize - (b.x0 - PAD_X * h).floor() as isize).max(1);
    let ch = (((b.y1 + PAD_Y * h).ceil() as isize) - ((b.y0 - PAD_Y * h).floor() as isize)).max(1);
    ((cw as f32 * LINE_HEIGHT as f32 / ch as f32).round() as usize)
        .clamp(WIDTH_MULTIPLE, MAX_CROP_WIDTH)
}

/// Right-pad ink maps (row-major `[rows, width]`) to one width, a multiple of
/// `multiple` (at least `WIDTH_MULTIPLE`); with `pow2_rows`, add blank lines up
/// to a power of two (fixed shapes help GPU providers). Returns data
/// `[B, 1, rows, W]`, widths `[B]`, `W` and `B`.
pub fn pad_batch(
    lines: &[(&[f32], usize)],
    rows: usize,
    multiple: usize,
    pow2_rows: bool,
) -> (Vec<f32>, Vec<i64>, usize, usize) {
    let multiple = multiple.max(WIDTH_MULTIPLE);
    let wmax = lines.iter().map(|l| l.1).max().unwrap_or(0);
    let wm = wmax.div_ceil(multiple).max(1) * multiple;
    let b = if pow2_rows {
        lines.len().next_power_of_two().max(8)
    } else {
        lines.len()
    };
    let mut out = vec![0f32; b * rows * wm];
    let mut widths = vec![wm as i64; b];
    for (i, (data, w)) in lines.iter().enumerate() {
        for r in 0..rows {
            let dst = (i * rows + r) * wm;
            out[dst..dst + w].copy_from_slice(&data[r * w..(r + 1) * w]);
        }
        widths[i] = *w as i64;
    }
    (out, widths, wm, b)
}

/// Group line indices into batches of similar width so little compute goes to
/// padding; also caps `lines x width` to bound ONNX Runtime's allocations.
pub fn width_batches(widths: &[usize], max_batch: usize) -> Vec<Vec<usize>> {
    const MAX_RATIO: f32 = 1.3;
    const MIN_WIDTH: usize = 64;
    const MAX_AREA: usize = 64 * 640;
    let mut order: Vec<usize> = (0..widths.len()).collect();
    order.sort_by_key(|&i| widths[i]); // stable, like Python's sorted
    let mut batches: Vec<Vec<usize>> = Vec::new();
    for i in order {
        let wi = widths[i].max(MIN_WIDTH);
        let joins = batches.last().is_some_and(|b| {
            b.len() < max_batch
                && wi as f32 <= MAX_RATIO * widths[b[0]].max(MIN_WIDTH) as f32
                && (b.len() + 1) * wi <= MAX_AREA
        });
        if joins {
            batches.last_mut().unwrap().push(i);
        } else {
            batches.push(vec![i]);
        }
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bx(x0: f32, y0: f32, x1: f32, y1: f32) -> LineBox {
        LineBox {
            x0,
            y0,
            x1,
            y1,
            score: 1.0,
        }
    }

    #[test]
    fn crop_shape_follows_aspect() {
        let page = vec![255u8; 200 * 100];
        let g = Gray {
            data: &page,
            width: 200,
            height: 100,
        };
        let b = bx(20.0, 40.0, 120.0, 60.0); // h = 20
        let (ink, w) = crop_line(&g, &b, 1, false);
        assert_eq!(ink.len(), LINE_HEIGHT * w);
        assert!(ink.iter().all(|&v| v == 0.0)); // blank paper = no ink
        assert_eq!(w, line_crop_width(&b));
        let (tall, tw) = crop_line(&g, &b, 3, false);
        assert_eq!(tall.len(), 3 * LINE_HEIGHT * tw);
        assert!(tw.abs_diff(w) <= 1);
    }

    #[test]
    fn dark_line_detection() {
        // Light text (255) on a dark ground (20): dark line.
        let mut page = vec![20u8; 100 * 40];
        for x in 10..20 {
            for y in 10..30 {
                page[y * 100 + x] = 255;
            }
        }
        let g = Gray {
            data: &page,
            width: 100,
            height: 40,
        };
        let b = bx(5.0, 5.0, 90.0, 35.0);
        assert!(is_dark_line(&g, &b));
        let white = vec![255u8; 100 * 40];
        let gw = Gray {
            data: &white,
            width: 100,
            height: 40,
        };
        assert!(!is_dark_line(&gw, &b));
    }

    #[test]
    fn batches_group_similar_widths() {
        let widths = [100, 400, 105, 410, 90];
        let batches = width_batches(&widths, 64);
        assert_eq!(batches, vec![vec![4, 0, 2], vec![1, 3]]);
        let capped = width_batches(&widths, 2);
        assert!(capped.iter().all(|b| b.len() <= 2));
    }

    #[test]
    fn pad_batch_pads_to_multiple() {
        let a = [1f32; 2 * 5];
        let b = [2f32; 2 * 3];
        let (data, widths, wm, rows) = pad_batch(&[(&a, 5), (&b, 3)], 2, 8, false);
        assert_eq!((wm, widths, rows), (8, vec![5, 3], 2));
        let (padded, w3, wm3, rows3) = pad_batch(&[(&a, 5), (&b, 3), (&b, 3)], 2, 512, true);
        assert_eq!((wm3, rows3, w3.len()), (512, 8, 8));
        assert_eq!(padded.len(), 8 * 2 * 512);
        assert_eq!(data.len(), 2 * 2 * 8);
        assert_eq!(&data[8..13], &[1.0; 5]); // second row of line 0
        assert_eq!(&data[16..19], &[2.0; 3]); // first row of line 1
        assert_eq!(data[19], 0.0);
    }
}
