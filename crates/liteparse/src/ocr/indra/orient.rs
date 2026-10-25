//! Page orientation: skew search, `orient.onnx` input, quarter turns and the
//! deskew rotation, and the mapping of points back to the page as given
//! (port of `pipeline.estimate_skew`, `preview`, `orient_input`, `_turn`).

use super::imgproc::{
    Border, Gray, GrayBuf, apply, fit_long_side, resize, rot90, rotation_matrix, warp_affine,
};

/// Degrees of tilt searched (quarter turns cover the rest).
const DESKEW_MAX: f64 = 45.0;
/// Smaller tilts are left alone (the search step is 0.1).
pub const DESKEW_MIN: f64 = 0.3;
/// The skew search and the orientation model read a preview of at most this.
const PREVIEW_SIDE: usize = 1600;
const SKEW_SIDE: usize = 800;
/// `orient.onnx` input side (long side scaled, padded square).
pub const ORIENT_SIDE: usize = 768;
/// A turn is applied only above this probability: real pages are almost all
/// upright, and a wrong turn loses the page.
pub const ORIENT_MIN: f32 = 0.95;
/// `orient.onnx` class: the page is set in vertical columns.
pub const VERTICAL: usize = 4;

/// The page shrunk once to at most `PREVIEW_SIDE` a side (area averaging).
pub fn preview(gray: &Gray) -> GrayBuf {
    let (nw, nh, s) = fit_long_side(gray.width, gray.height, PREVIEW_SIDE);
    if s >= 1.0 {
        return gray.to_buf();
    }
    resize(gray, nw, nh)
}

/// Ink map (255 = ink) of the page shrunk to `side`.
fn ink_thumb(gray: &Gray, side: usize) -> GrayBuf {
    let (nw, nh, _) = fit_long_side(gray.width, gray.height, side);
    let mut t = resize(gray, nw, nh);
    for v in &mut t.data {
        *v = 255 - *v;
    }
    t
}

/// How sharply the row profile of the ink changes, the page turned by `a`
/// degrees: highest when text rows are level. Equal to warping with
/// [`warp_affine`] (bilinear, zero border) and summing rows, without the
/// intermediate image.
fn row_sharpness(ink: &Gray, a: f64) -> f64 {
    let (w, h) = (ink.width, ink.height);
    let m = rotation_matrix(w as f64 / 2.0, h as f64 / 2.0, a);
    // Inverse of the rotation: output pixel -> source position.
    let inv = super::imgproc::invert(&m);
    let (wi, hi) = (w as isize, h as isize);
    let px = |x: isize, y: isize| -> f32 {
        if x >= 0 && y >= 0 && x < wi && y < hi {
            ink.data[(y * wi + x) as usize] as f32
        } else {
            0.0
        }
    };
    let mut prev: Option<f64> = None;
    let mut acc = 0f64;
    for y in 0..h {
        let (mut sx, mut sy) = (
            inv[0][1] * y as f64 + inv[0][2],
            inv[1][1] * y as f64 + inv[1][2],
        );
        let mut sum = 0u64;
        for _ in 0..w {
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = ((sx - x0) as f32, (sy - y0) as f32);
            let (x0, y0) = (x0 as isize, y0 as isize);
            // Fully outside: all four taps are border zeros.
            if x0 >= -1 && y0 >= -1 && x0 < wi && y0 < hi {
                let top = px(x0, y0) * (1.0 - fx) + px(x0 + 1, y0) * fx;
                let bot = px(x0, y0 + 1) * (1.0 - fx) + px(x0 + 1, y0 + 1) * fx;
                sum += (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u64;
            }
            sx += inv[0][0];
            sy += inv[1][0];
        }
        let s = sum as f64;
        if let Some(p) = prev {
            acc += (s - p) * (s - p);
        }
        prev = Some(s);
    }
    acc
}

/// [`row_sharpness`] at each angle, spread over the available cores.
fn sharpness_at(ink: &Gray, angles: &[f64]) -> Vec<f64> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(angles.len().max(1));
    if threads <= 1 {
        return angles.iter().map(|&a| row_sharpness(ink, a)).collect();
    }
    let chunk = angles.len().div_ceil(threads);
    std::thread::scope(|s| {
        let parts: Vec<_> = angles
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .map(|&a| row_sharpness(ink, a))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        parts
            .into_iter()
            .flat_map(|p| p.join().expect("skew worker panicked"))
            .collect()
    })
}

/// `np.arange(start, stop + 1e-6, step)`.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop + 1e-6 - start) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// First maximum, like Python's `max(..., key=...)`.
fn argmax_first(vals: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in vals.iter().enumerate() {
        if v > vals[best] {
            best = i;
        }
    }
    best
}

/// Degrees (counter-clockwise) that straighten the text rows of a page:
/// a 2 degree grid over +-45 on a half-size thumbnail, then 0.5 and 0.1 degree
/// refinement around the two best angles and 0 on the full thumbnail.
pub fn estimate_skew(gray: &Gray) -> f64 {
    let fine = ink_thumb(gray, SKEW_SIDE);
    let mean = fine.data.iter().map(|&v| v as f64).sum::<f64>() / fine.data.len().max(1) as f64;
    if mean < 0.003 * 255.0 {
        return 0.0; // a blank page has no rows to straighten
    }
    let coarse = ink_thumb(gray, SKEW_SIDE / 2);
    let grid = arange(-DESKEW_MAX, DESKEW_MAX, 2.0);
    let scores = sharpness_at(&coarse.view(), &grid);
    let mut idx: Vec<usize> = (0..grid.len()).collect();
    idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b])); // np.argsort, stable
    let mut starts = vec![0.0];
    for &i in &idx[idx.len().saturating_sub(2)..] {
        if !starts.contains(&grid[i]) {
            starts.push(grid[i]);
        }
    }
    let near: Vec<f64> = starts
        .iter()
        .flat_map(|&a0| arange(a0 - 2.0, a0 + 2.0, 0.5))
        .collect();
    // Overlapping ranges: each angle is scored once.
    let mut unique: Vec<f64> = Vec::new();
    for &a in &near {
        if !unique.contains(&a) {
            unique.push(a);
        }
    }
    let unique_scores = sharpness_at(&fine.view(), &unique);
    let near_scores: Vec<f64> = near
        .iter()
        .map(|a| unique_scores[unique.iter().position(|u| u == a).expect("scored above")])
        .collect();
    let a1 = near[argmax_first(&near_scores)];
    let last = arange(a1 - 0.5, a1 + 0.5, 0.1);
    let vals = sharpness_at(&fine.view(), &last);
    last[argmax_first(&vals)]
}

/// The page rotated by `degrees` counter-clockwise about its centre, same size,
/// edges replicated.
pub fn rotate_page(gray: &Gray, degrees: f64) -> GrayBuf {
    let (w, h) = (gray.width, gray.height);
    warp_affine(
        gray,
        &rotation_matrix(w as f64 / 2.0, h as f64 / 2.0, degrees),
        w,
        h,
        Border::Replicate,
    )
}

/// `orient.onnx` input: `[side, side]` ink map, long side scaled, padded.
pub fn orient_input(gray: &Gray) -> Vec<f32> {
    let (nw, nh, _) = fit_long_side(gray.width, gray.height, ORIENT_SIDE);
    let small = resize(gray, nw, nh);
    let mut out = vec![0f32; ORIENT_SIDE * ORIENT_SIDE];
    for y in 0..nh {
        for x in 0..nw {
            out[y * ORIENT_SIDE + x] = 1.0 - small.data[y * nw + x] as f32 / 255.0;
        }
    }
    out
}

/// How the page that is read was made from the page as given: a quarter turn
/// (or the vertical-page turn), then a deskew rotation. Maps points back.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    /// `orient.onnx` class: 0-3 quarter turns (counter-clockwise) the content
    /// is turned by, or [`VERTICAL`].
    pub class: usize,
    /// Degrees of deskew applied after the turn (0 = none).
    pub skew: f64,
    /// Size of the page as given.
    pub orig_w: usize,
    pub orig_h: usize,
    /// Size of the page that is read (after the turn; the deskew keeps it).
    pub w: usize,
    pub h: usize,
}

impl Frame {
    pub fn is_identity(&self) -> bool {
        self.class == 0 && self.skew == 0.0
    }

    pub fn vertical(&self) -> bool {
        self.class == VERTICAL
    }

    /// A point of the page that is read, on the page as given.
    pub fn to_page(&self, x: f64, y: f64) -> (f64, f64) {
        let (x, y) = if self.skew != 0.0 {
            apply(
                &rotation_matrix(self.w as f64 / 2.0, self.h as f64 / 2.0, -self.skew),
                x,
                y,
            )
        } else {
            (x, y)
        };
        let (w, h) = (self.orig_w as f64, self.orig_h as f64);
        match self.class {
            0 => (x, y),
            1 => (y, h - x),     // read page = rot90(page, -1)
            2 => (w - x, h - y), // rot90(page, 2)
            _ => (w - y, x),     // rot90(page, 1): class 3 and VERTICAL
        }
    }
}

/// The page turned upright (orientation `class`) and straightened: the page to
/// read and its [`Frame`]. `skew` is the angle measured on the preview before
/// any turn; `small` is that preview, measured again after a quarter turn
/// (the rows were columns).
pub fn turn(gray: &Gray, class: usize, skew: f64, small: &Gray) -> (GrayBuf, Frame) {
    let quarter = match class {
        VERTICAL => 1,
        k => -(k as i32),
    };
    let mut page = if quarter == 0 {
        gray.to_buf()
    } else {
        rot90(gray, quarter)
    };
    let mut a = skew;
    if matches!(class, 1 | 3 | VERTICAL) {
        a = estimate_skew(&rot90(small, quarter).view());
    }
    if a.abs() >= DESKEW_MIN {
        page = rotate_page(&page.view(), a);
    } else {
        a = 0.0;
    }
    let frame = Frame {
        class,
        skew: a,
        orig_w: gray.width,
        orig_h: gray.height,
        w: page.width,
        h: page.height,
    };
    (page, frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic page of dark horizontal bars ("text rows") on white.
    fn rows_page(w: usize, h: usize) -> GrayBuf {
        let mut data = vec![255u8; w * h];
        for y in 0..h {
            if (y / 6) % 3 == 1 && y > h / 8 && y < h - h / 8 {
                for x in w / 8..w - w / 8 {
                    data[y * w + x] = 0;
                }
            }
        }
        GrayBuf {
            data,
            width: w,
            height: h,
        }
    }

    #[test]
    fn row_sharpness_equals_warp_and_row_sums() {
        let page = rows_page(120, 90);
        let ink: Vec<u8> = page.data.iter().map(|v| 255 - v).collect();
        let g = Gray {
            data: &ink,
            width: 120,
            height: 90,
        };
        for a in [-7.3, 0.0, 2.0, 31.5] {
            let r = warp_affine(
                &g,
                &rotation_matrix(60.0, 45.0, a),
                120,
                90,
                Border::Constant(0),
            );
            let sums: Vec<f64> = r
                .data
                .chunks(120)
                .map(|row| row.iter().map(|&v| v as f64).sum())
                .collect();
            let want: f64 = sums.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
            assert_eq!(row_sharpness(&g, a), want, "angle {a}");
        }
        assert_eq!(sharpness_at(&g, &[1.0, 2.0, 3.0]).len(), 3);
    }

    #[test]
    fn arange_matches_numpy() {
        assert_eq!(arange(-45.0, 45.0, 2.0).len(), 46);
        assert_eq!(arange(-2.0, 2.0, 0.5).len(), 9);
        let l = arange(2.5, 3.5, 0.1);
        assert_eq!(l.len(), 11);
        assert!((l[10] - 3.5).abs() < 1e-9);
    }

    #[test]
    fn skew_of_level_and_tilted_rows() {
        let page = rows_page(600, 400);
        assert!(estimate_skew(&page.view()).abs() < 0.15);
        let tilted = rotate_page(&page.view(), 4.0); // rows turned 4 degrees CCW
        let a = estimate_skew(&tilted.view());
        assert!((a + 4.0).abs() < 0.25, "{a}");
        let blank = GrayBuf {
            data: vec![255; 300 * 200],
            width: 300,
            height: 200,
        };
        assert_eq!(estimate_skew(&blank.view()), 0.0);
    }

    #[test]
    fn frame_maps_turned_points_back() {
        // A 4x2 page; mark (x=3, y=0) and follow it through each turn.
        let mut data = vec![0u8; 8];
        data[3] = 255;
        let g = Gray {
            data: &data,
            width: 4,
            height: 2,
        };
        // A blank preview: no skew is measured after the quarter turns.
        let white = [255u8; 8];
        let small = Gray {
            data: &white,
            width: 4,
            height: 2,
        };
        for class in 0..=3usize {
            let (page, frame) = turn(&g, class, 0.0, &small);
            assert_eq!(frame.skew, 0.0);
            let i = page.data.iter().position(|&v| v == 255).unwrap();
            let (x, y) = ((i % page.width) as f64 + 0.5, (i / page.width) as f64 + 0.5);
            let (px, py) = frame.to_page(x, y);
            assert!(
                (px - 3.5).abs() < 1e-9 && (py - 0.5).abs() < 1e-9,
                "class {class}: {px},{py}"
            );
        }
    }

    #[test]
    fn frame_undoes_deskew() {
        let f = Frame {
            class: 0,
            skew: 10.0,
            orig_w: 100,
            orig_h: 50,
            w: 100,
            h: 50,
        };
        // The centre is fixed; a point right of it moves back along the arc.
        let (x, y) = f.to_page(50.0, 25.0);
        assert!((x - 50.0).abs() < 1e-9 && (y - 25.0).abs() < 1e-9);
        let m = rotation_matrix(50.0, 25.0, 10.0);
        let (fx, fy) = apply(&m, 80.0, 25.0); // where the deskew put (80, 25)
        let (x, y) = f.to_page(fx, fy);
        assert!((x - 80.0).abs() < 1e-6 && (y - 25.0).abs() < 1e-6);
    }
}
