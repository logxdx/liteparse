//! Image helpers with the `cv2` semantics Indra's pre- and post-processing
//! relies on. `image::imageops::resize` filters are not `INTER_AREA`, and a
//! fractional-pixel shift in a crop costs accuracy quietly, so these are ported
//! rather than substituted.

/// A single-channel `u8` image, row-major and tightly packed.
#[derive(Clone, Copy)]
pub struct Gray<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
}

/// An owned [`Gray`] image.
#[derive(Clone)]
pub struct GrayBuf {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl GrayBuf {
    pub fn view(&self) -> Gray<'_> {
        Gray {
            data: &self.data,
            width: self.width,
            height: self.height,
        }
    }
}

impl Gray<'_> {
    pub fn to_buf(&self) -> GrayBuf {
        GrayBuf {
            data: self.data.to_vec(),
            width: self.width,
            height: self.height,
        }
    }
}

/// `cv2.resize` the way Indra calls it: `INTER_AREA` when shrinking, else
/// `INTER_LINEAR`.
pub fn resize(src: &Gray, nw: usize, nh: usize) -> GrayBuf {
    let data = if (nw, nh) == (src.width, src.height) {
        src.data.to_vec()
    } else if nw < src.width || nh < src.height {
        resize_area(src, nw, nh)
    } else {
        resize_linear(src, nw, nh)
    };
    GrayBuf {
        data,
        width: nw,
        height: nh,
    }
}

/// Size with the long side scaled to `side`: `(max(1, round(w*s)), max(1, round(h*s)))`.
pub fn fit_long_side(w: usize, h: usize, side: usize) -> (usize, usize, f64) {
    let s = side as f64 / w.max(h) as f64;
    let nw = ((w as f64 * s).round() as usize).max(1);
    let nh = ((h as f64 * s).round() as usize).max(1);
    (nw, nh, s)
}

/// `np.rot90(img, k)`: `k` quarter turns counter-clockwise (negative = clockwise).
pub fn rot90(src: &Gray, k: i32) -> GrayBuf {
    let (w, h) = (src.width, src.height);
    let k = k.rem_euclid(4);
    let (ow, oh) = if k % 2 == 1 { (h, w) } else { (w, h) };
    let mut out = vec![0u8; w * h];
    for i in 0..oh {
        for j in 0..ow {
            let (r, c) = match k {
                0 => (i, j),
                1 => (j, w - 1 - i),
                2 => (h - 1 - i, w - 1 - j),
                _ => (h - 1 - j, i),
            };
            out[i * ow + j] = src.data[r * w + c];
        }
    }
    GrayBuf {
        data: out,
        width: ow,
        height: oh,
    }
}

/// `cv2.getRotationMatrix2D(center, degrees, 1.0)`: the 2x3 matrix that turns
/// a point counter-clockwise (on screen) by `degrees` about `center`.
pub fn rotation_matrix(cx: f64, cy: f64, degrees: f64) -> [[f64; 3]; 2] {
    let (b, a) = degrees.to_radians().sin_cos();
    [
        [a, b, (1.0 - a) * cx - b * cy],
        [-b, a, b * cx + (1.0 - a) * cy],
    ]
}

pub fn apply(m: &[[f64; 3]; 2], x: f64, y: f64) -> (f64, f64) {
    (
        m[0][0] * x + m[0][1] * y + m[0][2],
        m[1][0] * x + m[1][1] * y + m[1][2],
    )
}

pub fn invert(m: &[[f64; 3]; 2]) -> [[f64; 3]; 2] {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let (a, b, c, d) = (m[1][1] / det, -m[0][1] / det, -m[1][0] / det, m[0][0] / det);
    [
        [a, b, -(a * m[0][2] + b * m[1][2])],
        [c, d, -(c * m[0][2] + d * m[1][2])],
    ]
}

/// Pixels outside the source image in [`warp_affine`].
#[derive(Clone, Copy, PartialEq)]
pub enum Border {
    /// `BORDER_CONSTANT` with this value.
    Constant(u8),
    /// `BORDER_REPLICATE`: the nearest edge pixel.
    Replicate,
}

/// `cv2.warpAffine(src, m, (ow, oh), INTER_LINEAR, border)`: each output pixel
/// samples the source bilinearly at `m^-1 (x, y)`.
pub fn warp_affine(src: &Gray, m: &[[f64; 3]; 2], ow: usize, oh: usize, border: Border) -> GrayBuf {
    let inv = invert(m);
    let (w, h) = (src.width as isize, src.height as isize);
    let px = |x: isize, y: isize| -> f32 {
        if x >= 0 && y >= 0 && x < w && y < h {
            src.data[(y * w + x) as usize] as f32
        } else {
            match border {
                Border::Constant(v) => v as f32,
                Border::Replicate => {
                    src.data[(y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize] as f32
                }
            }
        }
    };
    let mut out = vec![0u8; ow * oh];
    for y in 0..oh {
        for x in 0..ow {
            let (sx, sy) = apply(&inv, x as f64, y as f64);
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = ((sx - x0) as f32, (sy - y0) as f32);
            let (x0, y0) = (x0 as isize, y0 as isize);
            let top = px(x0, y0) * (1.0 - fx) + px(x0 + 1, y0) * fx;
            let bot = px(x0, y0 + 1) * (1.0 - fx) + px(x0 + 1, y0 + 1) * fx;
            out[y * ow + x] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
        }
    }
    GrayBuf {
        data: out,
        width: ow,
        height: oh,
    }
}

/// `cv2.resize(..., INTER_AREA)` for shrinking: each output pixel is the
/// average of the source area it covers, with fractional edge weights.
pub fn resize_area(src: &Gray, nw: usize, nh: usize) -> Vec<u8> {
    let (w, h) = (src.width, src.height);
    let (sx, sy) = (w as f64 / nw as f64, h as f64 / nh as f64);
    let xw = area_weights(w, nw, sx);
    let yw = area_weights(h, nh, sy);
    // Horizontal pass into f32 rows, then vertical.
    let mut tmp = vec![0f32; nw * h];
    for y in 0..h {
        let row = &src.data[y * w..(y + 1) * w];
        for (x, (start, ws)) in xw.iter().enumerate() {
            let mut acc = 0f32;
            for (k, wt) in ws.iter().enumerate() {
                acc += row[start + k] as f32 * wt;
            }
            tmp[y * nw + x] = acc;
        }
    }
    let mut out = vec![0u8; nw * nh];
    for (y, (start, ws)) in yw.iter().enumerate() {
        for x in 0..nw {
            let mut acc = 0f32;
            for (k, wt) in ws.iter().enumerate() {
                acc += tmp[(start + k) * nw + x] * wt;
            }
            out[y * nw + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Per output index: first source index and normalised overlap weights.
fn area_weights(src_len: usize, dst_len: usize, scale: f64) -> Vec<(usize, Vec<f32>)> {
    (0..dst_len)
        .map(|d| {
            let lo = d as f64 * scale;
            let hi = ((d + 1) as f64 * scale).min(src_len as f64);
            let first = lo.floor() as usize;
            let last = (hi.ceil() as usize).min(src_len);
            let mut ws: Vec<f32> = (first..last)
                .map(|i| ((i as f64 + 1.0).min(hi) - (i as f64).max(lo)) as f32)
                .collect();
            let sum: f32 = ws.iter().sum();
            for w in &mut ws {
                *w /= sum;
            }
            (first, ws)
        })
        .collect()
}

/// `cv2.resize(..., INTER_LINEAR)`: half-pixel centres, edge clamping.
pub fn resize_linear(src: &Gray, nw: usize, nh: usize) -> Vec<u8> {
    let (w, h) = (src.width, src.height);
    let (sx, sy) = (w as f32 / nw as f32, h as f32 / nh as f32);
    let taps = |n: usize, s: f32, len: usize| -> Vec<(usize, usize, f32)> {
        (0..n)
            .map(|d| {
                let f = (d as f32 + 0.5) * s - 0.5;
                let i = f.floor();
                let t = f - i;
                let i0 = (i as isize).clamp(0, len as isize - 1) as usize;
                let i1 = (i as isize + 1).clamp(0, len as isize - 1) as usize;
                (i0, i1, t)
            })
            .collect()
    };
    let xt = taps(nw, sx, w);
    let yt = taps(nh, sy, h);
    let mut out = vec![0u8; nw * nh];
    for (y, &(y0, y1, ty)) in yt.iter().enumerate() {
        for (x, &(x0, x1, tx)) in xt.iter().enumerate() {
            let p = |yy: usize, xx: usize| src.data[yy * w + xx] as f32;
            let top = p(y0, x0) * (1.0 - tx) + p(y0, x1) * tx;
            let bot = p(y1, x0) * (1.0 - tx) + p(y1, x1) * tx;
            out[y * nw + x] = (top * (1.0 - ty) + bot * ty).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Bounding stats of one connected component, in mask pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Component {
    pub left: usize,
    pub width: usize,
    pub area: usize,
}

/// 4-connected components of a binary mask (`cv2.connectedComponentsWithStats`
/// with `connectivity=4`). Returns a label per pixel (0 = background, labels
/// start at 1 in raster order of first appearance) and per-label stats at index
/// `label - 1`.
pub fn components4(mask: &[bool], w: usize, h: usize) -> (Vec<u32>, Vec<Component>) {
    let mut labels = vec![0u32; w * h];
    let mut stats: Vec<Component> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..w * h {
        if !mask[start] || labels[start] != 0 {
            continue;
        }
        let id = stats.len() as u32 + 1;
        labels[start] = id;
        stack.push(start);
        let (mut min_x, mut max_x, mut area) = (usize::MAX, 0usize, 0usize);
        while let Some(p) = stack.pop() {
            let (x, y) = (p % w, p / w);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            area += 1;
            let mut visit = |q: usize| {
                if mask[q] && labels[q] == 0 {
                    labels[q] = id;
                    stack.push(q);
                }
            };
            if x > 0 {
                visit(p - 1);
            }
            if x + 1 < w {
                visit(p + 1);
            }
            if y > 0 {
                visit(p - w);
            }
            if y + 1 < h {
                visit(p + w);
            }
        }
        stats.push(Component {
            left: min_x,
            width: max_x - min_x + 1,
            area,
        });
    }
    (labels, stats)
}

/// `np.percentile(b, 100 * q)` (linear interpolation) for each `q`, computed
/// from the histogram of a `u8` slice.
pub fn percentiles(b: &[u8], qs: [f64; 3]) -> [f32; 3] {
    let n = b.len();
    let mut hist = [0usize; 256];
    for &v in b {
        hist[v as usize] += 1;
    }
    let mut cum = [0usize; 256];
    let mut run = 0;
    for (i, c) in hist.iter().enumerate() {
        run += c;
        cum[i] = run;
    }
    // searchsorted(cum, k, side="right"): first value whose cum exceeds k.
    let at = |k: usize| cum.partition_point(|&c| c <= k) as f32;
    qs.map(|q| {
        let x = q * (n - 1) as f64;
        let i = x.floor();
        let g = (x - i) as f32;
        let lo = at(i as usize);
        let hi = at((i as usize + 1).min(n - 1));
        let d = hi - lo;
        // numpy's lerp: from the nearer end.
        if g >= 0.5 {
            hi - d * (1.0 - g)
        } else {
            lo + d * g
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_integer_ratio_is_box_average() {
        let data = [0u8, 10, 20, 30, 40, 50, 60, 70];
        let src = Gray {
            data: &data,
            width: 4,
            height: 2,
        };
        // 2x2 boxes: (0+10+40+50)/4 = 25, (20+30+60+70)/4 = 45.
        assert_eq!(resize_area(&src, 2, 1), vec![25, 45]);
    }

    #[test]
    fn area_fractional_ratio_weights_edges() {
        // 3 -> 2: each output covers 1.5 source pixels.
        let data = [0u8, 90, 180];
        let src = Gray {
            data: &data,
            width: 3,
            height: 1,
        };
        // out0 = (0*1 + 90*0.5)/1.5 = 30; out1 = (90*0.5 + 180*1)/1.5 = 150.
        assert_eq!(resize_area(&src, 2, 1), vec![30, 150]);
    }

    #[test]
    fn linear_upscale_uses_half_pixel_centres() {
        let data = [0u8, 100];
        let src = Gray {
            data: &data,
            width: 2,
            height: 1,
        };
        // cv2.resize([[0,100]], (4,1), INTER_LINEAR) == [0, 25, 75, 100].
        assert_eq!(resize_linear(&src, 4, 1), vec![0, 25, 75, 100]);
    }

    #[test]
    fn components_are_4_connected() {
        // 1 1 0
        // 0 0 1   (the diagonal pixel is its own component)
        let mask = [true, true, false, false, false, true];
        let (labels, stats) = components4(&mask, 3, 2);
        assert_eq!(labels, vec![1, 1, 0, 0, 0, 2]);
        assert_eq!(
            stats[0],
            Component {
                left: 0,
                width: 2,
                area: 2
            }
        );
        assert_eq!(
            stats[1],
            Component {
                left: 2,
                width: 1,
                area: 1
            }
        );
    }

    #[test]
    fn rot90_matches_numpy() {
        let data = [0u8, 1, 2, 3, 4, 5];
        let g = Gray {
            data: &data,
            width: 3,
            height: 2,
        };
        // np.rot90([[0,1,2],[3,4,5]], 1) == [[2,5],[1,4],[0,3]]
        let r = rot90(&g, 1);
        assert_eq!((r.width, r.height, r.data), (2, 3, vec![2, 5, 1, 4, 0, 3]));
        // np.rot90(.., -1) == [[3,0],[4,1],[5,2]]
        assert_eq!(rot90(&g, -1).data, vec![3, 0, 4, 1, 5, 2]);
        assert_eq!(rot90(&g, 2).data, vec![5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn rotation_matrix_matches_cv2() {
        let m = rotation_matrix(1.5, 1.0, 10.0);
        let want = [
            [0.984807753012208, 0.17364817766693033, -0.15085980718524236],
            [-0.17364817766693033, 0.984807753012208, 0.2756645134881875],
        ];
        for r in 0..2 {
            for c in 0..3 {
                assert!((m[r][c] - want[r][c]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn warp_matches_cv2_within_one_level() {
        let data: Vec<u8> = (0..12).map(|v| (v * 20) as u8).collect();
        let g = Gray {
            data: &data,
            width: 4,
            height: 3,
        };
        // cv2 output; its fixed-point bilinear differs from float by at most 1.
        let rep = warp_affine(
            &g,
            &rotation_matrix(2.0, 1.5, 10.0),
            4,
            3,
            Border::Replicate,
        );
        let want = [6u8, 26, 47, 76, 55, 89, 122, 154, 132, 164, 198, 218];
        for (a, b) in rep.data.iter().zip(want) {
            assert!(a.abs_diff(b) <= 1, "{:?}", rep.data);
        }
        let cst = warp_affine(
            &g,
            &rotation_matrix(2.0, 1.5, 10.0),
            4,
            3,
            Border::Constant(0),
        );
        let want = [4u8, 22, 47, 57, 55, 89, 122, 143, 124, 164, 198, 182];
        for (a, b) in cst.data.iter().zip(want) {
            assert!(a.abs_diff(b) <= 1, "{:?}", cst.data);
        }
    }

    #[test]
    fn percentiles_match_numpy() {
        // np.percentile([0,10,20,30,40], [2, 50, 98]) == [0.8, 20.0, 39.2]
        let p = percentiles(&[0, 10, 20, 30, 40], [0.02, 0.5, 0.98]);
        assert!((p[0] - 0.8).abs() < 1e-4, "{p:?}");
        assert!((p[1] - 20.0).abs() < 1e-4, "{p:?}");
        assert!((p[2] - 39.2).abs() < 1e-4, "{p:?}");
    }
}
