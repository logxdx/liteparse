//! Minimal `.npz` reader for Indra's language-model tables: a zip of `.npy`
//! arrays (stored or deflated), little-endian `u64`, `i64`, `f32` or `f64`,
//! C order. Only what `lm.npz` holds is supported.

use std::collections::HashMap;
use std::io::{Cursor, Read};

use crate::error::LiteParseError;

fn err(msg: impl Into<String>) -> LiteParseError {
    LiteParseError::Ocr(format!("indra lm .npz: {}", msg.into()))
}

/// One array: its dtype descriptor (`<u8`, `<f4`, ...) and raw bytes.
pub struct Array {
    pub descr: String,
    pub data: Vec<u8>,
}

impl Array {
    fn elems<const N: usize>(
        &self,
        want: &[&str],
    ) -> Result<impl Iterator<Item = [u8; N]> + '_, LiteParseError> {
        if !want.contains(&self.descr.as_str()) {
            return Err(err(format!("dtype {} is not one of {want:?}", self.descr)));
        }
        Ok(self.data.as_chunks::<N>().0.iter().copied())
    }

    pub fn u64s(&self) -> Result<Vec<u64>, LiteParseError> {
        Ok(self.elems::<8>(&["<u8"])?.map(u64::from_le_bytes).collect())
    }

    pub fn f32s(&self) -> Result<Vec<f32>, LiteParseError> {
        Ok(self.elems::<4>(&["<f4"])?.map(f32::from_le_bytes).collect())
    }

    /// A scalar integer (`order`, `vocab_size`, `bits`).
    pub fn scalar_int(&self) -> Result<i64, LiteParseError> {
        match self.descr.as_str() {
            "<i8" => Ok(i64::from_le_bytes(
                self.data
                    .get(..8)
                    .ok_or_else(|| err("empty"))?
                    .try_into()
                    .unwrap(),
            )),
            "<u8" => Ok(u64::from_le_bytes(
                self.data
                    .get(..8)
                    .ok_or_else(|| err("empty"))?
                    .try_into()
                    .unwrap(),
            ) as i64),
            "<i4" => Ok(i32::from_le_bytes(
                self.data
                    .get(..4)
                    .ok_or_else(|| err("empty"))?
                    .try_into()
                    .unwrap(),
            ) as i64),
            d => Err(err(format!("scalar dtype {d}"))),
        }
    }
}

/// Parse one `.npy` file (format 1.0 / 2.0 / 3.0).
fn parse_npy(bytes: Vec<u8>) -> Result<Array, LiteParseError> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(err("not a .npy file"));
    }
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            let b = bytes.get(8..12).ok_or_else(|| err("short header"))?;
            (u32::from_le_bytes(b.try_into().unwrap()) as usize, 12)
        }
        v => return Err(err(format!("npy version {v}"))),
    };
    let header = std::str::from_utf8(
        bytes
            .get(start..start + hlen)
            .ok_or_else(|| err("short header"))?,
    )
    .map_err(|_| err("header is not UTF-8"))?;
    if header.contains("'fortran_order': True") {
        return Err(err("Fortran order is not supported"));
    }
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .ok_or_else(|| err("no descr"))?
        .to_owned();
    let mut data = bytes;
    data.drain(..start + hlen);
    Ok(Array { descr, data })
}

/// Read every array of an `.npz` file, keyed by name without `.npy`.
pub fn read(bytes: &[u8]) -> Result<HashMap<String, Array>, LiteParseError> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| err(e.to_string()))?;
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut f = zip.by_index(i).map_err(|e| err(e.to_string()))?;
        let name = f.name().trim_end_matches(".npy").to_owned();
        let mut buf = Vec::with_capacity(f.size() as usize);
        f.read_to_end(&mut buf).map_err(|e| err(e.to_string()))?;
        out.insert(name, parse_npy(buf)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(descr: &str, data: &[u8]) -> Vec<u8> {
        let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': (), }}");
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn parses_npy_arrays() {
        let a = parse_npy(npy(
            "<u8",
            &[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        ))
        .unwrap();
        assert_eq!(a.u64s().unwrap(), vec![1, 2]);
        let f = parse_npy(npy("<f4", &1.5f32.to_le_bytes())).unwrap();
        assert_eq!(f.f32s().unwrap(), vec![1.5]);
        let s = parse_npy(npy("<i8", &5i64.to_le_bytes())).unwrap();
        assert_eq!(s.scalar_int().unwrap(), 5);
        assert!(f.u64s().is_err());
    }
}
