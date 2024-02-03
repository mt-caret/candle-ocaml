use candle_core::{shape::Dim, DType, Error, D};
use ocaml_interop::{
    ocaml_alloc_variant, ocaml_unpack_variant, FromOCaml, OCaml, OCamlInt, OCamlRuntime, ToOCaml,
};

pub struct CandleDType {}

unsafe impl FromOCaml<CandleDType> for DType {
    fn from_ocaml(v: OCaml<CandleDType>) -> Self {
        let result = ocaml_unpack_variant! {
            v => {
                DType::U8,
                DType::U32,
                DType::I64,
                DType::BF16,
                DType::F16,
                DType::F32,
                DType::F64,
            }
        };
        result.expect("Failure when unpacking an OCaml<TimeUnit> variant into PolarsTimeUnit (unexpected tag value")
    }
}

unsafe impl ToOCaml<CandleDType> for DType {
    fn to_ocaml<'a>(&self, cr: &'a mut OCamlRuntime) -> OCaml<'a, CandleDType> {
        ocaml_alloc_variant! {
            cr, self => {
                DType::U8,
                DType::U32,
                DType::I64,
                DType::BF16,
                DType::F16,
                DType::F32,
                DType::F64,
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum CandleDim {
    Usize(usize),
    D(D),
}

impl Dim for CandleDim {
    fn to_index(&self, shape: &candle_core::Shape, op: &'static str) -> candle_core::Result<usize> {
        match self {
            CandleDim::Usize(usize) => usize.to_index(shape, op),
            CandleDim::D(d) => d.to_index(shape, op),
        }
    }

    fn to_index_plus_one(
        &self,
        shape: &candle_core::Shape,
        op: &'static str,
    ) -> candle_core::Result<usize> {
        match self {
            CandleDim::Usize(usize) => usize.to_index_plus_one(shape, op),
            CandleDim::D(d) => d.to_index_plus_one(shape, op),
        }
    }
}

pub struct CandleDimResult(pub candle_core::Result<CandleDim>);

unsafe impl FromOCaml<OCamlInt> for CandleDimResult {
    fn from_ocaml(v: OCaml<OCamlInt>) -> Self {
        let v: i64 = v.to_rust();

        CandleDimResult(match v {
            -1 => Ok(CandleDim::D(D::Minus1)),
            -2 => Ok(CandleDim::D(D::Minus2)),
            _ => {
                if v < 0 {
                    Err(Error::Msg(
                        "Dim doesn't support negative numbers other than -1 or -2".to_string(),
                    ))
                } else {
                    Ok(CandleDim::Usize(v as usize))
                }
            }
        })
    }
}
