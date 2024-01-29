use crate::interop::Abstract;
use candle_core::{Device, Error, Tensor};
use ocaml_interop::{DynBox, OCaml, OCamlFloat, OCamlRef, ToOCaml};

ocaml_interop::ocaml_export! {
    fn rust_tensor_arange(cr, start: OCamlRef<OCamlFloat>, end: OCamlRef<OCamlFloat>) -> OCaml<Result<DynBox<Tensor>, String>> {
        let start: f64 = start.to_rust(cr);
        let end: f64 = end.to_rust(cr);

        Tensor::arange(start, end, &Device::Cpu).map(Abstract).map_err(|err: Error|err.to_string()).to_ocaml(cr)
    }

    fn rust_tensor_to_string(cr, tensor: OCamlRef<DynBox<Tensor>>) -> OCaml<String> {
        let Abstract(tensor) = tensor.to_rust(cr);

        tensor.to_string().to_ocaml(cr)
    }
}
