// We ignore this lint since we can't really refactor these types to be "better"
#![allow(clippy::type_complexity)]
use crate::candle_types::{CandleDType, CandleDimResult};
use crate::interop::Abstract;
use candle_core::{safetensors, Device, Error, Tensor};
use candle_ocaml_macros::ocaml_interop_export;
use ocaml_interop::{
    bigarray, BoxRoot, DynBox, OCaml, OCamlFloat, OCamlFloatArray, OCamlInt, OCamlList, OCamlRef,
    OCamlRuntime, OCamlUniformArray, ToOCaml,
};
use std::borrow::Borrow;
use std::rc::Rc;

#[ocaml_interop_export]
fn rust_tensor_arange<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    start: OCamlRef<OCamlFloat>,
    end: OCamlRef<OCamlFloat>,
    device: OCamlRef<DynBox<Rc<Device>>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let start: f64 = start.to_rust(cr);
    let end: f64 = end.to_rust(cr);
    let Abstract(device) = device.to_rust(cr);

    Tensor::arange(start, end, device.borrow())
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_randn<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    mean: OCamlRef<OCamlFloat>,
    std: OCamlRef<OCamlFloat>,
    shape: OCamlRef<OCamlList<OCamlInt>>,
    device: OCamlRef<DynBox<Rc<Device>>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let mean: f64 = mean.to_rust(cr);
    let std: f64 = std.to_rust(cr);
    let shape: Vec<i64> = shape.to_rust(cr);
    let shape: Vec<usize> = shape.into_iter().map(|n| n as usize).collect();
    let Abstract(device) = device.to_rust(cr);

    Tensor::randn(mean, std, shape, device.borrow())
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_from_float_array<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    array: OCamlRef<OCamlFloatArray>,
    shape: OCamlRef<OCamlList<OCamlInt>>,
    device: OCamlRef<DynBox<Rc<Device>>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let array: Vec<f64> = array.to_rust(cr);
    let shape: Vec<i64> = shape.to_rust(cr);
    let shape: Vec<usize> = shape.into_iter().map(|n| n as usize).collect();
    let Abstract(device) = device.to_rust(cr);

    Tensor::from_vec(array, shape, device.borrow())
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_from_uniform_array<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    array: OCamlRef<OCamlUniformArray<OCamlFloat>>,
    shape: OCamlRef<OCamlList<OCamlInt>>,
    device: OCamlRef<DynBox<Rc<Device>>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let array: Vec<f64> = array.to_rust(cr);
    let shape: Vec<i64> = shape.to_rust(cr);
    let shape: Vec<usize> = shape.into_iter().map(|n| n as usize).collect();
    let Abstract(device) = device.to_rust(cr);

    Tensor::from_vec(array, shape, device.borrow())
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_from_bigarray<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    bigarray: OCamlRef<bigarray::Array1<f64>>,
    shape: OCamlRef<OCamlList<OCamlInt>>,
    device: OCamlRef<DynBox<Rc<Device>>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let bigarray: BoxRoot<bigarray::Array1<f64>> = bigarray.to_boxroot(cr);
    let shape: Vec<i64> = shape.to_rust(cr);
    let shape: Vec<usize> = shape.into_iter().map(|n| n as usize).collect();
    let Abstract(device) = device.to_rust(cr);

    Tensor::from_slice(bigarray.get(cr).borrow(), shape, device.borrow())
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_to_scalar<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<OCamlFloat, String>> {
    let Abstract(tensor) = tensor.to_rust(cr);

    tensor
        .to_scalar::<f64>()
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_to_dtype<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
    dtype: OCamlRef<CandleDType>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor) = tensor.to_rust(cr);
    let dtype = dtype.to_rust(cr);

    tensor
        .to_dtype(dtype)
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_to_device<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
    device: OCamlRef<DynBox<Rc<Device>>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor) = tensor.to_rust(cr);
    let Abstract(device) = device.to_rust(cr);

    tensor
        .to_device(device.borrow())
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_eq<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor1: OCamlRef<DynBox<Tensor>>,
    tensor2: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor1) = tensor1.to_rust(cr);
    let Abstract(tensor2) = tensor2.to_rust(cr);

    tensor1
        .eq(&tensor2)
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_ne<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor1: OCamlRef<DynBox<Tensor>>,
    tensor2: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor1) = tensor1.to_rust(cr);
    let Abstract(tensor2) = tensor2.to_rust(cr);

    tensor1
        .ne(&tensor2)
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_lt<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor1: OCamlRef<DynBox<Tensor>>,
    tensor2: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor1) = tensor1.to_rust(cr);
    let Abstract(tensor2) = tensor2.to_rust(cr);

    tensor1
        .lt(&tensor2)
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_gt<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor1: OCamlRef<DynBox<Tensor>>,
    tensor2: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor1) = tensor1.to_rust(cr);
    let Abstract(tensor2) = tensor2.to_rust(cr);

    tensor1
        .gt(&tensor2)
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_le<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor1: OCamlRef<DynBox<Tensor>>,
    tensor2: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor1) = tensor1.to_rust(cr);
    let Abstract(tensor2) = tensor2.to_rust(cr);

    tensor1
        .le(&tensor2)
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_ge<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor1: OCamlRef<DynBox<Tensor>>,
    tensor2: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor1) = tensor1.to_rust(cr);
    let Abstract(tensor2) = tensor2.to_rust(cr);

    tensor1
        .ge(&tensor2)
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_matmul<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor1: OCamlRef<DynBox<Tensor>>,
    tensor2: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor1) = tensor1.to_rust(cr);
    let Abstract(tensor2) = tensor2.to_rust(cr);

    tensor1
        .matmul(&tensor2)
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_relu<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor) = tensor.to_rust(cr);

    tensor
        .relu()
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_argmax<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
    dim: OCamlRef<OCamlInt>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor) = tensor.to_rust(cr);
    let CandleDimResult(dim_result) = dim.to_rust(cr);

    dim_result
        .and_then(|dim| tensor.argmax(dim))
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_argmin<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
    dim: OCamlRef<OCamlInt>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor) = tensor.to_rust(cr);
    let CandleDimResult(dim_result) = dim.to_rust(cr);

    dim_result
        .and_then(|dim| tensor.argmin(dim))
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_sum_all<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor) = tensor.to_rust(cr);

    tensor
        .sum_all()
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_mean_all<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor) = tensor.to_rust(cr);

    tensor
        .mean_all()
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_shape<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, OCamlList<OCamlInt>> {
    let Abstract(tensor) = tensor.to_rust(cr);

    tensor
        .shape()
        .clone()
        .into_dims()
        .into_iter()
        .map(|dim| dim as i64)
        .collect::<Vec<i64>>()
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_dim<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
    dim: OCamlRef<OCamlInt>,
) -> OCaml<'a, Result<OCamlInt, String>> {
    let Abstract(tensor) = tensor.to_rust(cr);
    let CandleDimResult(dim_result) = dim.to_rust(cr);

    dim_result
        .and_then(|dim| tensor.dim(dim).map(|dim| dim as i64))
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_dtype<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, CandleDType> {
    let Abstract(tensor) = tensor.to_rust(cr);

    tensor.dtype().to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_device<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, DynBox<Rc<Device>>> {
    let Abstract(tensor) = tensor.to_rust(cr);

    Abstract(Rc::new(tensor.device().clone())).to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_to_string<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, String> {
    let Abstract(tensor) = tensor.to_rust(cr);

    tensor.to_string().to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_save<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
    name: OCamlRef<String>,
    filename: OCamlRef<String>,
) -> OCaml<'a, Result<(), String>> {
    let Abstract(tensor) = tensor.to_rust(cr);
    let name: String = name.to_rust(cr);
    let filename: String = filename.to_rust(cr);

    tensor
        .save_safetensors(&name, filename)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_save_many<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensors: OCamlRef<OCamlList<(String, DynBox<Tensor>)>>,
    filename: OCamlRef<String>,
) -> OCaml<'a, Result<(), String>> {
    let tensors: Vec<(String, Abstract<Tensor>)> = tensors.to_rust(cr);
    let filename: String = filename.to_rust(cr);

    safetensors::save(
        &tensors
            .into_iter()
            .map(|(name, Abstract(tensor))| (name, tensor))
            .collect(),
        filename,
    )
    .map_err(|err: Error| err.to_string())
    .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_tensor_load_many<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    filename: OCamlRef<String>,
    device: OCamlRef<DynBox<Rc<Device>>>,
) -> OCaml<'a, Result<OCamlList<(String, DynBox<Tensor>)>, String>> {
    let filename: String = filename.to_rust(cr);
    let Abstract(device) = device.to_rust(cr);

    safetensors::load(filename, device.borrow())
        .map(|tensors| {
            tensors
                .into_iter()
                .map(|(name, tensor)| (name, Abstract(tensor)))
                .collect::<Vec<(String, Abstract<Tensor>)>>()
        })
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}
