// We ignore this lint since we can't really refactor these types to be "better"
#![allow(clippy::type_complexity)]
use crate::interop::Abstract;
use candle_core::{safetensors, Device, Error, Tensor};
use ocaml_interop::{
    bigarray, BoxRoot, DynBox, OCaml, OCamlFloat, OCamlFloatArray, OCamlInt, OCamlList, OCamlRef,
    OCamlUniformArray, ToOCaml,
};
use std::borrow::Borrow;

ocaml_interop::ocaml_export! {
    fn rust_tensor_arange(cr, start: OCamlRef<OCamlFloat>, end: OCamlRef<OCamlFloat>) -> OCaml<Result<DynBox<Tensor>, String>> {
        let start: f64 = start.to_rust(cr);
        let end: f64 = end.to_rust(cr);

        Tensor::arange(start, end, &Device::Cpu).map(Abstract).map_err(|err: Error| err.to_string()).to_ocaml(cr)
    }

    fn rust_tensor_randn(cr, mean: OCamlRef<OCamlFloat>, std: OCamlRef<OCamlFloat>, shape: OCamlRef<OCamlList<OCamlInt>>) -> OCaml<Result<DynBox<Tensor>, String>> {
        let mean: f64 = mean.to_rust(cr);
        let std: f64 = std.to_rust(cr);
        let shape: Vec<i64> = shape.to_rust(cr);
        let shape: Vec<usize> = shape.into_iter().map(|n| n as usize).collect();

        Tensor::randn(mean, std, shape, &Device::Cpu).map(Abstract).map_err(|err: Error| err.to_string()).to_ocaml(cr)
    }

    fn rust_tensor_from_float_array(cr, array: OCamlRef<OCamlFloatArray>, shape: OCamlRef<OCamlList<OCamlInt>>) -> OCaml<Result<DynBox<Tensor>, String>> {
        let array: Vec<f64> = array.to_rust(cr);
        let shape: Vec<i64> = shape.to_rust(cr);
        let shape: Vec<usize> = shape.into_iter().map(|n| n as usize).collect();

        Tensor::from_vec(array, shape, &Device::Cpu).map(Abstract).map_err(|err: Error| err.to_string()).to_ocaml(cr)
    }

    fn rust_tensor_from_uniform_array(cr, array: OCamlRef<OCamlUniformArray<OCamlFloat>>, shape: OCamlRef<OCamlList<OCamlInt>>) -> OCaml<Result<DynBox<Tensor>, String>> {
        let array: Vec<f64> = array.to_rust(cr);
        let shape: Vec<i64> = shape.to_rust(cr);
        let shape: Vec<usize> = shape.into_iter().map(|n| n as usize).collect();

        Tensor::from_vec(array, shape, &Device::Cpu).map(Abstract).map_err(|err: Error| err.to_string()).to_ocaml(cr)
    }

    fn rust_tensor_from_bigarray(cr, bigarray: OCamlRef<bigarray::Array1<f64>>, shape: OCamlRef<OCamlList<OCamlInt>>) -> OCaml<Result<DynBox<Tensor>, String>> {
        let bigarray: BoxRoot<bigarray::Array1<f64>> = bigarray.to_boxroot(cr);
        let shape: Vec<i64> = shape.to_rust(cr);
        let shape: Vec<usize> = shape.into_iter().map(|n| n as usize).collect();

        Tensor::from_slice(bigarray.get(cr).borrow(), shape, &Device::Cpu).map(Abstract).map_err(|err: Error| err.to_string()).to_ocaml(cr)
    }

    fn rust_tensor_matmul(cr, tensor1: OCamlRef<DynBox<Tensor>>, tensor2: OCamlRef<DynBox<Tensor>>) -> OCaml<Result<DynBox<Tensor>, String>> {
        let Abstract(tensor1) = tensor1.to_rust(cr);
        let Abstract(tensor2) = tensor2.to_rust(cr);

        tensor1.matmul(&tensor2).map(Abstract).map_err(|err: Error| err.to_string()).to_ocaml(cr)
    }

    fn rust_tensor_relu(cr, tensor: OCamlRef<DynBox<Tensor>>) -> OCaml<Result<DynBox<Tensor>, String>> {
        let Abstract(tensor) = tensor.to_rust(cr);

        tensor.relu().map(Abstract).map_err(|err: Error| err.to_string()).to_ocaml(cr)
    }

    fn rust_tensor_to_string(cr, tensor: OCamlRef<DynBox<Tensor>>) -> OCaml<String> {
        let Abstract(tensor) = tensor.to_rust(cr);

        tensor.to_string().to_ocaml(cr)
    }

    fn rust_tensor_save(cr, tensor: OCamlRef<DynBox<Tensor>>, name: OCamlRef<String>, filename: OCamlRef<String>) -> OCaml<Result<(), String>> {
        let Abstract(tensor) = tensor.to_rust(cr);
        let name: String = name.to_rust(cr);
        let filename: String = filename.to_rust(cr);

        tensor.save_safetensors(&name, filename).map_err(|err: Error| err.to_string()).to_ocaml(cr)
    }

    fn rust_tensor_save_many(cr, tensors: OCamlRef<OCamlList<(String, DynBox<Tensor>)>>, filename: OCamlRef<String>) -> OCaml<Result<(), String>> {
        let tensors: Vec<(String, Abstract<Tensor>)> = tensors.to_rust(cr);
        let filename: String = filename.to_rust(cr);

        safetensors::save(
            &tensors.into_iter()
            .map(|(name, Abstract(tensor))| (name, tensor)).collect(),
            filename)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
    }

    fn rust_tensor_load_many(cr, filename: OCamlRef<String>) -> OCaml<Result<OCamlList<(String, DynBox<Tensor>)>, String>> {
        let filename: String = filename.to_rust(cr);

        safetensors::load(filename, &Device::Cpu)
        .map(|tensors|
            tensors.into_iter()
            .map(|(name, tensor)| (name, Abstract(tensor)))
            .collect::<Vec<(String, Abstract<Tensor>)>>())
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
    }
}
