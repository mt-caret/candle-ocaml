use crate::candle_types::CandleDeviceLocation;
use crate::interop::Abstract;
use candle_core::{Device, Error};
use candle_ocaml_macros::ocaml_interop_export;
use ocaml_interop::{DynBox, OCaml, OCamlInt, OCamlRef, OCamlRuntime, ToOCaml};
use std::rc::Rc;

#[ocaml_interop_export]
fn rust_device_cpu<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    _unit: OCamlRef<()>,
) -> OCaml<'a, DynBox<Rc<Device>>> {
    Abstract(Rc::new(Device::Cpu)).to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_device_new_cuda<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    ordinal: OCamlRef<OCamlInt>,
) -> OCaml<'a, Result<DynBox<Rc<Device>>, String>> {
    let ordinal: i64 = ordinal.to_rust(cr);

    Device::new_cuda(ordinal as usize)
        .map(|device| Abstract(Rc::new(device)))
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_device_new_metal<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    ordinal: OCamlRef<OCamlInt>,
) -> OCaml<'a, Result<DynBox<Rc<Device>>, String>> {
    let ordinal: i64 = ordinal.to_rust(cr);

    Device::new_metal(ordinal as usize)
        .map(|device| Abstract(Rc::new(device)))
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_device_cuda_if_available<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    ordinal: OCamlRef<OCamlInt>,
) -> OCaml<'a, Result<DynBox<Rc<Device>>, String>> {
    let ordinal: i64 = ordinal.to_rust(cr);

    Device::cuda_if_available(ordinal as usize)
        .map(|device| Abstract(Rc::new(device)))
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_device_location<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    device: OCamlRef<DynBox<Rc<Device>>>,
) -> OCaml<'a, CandleDeviceLocation> {
    let Abstract(device) = device.to_rust(cr);

    device.as_ref().location().to_ocaml(cr)
}
