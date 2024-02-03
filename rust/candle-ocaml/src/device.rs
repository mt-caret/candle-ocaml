use crate::candle_types::CandleDeviceLocation;
use crate::interop::Abstract;
use candle_core::{Device, Error};
use ocaml_interop::{DynBox, OCaml, OCamlInt, OCamlRef, ToOCaml};
use std::rc::Rc;

ocaml_interop::ocaml_export! {
    fn rust_device_cpu(cr, _unit: OCamlRef<()>) -> OCaml<DynBox<Rc<Device>>> {
        Abstract(Rc::new(Device::Cpu)).to_ocaml(cr)
    }

    fn rust_device_new_cuda(cr, ordinal: OCamlRef<OCamlInt>) -> OCaml<Result<DynBox<Rc<Device>>, String>> {
        let ordinal: i64 = ordinal.to_rust(cr);

        Device::new_cuda(ordinal as usize)
        .map(|device| Abstract(Rc::new(device)))
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
    }

    fn rust_device_new_metal(cr, ordinal: OCamlRef<OCamlInt>) -> OCaml<Result<DynBox<Rc<Device>>, String>> {
        let ordinal: i64 = ordinal.to_rust(cr);

        Device::new_metal(ordinal as usize)
        .map(|device| Abstract(Rc::new(device)))
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
    }

    fn rust_device_cuda_if_available(cr, ordinal: OCamlRef<OCamlInt>) -> OCaml<Result<DynBox<Rc<Device>>,String>> {
        let ordinal: i64 = ordinal.to_rust(cr);

        Device::cuda_if_available(ordinal as usize)
        .map(|device| Abstract(Rc::new(device)))
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
    }

    fn rust_device_location(cr, device: OCamlRef<DynBox<Rc<Device>>>) -> OCaml<CandleDeviceLocation> {
        let Abstract(device) = device.to_rust(cr);

        device.as_ref().location().to_ocaml(cr)
    }
}
