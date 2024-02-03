use crate::{candle_types::CandleDimResult, interop::Abstract};
use candle_core::{DType, Device, Error, Module, Tensor};
use candle_nn::{linear, loss, ops, optim, var_builder, var_map, Linear, Optimizer};
use candle_ocaml_macros::ocaml_interop_export;
use ocaml_interop::{
    impl_from_ocaml_variant, DynBox, OCaml, OCamlFloat, OCamlInt, OCamlRef, OCamlRuntime, ToOCaml,
};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::rc::Rc;

enum CandleModule {
    Linear(Abstract<Linear>),
}

impl_from_ocaml_variant! {
    CandleModule {
        CandleModule::Linear(linear: DynBox<Linear>)
    }
}

#[ocaml_interop_export]
fn rust_var_map_new<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    _unit: OCamlRef<()>,
) -> OCaml<'a, DynBox<var_map::VarMap>> {
    Abstract(var_map::VarMap::new()).to_ocaml(cr)
}

// TODO: the 'static lifetime parameter in the dynbox is required to get things
// to compile, but als seems pretty sus; is this actually safe...?
//
// Do we actually need a struct that keeps track of references between
// VarBuilders via refcounting to ensure safety? Would that even work?
#[ocaml_interop_export]
fn rust_var_builder<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    varmap: OCamlRef<DynBox<var_map::VarMap>>,
    device: OCamlRef<DynBox<Rc<Device>>>,
) -> OCaml<'a, DynBox<var_builder::VarBuilder<'static>>> {
    let Abstract(varmap) = varmap.to_rust(cr);
    let Abstract(device) = device.to_rust(cr);

    Abstract(var_builder::VarBuilder::from_varmap(
        &varmap,
        DType::F64,
        device.borrow(),
    ))
    .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_var_builder_push_prefix<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    vs: OCamlRef<DynBox<var_builder::VarBuilder<'static>>>,
    prefix: OCamlRef<String>,
) -> OCaml<'a, DynBox<var_builder::VarBuilder<'static>>> {
    let Abstract(vs) = vs.to_rust(cr);
    let prefix: String = prefix.to_rust(cr);

    Abstract(vs.push_prefix(prefix)).to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_sgd_new<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    varmap: OCamlRef<DynBox<var_map::VarMap>>,
    learning_rate: OCamlRef<OCamlFloat>,
) -> OCaml<'a, Result<DynBox<Rc<RefCell<optim::SGD>>>, String>> {
    let Abstract(varmap) = varmap.to_rust(cr);
    let learning_rate = learning_rate.to_rust(cr);

    optim::SGD::new(varmap.all_vars(), learning_rate)
        .map(|sgd| Abstract(Rc::new(RefCell::new(sgd))))
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_sgd_backwards_step<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    sgd: OCamlRef<DynBox<Rc<RefCell<optim::SGD>>>>,
    loss: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<(), String>> {
    let Abstract(sgd) = sgd.to_rust(cr);
    let Abstract(loss) = loss.to_rust(cr);

    let result = sgd
        .borrow_mut()
        .backward_step(&loss)
        .map_err(|err: Error| err.to_string());

    result.to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_loss_nll<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    inp: OCamlRef<DynBox<Tensor>>,
    target: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(inp) = inp.to_rust(cr);
    let Abstract(target) = target.to_rust(cr);

    loss::nll(&inp, &target)
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_linear<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    in_dim: OCamlRef<OCamlInt>,
    out_dim: OCamlRef<OCamlInt>,
    vs: OCamlRef<DynBox<var_builder::VarBuilder<'static>>>,
) -> OCaml<'a, Result<DynBox<Linear>, String>> {
    let in_dim: i64 = in_dim.to_rust(cr);
    let out_dim: i64 = out_dim.to_rust(cr);
    let Abstract(vs) = vs.to_rust(cr);

    linear(in_dim as usize, out_dim as usize, vs)
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_forward<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    module: OCamlRef<CandleModule>,
    tensor: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let module: CandleModule = module.to_rust(cr);
    let Abstract(tensor) = tensor.to_rust(cr);

    (match module {
        CandleModule::Linear(Abstract(linear)) => linear.forward(&tensor),
    })
    .map(Abstract)
    .map_err(|err: Error| err.to_string())
    .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_ops_log_softmax<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    tensor: OCamlRef<DynBox<Tensor>>,
    dim: OCamlRef<OCamlInt>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(tensor) = tensor.to_rust(cr);
    let CandleDimResult(dim_result) = dim.to_rust(cr);

    dim_result
        .and_then(|dim| ops::log_softmax(&tensor, dim))
        .map(Abstract)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
}
