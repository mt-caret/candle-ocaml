// TODO: fix macro so that we can attach clippy allow attributes to functions
#![allow(clippy::too_many_arguments)]
use crate::{candle_types::CandleDimResult, interop::Abstract};
use candle_core::{DType, Device, Error, Module, Tensor};
use candle_nn::{
    conv2d, linear, loss, ops, optim, var_builder, var_map, Conv2d, Conv2dConfig, Dropout, Linear,
    Optimizer,
};
use candle_ocaml_macros::ocaml_interop_export;
use ocaml_interop::{
    impl_from_ocaml_variant, DynBox, OCaml, OCamlFloat, OCamlInt, OCamlRef, OCamlRuntime, ToOCaml,
};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::rc::Rc;

enum CandleModule {
    Linear(Abstract<Linear>),
    Conv2d(Abstract<Conv2d>),
}

impl_from_ocaml_variant! {
    CandleModule {
        CandleModule::Linear(linear: DynBox<Linear>),
        CandleModule::Conv2d(conv2d: DynBox<Conv2d>)
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
fn rust_adamw_new<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    varmap: OCamlRef<DynBox<var_map::VarMap>>,
    lr: OCamlRef<OCamlFloat>,
    beta1: OCamlRef<OCamlFloat>,
    beta2: OCamlRef<OCamlFloat>,
    eps: OCamlRef<OCamlFloat>,
    weight_decay: OCamlRef<OCamlFloat>,
) -> OCaml<'a, Result<DynBox<Rc<RefCell<optim::AdamW>>>, String>> {
    let Abstract(varmap) = varmap.to_rust(cr);
    let lr = lr.to_rust(cr);
    let beta1 = beta1.to_rust(cr);
    let beta2 = beta2.to_rust(cr);
    let eps = eps.to_rust(cr);
    let weight_decay = weight_decay.to_rust(cr);

    optim::AdamW::new(
        varmap.all_vars(),
        optim::ParamsAdamW {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        },
    )
    .map(|adamw| Abstract(Rc::new(RefCell::new(adamw))))
    .map_err(|err: Error| err.to_string())
    .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_adamw_backwards_step<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    adamw: OCamlRef<DynBox<Rc<RefCell<optim::AdamW>>>>,
    loss: OCamlRef<DynBox<Tensor>>,
) -> OCaml<'a, Result<(), String>> {
    let Abstract(adamw) = adamw.to_rust(cr);
    let Abstract(loss) = loss.to_rust(cr);

    let result = adamw
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
fn rust_conv2d<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    in_channels: OCamlRef<OCamlInt>,
    out_channels: OCamlRef<OCamlInt>,
    kernel_size: OCamlRef<OCamlInt>,
    padding: OCamlRef<OCamlInt>,
    stride: OCamlRef<OCamlInt>,
    dilation: OCamlRef<OCamlInt>,
    groups: OCamlRef<OCamlInt>,
    vs: OCamlRef<DynBox<var_builder::VarBuilder<'static>>>,
) -> OCaml<'a, Result<DynBox<Conv2d>, String>> {
    let in_channels: i64 = in_channels.to_rust(cr);
    let out_channels: i64 = out_channels.to_rust(cr);
    let kernel_size: i64 = kernel_size.to_rust(cr);
    let padding: i64 = padding.to_rust(cr);
    let stride: i64 = stride.to_rust(cr);
    let dilation: i64 = dilation.to_rust(cr);
    let groups: i64 = groups.to_rust(cr);
    let Abstract(vs) = vs.to_rust(cr);

    conv2d(
        in_channels as usize,
        out_channels as usize,
        kernel_size as usize,
        Conv2dConfig {
            padding: padding as usize,
            stride: stride as usize,
            dilation: dilation as usize,
            groups: groups as usize,
        },
        vs,
    )
    .map(Abstract)
    .map_err(|err: Error| err.to_string())
    .to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_dropout_new<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    drop_p: OCamlRef<OCamlFloat>,
) -> OCaml<'a, DynBox<Rc<Dropout>>> {
    let drop_p: f64 = drop_p.to_rust(cr);

    Abstract(Rc::new(Dropout::new(drop_p as f32))).to_ocaml(cr)
}

#[ocaml_interop_export]
fn rust_dropout_forward<'a>(
    cr: &'a mut &'a mut OCamlRuntime,
    dropout: OCamlRef<DynBox<Rc<Dropout>>>,
    tensor: OCamlRef<DynBox<Tensor>>,
    train: OCamlRef<bool>,
) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
    let Abstract(dropout) = dropout.to_rust(cr);
    let Abstract(tensor) = tensor.to_rust(cr);
    let train = train.to_rust(cr);

    dropout
        .as_ref()
        .forward(&tensor, train)
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
        CandleModule::Conv2d(Abstract(conv2d)) => conv2d.forward(&tensor),
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
