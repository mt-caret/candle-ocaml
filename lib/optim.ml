open! Core

module Sgd = struct
  type t

  external create : Var.Map.t -> lr:float -> (t, string) result = "rust_sgd_new"

  let create varmap ~lr = create varmap ~lr |> Result.map_error ~f:Error.of_string

  external backwards_step
    :  t
    -> loss:Tensor.t
    -> (unit, string) result
    = "rust_sgd_backwards_step"

  let backwards_step t ~loss =
    backwards_step t ~loss |> Result.map_error ~f:Error.of_string
  ;;
end

module Adam_w = struct
  type t

  external create
    :  Var.Map.t
    -> lr:float
    -> beta1:float
    -> beta2:float
    -> eps:float
    -> weight_decay:float
    -> (t, string) result
    = "rust_adamw_new_bytecode" "rust_adamw_new"

  let create
    ?(lr = 0.001)
    ?(beta1 = 0.9)
    ?(beta2 = 0.999)
    ?(eps = 1e-8)
    ?(weight_decay = 0.01)
    varmap
    =
    create varmap ~lr ~beta1 ~beta2 ~eps ~weight_decay
    |> Result.map_error ~f:Error.of_string
  ;;

  external backwards_step
    :  t
    -> loss:Tensor.t
    -> (unit, string) result
    = "rust_adamw_backwards_step"

  let backwards_step t ~loss =
    backwards_step t ~loss |> Result.map_error ~f:Error.of_string
  ;;
end
