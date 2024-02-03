open! Core

module Sgd = struct
  type t

  external create
    :  Var.Map.t
    -> learning_rate:float
    -> (t, string) result
    = "rust_sgd_new"

  let create varmap ~learning_rate =
    create varmap ~learning_rate |> Result.map_error ~f:Error.of_string
  ;;

  external backwards_step
    :  t
    -> loss:Tensor.t
    -> (unit, string) result
    = "rust_sgd_backwards_step"

  let backwards_step t ~loss =
    backwards_step t ~loss |> Result.map_error ~f:Error.of_string
  ;;
end
