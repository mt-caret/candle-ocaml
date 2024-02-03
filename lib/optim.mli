open! Core

module Sgd : sig
  type t

  val create : Var.Map.t -> lr:float -> t Or_error.t
  val backwards_step : t -> loss:Tensor.t -> unit Or_error.t
end

module Adam_w : sig
  type t

  val create
    :  ?lr:float
    -> ?beta1:float
    -> ?beta2:float
    -> ?eps:float
    -> ?weight_decay:float
    -> Var.Map.t
    -> t Or_error.t

  val backwards_step : t -> loss:Tensor.t -> unit Or_error.t
end
