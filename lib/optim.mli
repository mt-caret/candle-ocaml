open! Core

module Sgd : sig
  type t

  val create : Var.Map.t -> learning_rate:float -> t Or_error.t
  val backwards_step : t -> loss:Tensor.t -> unit Or_error.t
end
