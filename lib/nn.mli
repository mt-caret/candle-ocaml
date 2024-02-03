open! Core

module Linear : sig
  type t

  val create : Var.Builder.t -> in_dim:int -> out_dim:int -> t Or_error.t
  val forward : t -> Tensor.t -> Tensor.t Or_error.t
end
