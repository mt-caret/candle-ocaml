open! Core

module Linear : sig
  type t

  val create : Var.Builder.t -> in_dim:int -> out_dim:int -> t Or_error.t
  val forward : t -> Tensor.t -> Tensor.t Or_error.t
end

module Conv2d : sig
  type t

  val create
    :  ?padding:int
    -> ?stride:int
    -> ?dilation:int
    -> ?groups:int
    -> Var.Builder.t
    -> in_channels:int
    -> out_channels:int
    -> kernel_size:int
    -> t Or_error.t

  val forward : t -> Tensor.t -> Tensor.t Or_error.t
end

module Dropout : sig
  type t

  val create : drop_p:float -> t
  val forward : t -> Tensor.t -> train:bool -> Tensor.t Or_error.t
end
