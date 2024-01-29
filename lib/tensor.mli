open! Core

(** [Tensor.t] represents a multi-dimensional matrix in candle. *)
type t

val arange : start:float -> end_:float -> t Or_error.t
val to_string : t -> string
