open! Core

(** [Tensor.t] represents a multi-dimensional matrix in candle. *)
type t

val arange : start:float -> end_:float -> t Or_error.t
val randn : mean:float -> std:float -> shape:int list -> t Or_error.t
val from_array : ?shape:int list -> float array -> t Or_error.t
val from_float_array : ?shape:int list -> floatarray -> t Or_error.t
val matmul : t -> t -> t Or_error.t
val relu : t -> t Or_error.t
val to_string : t -> string

(** [save] saves a single tensor in safetensors format *)
val save : t -> name:string -> filename:string -> unit Or_error.t

(** [save_many] saves multiple tensors in safetensors format *)
val save_many : t String.Map.t -> filename:string -> unit Or_error.t

(** [load_many] loads tensors saved in safetensors format *)
val load_many : filename:string -> t String.Map.t Or_error.t
