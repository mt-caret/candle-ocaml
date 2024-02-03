open! Core

(** [Tensor.t] represents a multi-dimensional matrix in candle. *)
type t [@@deriving sexp_of]

val arange : start:float -> end_:float -> device:Device.t -> t Or_error.t
val randn : mean:float -> std:float -> shape:int list -> device:Device.t -> t Or_error.t
val from_array : ?shape:int list -> float array -> device:Device.t -> t Or_error.t
val from_float_array : ?shape:int list -> floatarray -> device:Device.t -> t Or_error.t

(** NOTE: [from_bigarray] unfortunately incurs a copy. *)
val from_bigarray
  :  ?shape:int list
  -> (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> device:Device.t
  -> t Or_error.t

val to_scalar : t -> float Or_error.t
val to_dtype : t -> dtype:Dtype.t -> t Or_error.t
val to_device : t -> device:Device.t -> t Or_error.t
val ( = ) : t -> t -> t Or_error.t
val ( <> ) : t -> t -> t Or_error.t
val ( < ) : t -> t -> t Or_error.t
val ( > ) : t -> t -> t Or_error.t
val ( <= ) : t -> t -> t Or_error.t
val ( >= ) : t -> t -> t Or_error.t
val matmul : t -> t -> t Or_error.t
val relu : t -> t Or_error.t
val argmax : t -> dim:int -> t Or_error.t
val argmin : t -> dim:int -> t Or_error.t
val sum_all : t -> t Or_error.t
val mean_all : t -> t Or_error.t
val shape : t -> int list
val dtype : t -> Dtype.t
val to_string : t -> string

(** [save] saves a single tensor in safetensors format *)
val save : t -> name:string -> filename:string -> unit Or_error.t

(** [save_many] saves multiple tensors in safetensors format *)
val save_many : t String.Map.t -> filename:string -> unit Or_error.t

(** [load_many] loads tensors saved in safetensors format *)
val load_many : filename:string -> device:Device.t -> t String.Map.t Or_error.t
