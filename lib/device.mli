open! Core

type t [@@deriving sexp_of]

module Location : sig
  type t =
    | Cpu
    | Cuda of { gpu_id : int }
    | Metal of { gpu_id : int }
  [@@deriving sexp]
end

val cpu : t
val cuda : ordinal:int -> t Or_error.t
val metal : ordinal:int -> t Or_error.t
val cuda_if_available : ordinal:int -> t Or_error.t
val location : t -> Location.t
