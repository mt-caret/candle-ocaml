open! Core

type t

module Location = struct
  type t =
    | Cpu
    | Cuda of { gpu_id : int }
    | Metal of { gpu_id : int }
  [@@deriving sexp]
end

external cpu : unit -> t = "rust_device_cpu"

let cpu = cpu ()

(* TODO: currently all these functions segfaults if a negative number is passed;
   we should probably fix that (ideally on the Rust side) *)
external cuda : ordinal:int -> (t, string) result = "rust_device_new_cuda"

let cuda ~ordinal = cuda ~ordinal |> Result.map_error ~f:Error.of_string

external metal : ordinal:int -> (t, string) result = "rust_device_new_metal"

let metal ~ordinal = metal ~ordinal |> Result.map_error ~f:Error.of_string

external cuda_if_available
  :  ordinal:int
  -> (t, string) result
  = "rust_device_cuda_if_available"

let cuda_if_available ~ordinal =
  cuda_if_available ~ordinal |> Result.map_error ~f:Error.of_string
;;

external location : t -> Location.t = "rust_device_location"

let sexp_of_t t = [%sexp_of: Location.t] (location t)
