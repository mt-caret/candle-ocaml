open! Core

type t

external arange
  :  start:float
  -> end_:float
  -> device:Device.t
  -> (t, string) result
  = "rust_tensor_arange"

let arange ~start ~end_ ~device =
  arange ~start ~end_ ~device |> Result.map_error ~f:Error.of_string
;;

external randn
  :  mean:float
  -> std:float
  -> shape:int list
  -> device:Device.t
  -> (t, string) result
  = "rust_tensor_randn"

let randn ~mean ~std ~shape ~device =
  randn ~mean ~std ~shape ~device |> Result.map_error ~f:Error.of_string
;;

external from_float_array
  :  floatarray
  -> shape:int list
  -> device:Device.t
  -> (t, string) result
  = "rust_tensor_from_float_array"

let from_float_array ?shape array ~device =
  let shape = Option.value shape ~default:[ Stdlib.Float.Array.length array ] in
  from_float_array array ~shape ~device |> Result.map_error ~f:Error.of_string
;;

external from_uniform_array
  :  float Uniform_array.t
  -> shape:int list
  -> device:Device.t
  -> (t, string) result
  = "rust_tensor_from_uniform_array"

let from_array ?shape array ~device =
  let shape = Option.value shape ~default:[ Array.length array ] in
  if Obj.tag (Obj.repr array) = Obj.double_array_tag
  then from_float_array (Obj.magic array) ~shape ~device
  else
    from_uniform_array (Obj.magic array) ~shape ~device
    |> Result.map_error ~f:Error.of_string
;;

external from_bigarray
  :  (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> shape:int list
  -> device:Device.t
  -> (t, string) result
  = "rust_tensor_from_bigarray"

let from_bigarray ?shape bigarray ~device =
  let shape = Option.value shape ~default:[ Bigarray.Array1.dim bigarray ] in
  from_bigarray bigarray ~shape ~device |> Result.map_error ~f:Error.of_string
;;

external to_scalar : t -> (float, string) result = "rust_tensor_to_scalar"

let to_scalar t = to_scalar t |> Result.map_error ~f:Error.of_string

external to_dtype : t -> dtype:Dtype.t -> (t, string) result = "rust_tensor_to_dtype"

let to_dtype t ~dtype = to_dtype t ~dtype |> Result.map_error ~f:Error.of_string

external to_device : t -> device:Device.t -> (t, string) result = "rust_tensor_to_device"

let to_device t ~device = to_device t ~device |> Result.map_error ~f:Error.of_string

external eq : t -> t -> (t, string) result = "rust_tensor_eq"
external ne : t -> t -> (t, string) result = "rust_tensor_ne"
external lt : t -> t -> (t, string) result = "rust_tensor_lt"
external gt : t -> t -> (t, string) result = "rust_tensor_gt"
external le : t -> t -> (t, string) result = "rust_tensor_le"
external ge : t -> t -> (t, string) result = "rust_tensor_ge"

let ( = ) t1 t2 = eq t1 t2 |> Result.map_error ~f:Error.of_string
let ( <> ) t1 t2 = ne t1 t2 |> Result.map_error ~f:Error.of_string
let ( < ) t1 t2 = lt t1 t2 |> Result.map_error ~f:Error.of_string
let ( > ) t1 t2 = gt t1 t2 |> Result.map_error ~f:Error.of_string
let ( <= ) t1 t2 = le t1 t2 |> Result.map_error ~f:Error.of_string
let ( >= ) t1 t2 = ge t1 t2 |> Result.map_error ~f:Error.of_string

external matmul : t -> t -> (t, string) result = "rust_tensor_matmul"

let matmul t1 t2 = matmul t1 t2 |> Result.map_error ~f:Error.of_string

external relu : t -> (t, string) result = "rust_tensor_relu"

let relu t = relu t |> Result.map_error ~f:Error.of_string

external argmax : t -> dim:int -> (t, string) result = "rust_tensor_argmax"

let argmax t ~dim = argmax t ~dim |> Result.map_error ~f:Error.of_string

external argmin : t -> dim:int -> (t, string) result = "rust_tensor_argmin"

let argmin t ~dim = argmin t ~dim |> Result.map_error ~f:Error.of_string

external sum_all : t -> (t, string) result = "rust_tensor_sum_all"

let sum_all t = sum_all t |> Result.map_error ~f:Error.of_string

external mean_all : t -> (t, string) result = "rust_tensor_mean_all"

let mean_all t = mean_all t |> Result.map_error ~f:Error.of_string

external shape : t -> int list = "rust_tensor_shape"
external dtype : t -> Dtype.t = "rust_tensor_dtype"
external to_string : t -> string = "rust_tensor_to_string"

external save
  :  t
  -> name:string
  -> filename:string
  -> (unit, string) result
  = "rust_tensor_save"

let save t ~name ~filename = save t ~name ~filename |> Result.map_error ~f:Error.of_string

external save_many
  :  (string * t) list
  -> filename:string
  -> (unit, string) result
  = "rust_tensor_save_many"

let save_many ts ~filename =
  Map.to_alist ts |> save_many ~filename |> Result.map_error ~f:Error.of_string
;;

external load_many
  :  filename:string
  -> device:Device.t
  -> ((string * t) list, string) result
  = "rust_tensor_load_many"

let load_many ~filename ~device =
  load_many ~filename ~device
  |> Result.map_error ~f:Error.of_string
  |> Or_error.map ~f:String.Map.of_alist_exn
;;

(* TODO: replicate Debug instance in Rust *)
let sexp_of_t t = [%sexp { shape : int list = shape t; dtype : Dtype.t = dtype t }]
