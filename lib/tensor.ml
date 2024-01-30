open! Core

type t

external arange : start:float -> end_:float -> (t, string) result = "rust_tensor_arange"

let arange ~start ~end_ = arange ~start ~end_ |> Result.map_error ~f:Error.of_string

external randn
  :  mean:float
  -> std:float
  -> shape:int list
  -> (t, string) result
  = "rust_tensor_randn"

let randn ~mean ~std ~shape =
  randn ~mean ~std ~shape |> Result.map_error ~f:Error.of_string
;;

external from_float_array
  :  floatarray
  -> shape:int list
  -> (t, string) result
  = "rust_tensor_from_float_array"

let from_float_array ?shape array =
  let shape = Option.value shape ~default:[ Stdlib.Float.Array.length array ] in
  from_float_array array ~shape |> Result.map_error ~f:Error.of_string
;;

external from_uniform_array
  :  float Uniform_array.t
  -> shape:int list
  -> (t, string) result
  = "rust_tensor_from_uniform_array"

let from_array ?shape array =
  let shape = Option.value shape ~default:[ Array.length array ] in
  if Obj.tag (Obj.repr array) = Obj.double_array_tag
  then from_float_array (Obj.magic array) ~shape
  else from_uniform_array (Obj.magic array) ~shape |> Result.map_error ~f:Error.of_string
;;

external matmul : t -> t -> (t, string) result = "rust_tensor_matmul"

let matmul t1 t2 = matmul t1 t2 |> Result.map_error ~f:Error.of_string

external relu : t -> (t, string) result = "rust_tensor_relu"

let relu t = relu t |> Result.map_error ~f:Error.of_string

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
  -> ((string * t) list, string) result
  = "rust_tensor_load_many"

let load_many ~filename =
  load_many ~filename
  |> Result.map_error ~f:Error.of_string
  |> Or_error.map ~f:String.Map.of_alist_exn
;;

let%expect_test "create" =
  arange ~start:0. ~end_:10. |> Or_error.ok_exn |> to_string |> print_endline;
  [%expect {|
    [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    Tensor[[10], f64] |}]
;;
(* randn ~mean:0. ~std:1. ~shape:[ 1 ] |> Or_error.ok_exn |> to_string |> print_endline; *)
