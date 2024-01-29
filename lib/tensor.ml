open! Core

type t

external arange : start:float -> end_:float -> (t, string) result = "rust_tensor_arange"

let arange ~start ~end_ = arange ~start ~end_ |> Result.map_error ~f:Error.of_string

external to_string : t -> string = "rust_tensor_to_string"

let%expect_test "create" =
  arange ~start:0. ~end_:10. |> Or_error.ok_exn |> to_string |> print_endline;
  [%expect {|
    [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    Tensor[[10], f64] |}]
;;
