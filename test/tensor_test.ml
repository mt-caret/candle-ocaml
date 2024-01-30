open! Core
open! Candle

let print t = Tensor.to_string t |> print_endline

let%expect_test "create" =
  let open Or_error.Let_syntax in
  Or_error.ok_exn
  @@
  let%bind t = Tensor.arange ~start:(-5.) ~end_:5. in
  print t;
  [%expect
    {|
    [-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.]
    Tensor[[10], f64] |}];
  let%map t = Tensor.relu t in
  print t;
  [%expect {|
    [0., 0., 0., 0., 0., 0., 1., 2., 3., 4.]
    Tensor[[10], f64] |}]
;;

let%expect_test "create from arrays" =
  let open Or_error.Let_syntax in
  Or_error.ok_exn
  @@
  let%bind t = Tensor.from_array [| 0.; 1.; 2.; 3.; 4.; 5. |] ~shape:[ 2; 3 ] in
  print t;
  [%expect {|
    [[0., 1., 2.],
     [3., 4., 5.]]
    Tensor[[2, 3], f64] |}];
  let%map t =
    Tensor.from_float_array
      (Stdlib.Float.Array.of_list [ 0.; 1.; 2.; 3.; 4.; 5. ])
      ~shape:[ 2; 3 ]
  in
  print t;
  [%expect {|
    [[0., 1., 2.],
     [3., 4., 5.]]
    Tensor[[2, 3], f64] |}]
;;

let%expect_test "saving and loading tensors" =
  let open Or_error.Let_syntax in
  Or_error.ok_exn
  @@ Filename_extended.with_temp_dir "candle-ocaml" "tensor" ~f:(fun temp_dir ->
    let filename = temp_dir ^/ "tensor.safetensors" in
    let%bind t = Tensor.arange ~start:0. ~end_:10. in
    let%bind () = Tensor.save t ~name:"tensor_name" ~filename in
    let%map () =
      Tensor.load_many ~filename
      >>| Map.to_alist
      >>| List.iter ~f:(fun (name, tensor) ->
        print_endline [%string "%{name}: %{tensor#Tensor}"])
    in
    [%expect
      {|
      tensor_name: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
      Tensor[[10], f64] |}])
;;
