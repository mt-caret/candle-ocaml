open! Core

module Mnist = struct
  type t =
    { train_images : Tensor.t
    ; train_labels : Tensor.t
    ; test_images : Tensor.t
    ; test_labels : Tensor.t
    ; labels : int
    }

  external load : dir:string option -> (t, string) result = "rust_mnist_load"

  let load ~dir = load ~dir |> Result.map_error ~f:Error.of_string
end
