open! Core

module Mnist : sig
  type t =
    { train_images : Tensor.t
    ; train_labels : Tensor.t
    ; test_images : Tensor.t
    ; test_labels : Tensor.t
    ; labels : int
    }

  val load : dir:string option -> t Or_error.t
end
