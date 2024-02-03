open! Core

external nll : Tensor.t -> Tensor.t -> (Tensor.t, string) result = "rust_loss_nll"

let nll inp target = nll inp target |> Result.map_error ~f:Error.of_string
