open! Core

external log_softmax
  :  Tensor.t
  -> dim:int
  -> (Tensor.t, string) result
  = "rust_ops_log_softmax"

let log_softmax tensor ~dim =
  log_softmax tensor ~dim |> Result.map_error ~f:Error.of_string
;;
