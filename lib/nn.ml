open! Core

type linear
type module_ = Linear of linear [@@boxed]

external forward : module_ -> Tensor.t -> (Tensor.t, string) result = "rust_forward"

let forward module_ tensor = forward module_ tensor |> Result.map_error ~f:Error.of_string

module Linear = struct
  type t = linear

  external create
    :  in_dim:int
    -> out_dim:int
    -> Var.Builder.t
    -> (t, string) result
    = "rust_linear"

  let create vs ~in_dim ~out_dim =
    create ~in_dim ~out_dim vs |> Result.map_error ~f:Error.of_string
  ;;

  let forward t tensor = forward (Linear t) tensor
end
