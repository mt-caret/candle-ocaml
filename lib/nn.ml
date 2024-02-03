open! Core

type linear
type conv2d

type module_ =
  | Linear of linear
  | Conv2d of conv2d

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

module Conv2d = struct
  type t = conv2d

  external create
    :  in_channels:int
    -> out_channels:int
    -> kernel_size:int
    -> padding:int
    -> stride:int
    -> dilation:int
    -> groups:int
    -> Var.Builder.t
    -> (t, string) result
    = "rust_conv2d_bytecode" "rust_conv2d"

  let create
    ?(padding = 0)
    ?(stride = 1)
    ?(dilation = 1)
    ?(groups = 1)
    vs
    ~in_channels
    ~out_channels
    ~kernel_size
    =
    create ~in_channels ~out_channels ~kernel_size ~padding ~stride ~dilation ~groups vs
    |> Result.map_error ~f:Error.of_string
  ;;

  let forward t tensor = forward (Conv2d t) tensor
end

module Dropout = struct
  type t

  external create : drop_p:float -> t = "rust_dropout_new"

  external forward
    :  t
    -> Tensor.t
    -> train:bool
    -> (Tensor.t, string) result
    = "rust_dropout_forward"

  let forward t tensor ~train =
    forward t tensor ~train |> Result.map_error ~f:Error.of_string
  ;;
end
