open! Core

module Map = struct
  type t

  external create : unit -> t = "rust_var_map_new"
end

module Builder = struct
  type t

  external of_varmap : Map.t -> device:Device.t -> t = "rust_var_builder"
  external push_prefix : t -> string -> t = "rust_var_builder_push_prefix"
end
