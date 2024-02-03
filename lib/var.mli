open! Core

module Map : sig
  type t

  val create : unit -> t
end

module Builder : sig
  type t

  val of_varmap : Map.t -> device:Device.t -> t
  val push_prefix : t -> string -> t
end
