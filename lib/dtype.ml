open! Core

type t =
  | U8
  | U32
  | I64
  | BF16
  | F16
  | F32
  | F64
[@@deriving sexp, enumerate]
