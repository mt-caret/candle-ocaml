(lang dune 3.13)

(name candle)

(generate_opam_files true)

(source
 (github mt-caret/candle-ocaml))

(authors "mtakeda")

(maintainers "mtakeda")

(license MIT)

(documentation https://github.com/mt-caret/polars-ocaml/)

(package
 (name candle)
 (synopsis "OCaml bindings to then candle ML framework")
 (description "A longer description")
 (depends ocaml dune core)
 (tags
  (machine-learning rust)))
