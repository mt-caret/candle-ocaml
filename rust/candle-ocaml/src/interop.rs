use ocaml_interop::{DynBox, FromOCaml, OCaml, OCamlRuntime, ToOCaml};
use std::borrow::Borrow;

// Below struct + trait implementations are borrowed from polars-ocaml; there's
// gotta be a better way to do this...
#[derive(Debug, Clone)]
pub struct Abstract<T>(pub T);

unsafe impl<T: 'static + Clone> FromOCaml<DynBox<T>> for Abstract<T> {
    fn from_ocaml(v: OCaml<DynBox<T>>) -> Self {
        Abstract(Borrow::<T>::borrow(&v).clone())
    }
}

unsafe impl<T: 'static + Clone> ToOCaml<DynBox<T>> for Abstract<T> {
    fn to_ocaml<'a>(&self, cr: &'a mut OCamlRuntime) -> OCaml<'a, DynBox<T>> {
        // TODO: I don't fully understand why ToOCaml takes a &self, since that
        // prevents us from using box_value without a clone() call.
        OCaml::box_value(cr, self.0.clone())
    }
}
