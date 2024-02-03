// Below code is a tweaked version of the macro defined in polars-ocaml;
// probably should release as a separate crate or upstream into ocaml-interop.
use proc_macro::TokenStream;
use proc_macro2::Span;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse_macro_input;
use syn::punctuated::Punctuated;

fn drop_lifetime_parameters(type_: syn::Type) -> syn::Type {
    match type_ {
        syn::Type::Path(
            ref path @ syn::TypePath {
                path: syn::Path { ref segments, .. },
                ..
            },
        ) => {
            let segments = segments
                .clone()
                .into_iter()
                .map(
                    |ref path_segment @ syn::PathSegment { ref arguments, .. }| match arguments
                        .clone()
                    {
                        syn::PathArguments::AngleBracketed(
                            ref generic_arguments @ syn::AngleBracketedGenericArguments {
                                ref args,
                                ..
                            },
                        ) => {
                            let args = args
                                .clone()
                                .into_iter()
                                .filter(|generic_argument| {
                                    !matches!(generic_argument, syn::GenericArgument::Lifetime(_))
                                })
                                .collect();
                            syn::PathSegment {
                                arguments: syn::PathArguments::AngleBracketed(
                                    syn::AngleBracketedGenericArguments {
                                        args,
                                        ..generic_arguments.clone()
                                    },
                                ),
                                ..path_segment.clone()
                            }
                        }
                        _ => path_segment.clone(),
                    },
                )
                .collect();

            syn::Type::Path(syn::TypePath {
                path: syn::Path {
                    segments,
                    ..path.path
                },
                ..path.clone()
            })
        }
        _ => type_,
    }
}

// TODO: a common mistake when using the attribute macro is to specify OCaml<_>
// for arguments or OCamlRef<_> for return types, which should never happen.
// In these cases, the macro should probably point out this issue and suggest
// what to do (use the other type).

fn try_ocaml_interop_export_implementation(item_fn: syn::ItemFn) -> Result<TokenStream2, String> {
    let (inner_function_name, inner_function) = {
        let item_fn = item_fn.clone();
        let inner_function_name =
            syn::Ident::new(&format!("{}_inner", item_fn.sig.ident), Span::call_site());
        (
            inner_function_name.clone(),
            syn::ItemFn {
                sig: syn::Signature {
                    ident: inner_function_name,
                    ..item_fn.sig
                },
                ..item_fn
            },
        )
    };

    let mut inputs_iter = item_fn.sig.inputs.iter().map(|fn_arg| match fn_arg {
        syn::FnArg::Receiver(_) => {
            Err("'self' arguments are not supported by ocaml_interop_export")
        }
        syn::FnArg::Typed(pat_type) => Ok(pat_type.clone()),
    });

    let first_argument = inputs_iter
        .next()
        .ok_or("expected at least one argument")??;

    // The first argument to the function corresponds to the OCaml runtime.
    let runtime_name = match *first_argument.pat {
        syn::Pat::Ident(pat_ident) => Ok(pat_ident.ident),
        _ => Err("expected identifier corresponding to runtime for first argument"),
    }?;

    // The remaining arguments are stripped of their types and converted to
    // `RawOCaml` values.
    let new_inputs: Punctuated<_, _> = inputs_iter
        .clone()
        .map(|pat_type| {
            let pat_type = pat_type?;
            Ok::<_, String>(syn::FnArg::Typed(syn::PatType {
                ty: syn::parse2(quote! {
                    ::ocaml_interop::RawOCaml
                })
                .unwrap(),
                ..pat_type
            }))
        })
        .collect::<Result<_, _>>()?;
    let number_of_arguments = new_inputs.len();

    let signature = syn::Signature {
        inputs: new_inputs,
        output: syn::parse2(quote! {
            -> ::ocaml_interop::RawOCaml
        })
        .unwrap(),
        ..item_fn.sig.clone()
    };

    // We take each non-runtime argument to the function and convert them to the
    // appropriate Rust type.
    let locals = inputs_iter
        .map(|pat_type| {
            let pat_type = pat_type?;
            match *pat_type.pat {
                syn::Pat::Ident(pat_ident) => {
                    let ident = pat_ident.ident;
                    let ty = drop_lifetime_parameters(*pat_type.ty);
                    Ok((
                        ident.clone(),
                        quote! {
                            let #ident: #ty = &::ocaml_interop::BoxRoot::new(unsafe {
                                OCaml::new(cr, #ident)
                            });
                        },
                    ))
                }
                _ => Err("expected ident"),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (local_idents, local_decls): (Vec<_>, Vec<_>) = locals.into_iter().unzip();

    let return_type = match item_fn.sig.output.clone() {
        syn::ReturnType::Default => Err("functions with no return type are not supported"),
        syn::ReturnType::Type(_, ty) => Ok(drop_lifetime_parameters(*ty)),
    }?;

    let native_function = quote! {
        #[no_mangle]
        pub extern "C" #signature {
            let #runtime_name = unsafe {
                &mut ::ocaml_interop::OCamlRuntime::recover_handle()
            };

            #( #local_decls )*

            {
                let return_value: #return_type = #inner_function_name(#runtime_name, #( #local_idents ),*);

                unsafe { return_value.raw() }
            }
        }
    };

    // We need to generate different functions for the bytecode and native
    // versions of the function if there is more than a certain number of arguments.
    // See https://v2.ocaml.org/manual/intfc.html#ss:c-prim-impl for details.
    if number_of_arguments > 5 {
        let native_function_name = item_fn.sig.ident;

        let bytecode_function_name = syn::Ident::new(
            &format!("{}_bytecode", native_function_name),
            Span::call_site(),
        );

        let arguments = (0..number_of_arguments).map(|i| {
            quote! {
                argv[#i]
            }
        });

        Ok(quote! {
            #inner_function

            #native_function

            #[no_mangle]
            pub extern "C" fn #bytecode_function_name(
            argv: *const ::ocaml_interop::RawOCaml,
            argn: isize,
            ) -> ::ocaml_interop::RawOCaml {
                if argn as usize != #number_of_arguments {
                    panic!("expected {} arguments, got {}", #number_of_arguments, argn);
                }

                let argv = unsafe { ::std::slice::from_raw_parts(argv, argn as usize) };

                #native_function_name(#( #arguments ),*)
            }
        })
    } else {
        Ok(quote! {
            #inner_function

            #native_function
        })
    }
}

fn ocaml_interop_export_implementation(item_fn: syn::ItemFn) -> TokenStream2 {
    match try_ocaml_interop_export_implementation(item_fn) {
        Ok(expanded) => expanded,
        Err(error_message) => quote! { compile_error!(#error_message); },
    }
}

#[proc_macro_attribute]
pub fn ocaml_interop_export(_args: TokenStream, annotated_item: TokenStream) -> TokenStream {
    let item_fn = parse_macro_input!(annotated_item as syn::ItemFn);

    let expanded = ocaml_interop_export_implementation(item_fn);

    TokenStream::from(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::expect;
    use proc_macro2::TokenStream as TokenStream2;

    fn pretty_print_item(item: &TokenStream2) -> String {
        let file: syn::File = syn::parse2(item.clone()).unwrap();

        prettyplease::unparse(&file)
    }

    fn apply_macro_and_pretty_print(input: TokenStream2) -> String {
        let item_fn = syn::parse2(input).unwrap();
        let expanded = ocaml_interop_export_implementation(item_fn);
        pretty_print_item(&expanded)
    }

    #[test]
    fn test_simple_function() {
        let macro_output = apply_macro_and_pretty_print(quote! {
            fn rust_tensor_to_string<'a>(
                cr: &'a mut &'a mut OCamlRuntime,
                tensor: OCamlRef<'a, DynBox<Tensor>>,
            ) -> OCaml<'a, String> {
                let Abstract(tensor) = tensor.to_rust(cr);

                tensor.to_string().to_ocaml(cr)
            }
        });

        expect![[r#"
            fn rust_tensor_to_string_inner<'a>(
                cr: &'a mut &'a mut OCamlRuntime,
                tensor: OCamlRef<'a, DynBox<Tensor>>,
            ) -> OCaml<'a, String> {
                let Abstract(tensor) = tensor.to_rust(cr);
                tensor.to_string().to_ocaml(cr)
            }
            #[no_mangle]
            pub extern "C" fn rust_tensor_to_string<'a>(
                tensor: ::ocaml_interop::RawOCaml,
            ) -> ::ocaml_interop::RawOCaml {
                let cr = unsafe { &mut ::ocaml_interop::OCamlRuntime::recover_handle() };
                let tensor: OCamlRef<DynBox<Tensor>> = &::ocaml_interop::BoxRoot::new(unsafe {
                    OCaml::new(cr, tensor)
                });
                {
                    let return_value: OCaml<String> = rust_tensor_to_string_inner(cr, tensor);
                    unsafe { return_value.raw() }
                }
            }
        "#]]
        .assert_eq(&macro_output);

        let macro_output = apply_macro_and_pretty_print(quote! {
            fn rust_tensor_arange<'a>(
                cr: &'a mut &'a mut OCamlRuntime,
                start: OCamlRef<'a, OCamlFloat>,
                end: OCamlRef<'a, OCamlFloat>,
                device: OCamlRef<'a, DynBox<Rc<Device>>>,
            ) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
                let start: f64 = start.to_rust(cr);
                let end: f64 = end.to_rust(cr);
                let Abstract(device) = device.to_rust(cr);

                Tensor::arange(start, end, device.borrow())
                    .map(Abstract)
                    .map_err(|err: Error| err.to_string())
                    .to_ocaml(cr)
            }
        });

        // TODO: the extern version of the function should not have a lifetime
        // parameter.
        expect![[r#"
            fn rust_tensor_arange_inner<'a>(
                cr: &'a mut &'a mut OCamlRuntime,
                start: OCamlRef<'a, OCamlFloat>,
                end: OCamlRef<'a, OCamlFloat>,
                device: OCamlRef<'a, DynBox<Rc<Device>>>,
            ) -> OCaml<'a, Result<DynBox<Tensor>, String>> {
                let start: f64 = start.to_rust(cr);
                let end: f64 = end.to_rust(cr);
                let Abstract(device) = device.to_rust(cr);
                Tensor::arange(start, end, device.borrow())
                    .map(Abstract)
                    .map_err(|err: Error| err.to_string())
                    .to_ocaml(cr)
            }
            #[no_mangle]
            pub extern "C" fn rust_tensor_arange<'a>(
                start: ::ocaml_interop::RawOCaml,
                end: ::ocaml_interop::RawOCaml,
                device: ::ocaml_interop::RawOCaml,
            ) -> ::ocaml_interop::RawOCaml {
                let cr = unsafe { &mut ::ocaml_interop::OCamlRuntime::recover_handle() };
                let start: OCamlRef<OCamlFloat> = &::ocaml_interop::BoxRoot::new(unsafe {
                    OCaml::new(cr, start)
                });
                let end: OCamlRef<OCamlFloat> = &::ocaml_interop::BoxRoot::new(unsafe {
                    OCaml::new(cr, end)
                });
                let device: OCamlRef<DynBox<Rc<Device>>> = &::ocaml_interop::BoxRoot::new(unsafe {
                    OCaml::new(cr, device)
                });
                {
                    let return_value: OCaml<Result<DynBox<Tensor>, String>> = rust_tensor_arange_inner(
                        cr,
                        start,
                        end,
                        device,
                    );
                    unsafe { return_value.raw() }
                }
            }
        "#]]
        .assert_eq(&macro_output);
    }
}
