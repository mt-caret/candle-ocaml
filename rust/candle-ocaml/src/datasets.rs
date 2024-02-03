use crate::interop::Abstract;
use candle_core::{Error, Tensor};
use candle_datasets::vision;
use ocaml_interop::{impl_to_ocaml_record, DynBox, OCaml, OCamlInt, OCamlRef, ToOCaml};

struct CandleVisionDataset {
    train_images: Tensor,
    train_labels: Tensor,
    test_images: Tensor,
    test_labels: Tensor,
    labels: usize,
}

impl CandleVisionDataset {
    fn create(
        vision::Dataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
            labels,
        }: vision::Dataset,
    ) -> Self {
        CandleVisionDataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
            labels,
        }
    }
}

impl_to_ocaml_record! {
    CandleVisionDataset {
        train_images: DynBox<Tensor> => Abstract(train_images.clone()),
        train_labels: DynBox<Tensor> => Abstract(train_labels.clone()),
        test_images: DynBox<Tensor> => Abstract(test_images.clone()),
        test_labels: DynBox<Tensor> => Abstract(test_labels.clone()),
        labels: OCamlInt => *labels as i64,
    }
}

ocaml_interop::ocaml_export! {
    fn rust_mnist_load(cr, dir: OCamlRef<Option<String>>) -> OCaml<Result<CandleVisionDataset,String>> {
        let dir: Option<String> = dir.to_rust(cr);

        (match dir {
            Some(dir) => vision::mnist::load_dir(dir),
            None => vision::mnist::load(),
        })
        .map(CandleVisionDataset::create)
        .map_err(|err: Error| err.to_string())
        .to_ocaml(cr)
    }
}
