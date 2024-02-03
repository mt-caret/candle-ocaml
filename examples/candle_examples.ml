(* A port of Candle's minst-training example:
   https://github.com/huggingface/candle/blob/9e824ec810fbe490f21b7404058b6cb47d24c6cf/candle-examples/examples/mnist-training/main.rs
*)
open! Core
open! Candle

let image_dim = 784
let labels = 10

module Mlp = struct
  type t =
    { ln1 : Nn.Linear.t
    ; ln2 : Nn.Linear.t
    }

  let create vs =
    let open Or_error.Let_syntax in
    let%bind ln1 =
      Nn.Linear.create (Var.Builder.push_prefix vs "ln1") ~in_dim:image_dim ~out_dim:100
    in
    let%map ln2 =
      Nn.Linear.create (Var.Builder.push_prefix vs "ln2") ~in_dim:100 ~out_dim:labels
    in
    { ln1; ln2 }
  ;;

  let forward t tensor =
    let open Or_error.Let_syntax in
    Nn.Linear.forward t.ln1 tensor >>= Tensor.relu >>= Nn.Linear.forward t.ln2
  ;;
end

let mlp_training_loop (dataset : Dataset.Mnist.t) ~lr ~epochs ~device =
  let open Or_error.Let_syntax in
  let%bind train_labels =
    Tensor.to_dtype dataset.train_labels ~dtype:U32 >>= Tensor.to_device ~device
  in
  let%bind train_images =
    Tensor.to_dtype dataset.train_images ~dtype:F64 >>= Tensor.to_device ~device
  in
  let varmap = Var.Map.create () in
  let vs = Var.Builder.of_varmap varmap ~device in
  let%bind model = Mlp.create vs in
  let%bind sgd = Optim.Sgd.create varmap ~lr in
  let%bind test_labels =
    Tensor.to_dtype dataset.test_labels ~dtype:U32 >>= Tensor.to_device ~device
  in
  let%bind test_images =
    Tensor.to_dtype dataset.test_images ~dtype:F64 >>= Tensor.to_device ~device
  in
  let num_test_labels = Tensor.shape test_labels |> List.hd_exn in
  List.range 1 (epochs + 1)
  |> List.fold_result ~init:() ~f:(fun () epoch ->
    print_endline [%string "Epoch %{epoch#Int}: start"];
    let%bind logits = Mlp.forward model train_images in
    let%bind log_sm = Ops.log_softmax logits ~dim:(-1) in
    let%bind loss = Loss.nll log_sm train_labels in
    let%bind () = Optim.Sgd.backwards_step sgd ~loss in
    let%bind test_logits = Mlp.forward model test_images in
    let%bind sum_ok =
      Tensor.argmax test_logits ~dim:(-1)
      >>= Tensor.( = ) test_labels
      >>= Tensor.to_dtype ~dtype:F64
      >>= Tensor.sum_all
      >>= Tensor.to_scalar
    in
    let%bind loss = Tensor.to_scalar loss in
    let test_accuracy = 100. *. sum_ok /. Int.to_float num_test_labels in
    print_endline
      [%string
        "Epoch %{epoch#Int}: train loss: %{loss#Float} test acc: %{test_accuracy#Float}"];
    (* TODO: It's pretty sad we need to do something like this :/ *)
    if epoch % 10 = 0 then Gc.full_major ();
    return ())
;;

module ConvNet = struct
  type t =
    { conv1 : Nn.Conv2d.t
    ; conv2 : Nn.Conv2d.t
    ; fc1 : Nn.Linear.t
    ; fc2 : Nn.Linear.t
    ; dropout : Nn.Dropout.t
    }

  let create vs =
    let open Or_error.Let_syntax in
    let%bind conv1 =
      Nn.Conv2d.create
        ~in_channels:1
        ~out_channels:32
        ~kernel_size:5
        (Var.Builder.push_prefix vs "c1")
    in
    let%bind conv2 =
      Nn.Conv2d.create
        ~in_channels:32
        ~out_channels:64
        ~kernel_size:5
        (Var.Builder.push_prefix vs "c2")
    in
    let%bind fc1 =
      Nn.Linear.create ~in_dim:1024 ~out_dim:1024 (Var.Builder.push_prefix vs "fc1")
    in
    let%bind fc2 =
      Nn.Linear.create ~in_dim:1024 ~out_dim:labels (Var.Builder.push_prefix vs "fc2")
    in
    let dropout = Nn.Dropout.create ~drop_p:0.5 in
    return { conv1; conv2; fc1; fc2; dropout }
  ;;

  let forward t tensor ~train =
    let open Or_error.Let_syntax in
    let%bind batch_size =
      match Tensor.shape tensor with
      | [ batch_size; _image_dim ] -> Ok batch_size
      | _ -> Or_error.error_s [%message "unexpected tensor shape" (tensor : Tensor.t)]
    in
    Tensor.reshape tensor ~shape:[ batch_size; 1; 28; 28 ]
    >>= Nn.Conv2d.forward t.conv1
    >>= Tensor.max_pool2d ~size:(`Square 2)
    >>= Nn.Conv2d.forward t.conv2
    >>= Tensor.max_pool2d ~size:(`Square 2)
    >>= Tensor.flatten_from ~dim:1
    >>= Tensor.relu
    >>= Nn.Linear.forward t.fc1
    >>= Nn.Dropout.forward t.dropout ~train
    >>= Nn.Linear.forward t.fc2
  ;;
end

let convnet_training_loop (dataset : Dataset.Mnist.t) ~lr ~epochs ~device =
  let open Or_error.Let_syntax in
  let batch_size = 64 in
  let%bind train_labels =
    Tensor.to_dtype dataset.train_labels ~dtype:U32 >>= Tensor.to_device ~device
  in
  let%bind train_images =
    Tensor.to_dtype dataset.train_images ~dtype:F64 >>= Tensor.to_device ~device
  in
  let varmap = Var.Map.create () in
  let vs = Var.Builder.of_varmap varmap ~device in
  let%bind model = ConvNet.create vs in
  let%bind adamw = Optim.Adam_w.create ~lr varmap in
  let%bind test_labels =
    Tensor.to_dtype dataset.test_labels ~dtype:U32 >>= Tensor.to_device ~device
  in
  let%bind test_images =
    Tensor.to_dtype dataset.test_images ~dtype:F64 >>= Tensor.to_device ~device
  in
  let num_test_labels = Tensor.shape test_labels |> List.hd_exn in
  let n_batches = (Tensor.shape train_images |> List.hd_exn) / batch_size in
  let batch_idxs = Array.init n_batches ~f:Fn.id in
  List.range 1 (epochs + 1)
  |> List.fold_result ~init:() ~f:(fun () epoch ->
    print_endline [%string "Epoch %{epoch#Int}: start"];
    Array.permute batch_idxs;
    let%bind sum_loss =
      Array.fold_result batch_idxs ~init:0. ~f:(fun sum_loss batch_idx ->
        let%bind train_images =
          Tensor.narrow
            train_images
            ~dim:0
            ~start:(batch_idx * batch_size)
            ~len:batch_size
        in
        let%bind train_labels =
          Tensor.narrow
            train_labels
            ~dim:0
            ~start:(batch_idx * batch_size)
            ~len:batch_size
        in
        let%bind logits = ConvNet.forward model train_images ~train:true in
        let%bind log_sm = Ops.log_softmax logits ~dim:(-1) in
        let%bind loss = Loss.nll log_sm train_labels in
        let%bind () = Optim.Adam_w.backwards_step adamw ~loss in
        let%bind loss = Tensor.to_scalar loss in
        Gc.full_major ();
        return (sum_loss +. loss))
    in
    let avg_loss = sum_loss /. Int.to_float n_batches in
    let%bind test_logits = ConvNet.forward model test_images ~train:false in
    let%bind sum_ok =
      Tensor.argmax test_logits ~dim:(-1)
      >>= Tensor.( = ) test_labels
      >>= Tensor.to_dtype ~dtype:F64
      >>= Tensor.sum_all
      >>= Tensor.to_scalar
    in
    let test_accuracy = 100. *. sum_ok /. Int.to_float num_test_labels in
    print_endline
      [%string
        "Epoch %{epoch#Int}: train loss: %{avg_loss#Float} test acc: \
         %{test_accuracy#Float}"];
    return ())
;;

let mnist_training =
  Command.basic_or_error ~summary:"Train a simple model on MNIST"
  @@
  let%map_open.Command mnist_dataset_dir =
    flag
      "mnist-dataset-dir"
      (optional Filename_unix.arg_type)
      ~doc:"DIR directory containing MNIST dataset"
  and epochs =
    flag "epochs" (optional_with_default 200 int) ~doc:"INT number of epochs to train for"
  and device =
    flag
      "device"
      (optional_with_default
         `cuda
         (Arg_type.enumerated_sexpable
            (module struct
              type t =
                [ `cuda
                | `cpu
                ]
              [@@deriving enumerate, sexp]
            end)))
      ~doc:" device to run it on"
  and model =
    flag
      "model"
      (optional_with_default
         `convnet
         (Arg_type.enumerated_sexpable
            (module struct
              type t =
                [ `convnet
                | `mlp
                ]
              [@@deriving enumerate, sexp]
            end)))
      ~doc:" model to run"
  in
  fun () ->
    let open Or_error.Let_syntax in
    let%bind device =
      match device with
      | `cuda -> Device.cuda ~ordinal:0
      | `cpu -> Ok Device.cpu
    in
    let%bind dataset = Dataset.Mnist.load ~dir:mnist_dataset_dir in
    print_s
      [%message
        ""
          ~train_images:(dataset.train_images : Tensor.t)
          ~train_labels:(dataset.train_labels : Tensor.t)
          ~test_images:(dataset.test_images : Tensor.t)
          ~test_labels:(dataset.test_labels : Tensor.t)];
    match model with
    | `convnet -> convnet_training_loop dataset ~lr:0.001 ~epochs ~device
    | `mlp -> mlp_training_loop dataset ~lr:0.05 ~epochs ~device
;;

let () =
  Command_unix.run
  @@ Command.group
       ~summary:"Some examples showcasing candle-ocaml"
       [ "mnist-training", mnist_training ]
;;
