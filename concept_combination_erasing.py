import itertools
import logging
import math
import sys
import json
import os
from pathlib import Path
from typing import Any
import torch.nn.functional as F
import torch
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import argparse
from logic_graph_iter_construct import generate_and_save_iterative_graphs, extract_concept_from_graph
from config import (PRETRAINED_MODEL_PATH, LOGICGRAPH_DIR, FINETUNE_MODEL_DIR, PREPARED_DATA_DIR)


logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


class COGFDDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self,
                 tokenizer,
                 size=512,
                 center_crop=False,
                 use_pooler=False,
                 task_info=None,
                 concept_combination=None,
                 labels=None,
                 args=None):
        self.use_pooler = use_pooler
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        if task_info is None or len(task_info) != 2:
            raise ValueError("task_info must be a list/tuple of length 2 containing [concept, theme]")
            
        if concept_combination is None or len(concept_combination) == 0:
            raise ValueError("concept_combination cannot be None or empty")
            
        if labels is None or len(labels) == 0:
            raise ValueError("labels cannot be None or empty")
            
        if len(concept_combination) != len(labels):
            raise ValueError(f"Length mismatch: concept_combination ({len(concept_combination)}) != labels ({len(labels)})")

        self.instance_images_path = []
        self.instance_prompt = []

        c, t = task_info
        p = Path(args.prepared_data_dir)
        if not p.exists():
            raise ValueError(f"Instance {p} images root doesn't exists.")

        image_paths = list(p.iterdir())
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {p}")
            
        self.instance_images_path += image_paths

        self.prompts = concept_combination
        self.labels = labels

        self.num_instance_images = len(self.instance_images_path)
        self._length = len(self.prompts)

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index >= self._length:
            raise IndexError(f"Index {index} out of range for dataset of length {self._length}")
            
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        concept = self.prompts[index % self._length]
        label = self.labels[index % self._length]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["concept"] = concept
        example["label"] = label
        example["instance_images"] = self.image_transforms(instance_image)

        example["prompt_ids"] = self.tokenizer(
            concept,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        example["attention_mask"] = self.tokenizer(
            concept,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).attention_mask

        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    source_prompts = [example["concept"] for example in examples]
    source_ids = [example["prompt_ids"] for example in examples]
    source_labels = [example["label"] for example in examples]
    source_mask = [example["attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    source_labels = torch.Tensor(source_labels).float()
    source_ids = torch.cat(source_ids, dim=0)
    source_mask = torch.cat(source_mask, dim=0)

    batch = {
        "source_prompts": source_prompts,
        "source_labels": source_labels,
        "source_ids": source_ids,
        "source_mask": source_mask,
        "pixel_values": pixel_values,
    }
    return batch


def main(task_info=["child drinking wine", "underage drinking"],
         concept_combination=[],
         labels=[],
         output_dir="",
         pretrained_model_name_or_path="",
         only_optimize_ca=False,
         use_pooler=True,
         train_batch_size=5,
         learning_rate=2.0e-06,
         max_train_steps=35,
         revision=None,
         tokenizer_name=None,
         with_prior_preservation=False,
         seed=None,
         resolution=512,
         center_crop=False,
         train_text_encoder=False,
         num_train_epochs=1,
         checkpointing_steps=500,
         resume_from_checkpoint=None,
         gradient_accumulation_steps=1,
         gradient_checkpointing=False,
         scale_lr=False,
         lr_scheduler="constant",
         lr_warmup_steps=0,
         lr_num_cycles=1,
         lr_power=1.0,
         use_8bit_adam=False,
         dataloader_num_workers=0,
         adam_beta1=0.9,
         adam_beta2=0.999,
         adam_weight_decay=0.01,
         adam_epsilon=1.0e-08,
         max_grad_norm=1.,
         allow_tf32=False,
         mixed_precision="fp16",
         enable_xformers_memory_efficient_attention=False,
         set_grads_to_none=False,
         args=None):

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    if train_text_encoder and gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError("Gradient accumulation is not supported when training the text encoder in distributed training. "
                         "Please set gradient_accumulation_steps to 1. This feature will be supported in the future.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    # Load the tokenizer
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision, use_fast=False)
    elif pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", revision=revision)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision)
    unet_1 = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision)


    class HiddenStatesController:  
        def __init__(self) -> None:
            self.encoder_attn_mask = []

        def set_encoder_attn_mask(self, attn_mask):
            self.encoder_attn_mask = attn_mask

        def zero_attn_probs(self):
            self.encoder_attn_mask = []

    class MyCrossAttnProcessor:

        def __init__(self, hiddenstates_controller: "HiddenStatesController", module_name) -> None:
            self.hiddenstates_controller = hiddenstates_controller
            self.module_name = module_name

        def __call__(self, attn: "CrossAttention", hidden_states, encoder_hidden_states=None, attention_mask=None):

            encoder_attention_mask = self.hiddenstates_controller.encoder_attn_mask

            batch_size, sequence_length, _ = hidden_states.shape


            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length,
                                                         batch_size=batch_size)  # * 这里对应的应该是图像mask？

            if encoder_attention_mask is not None and encoder_hidden_states is not None:  # * 加入text的attention mask操作
                # B x 77 -> B x 4096 x 77
                attention_mask = encoder_attention_mask.unsqueeze(1).repeat(1, hidden_states.size(1), 1)
                attention_mask = attention_mask.repeat_interleave(attn.heads, dim=0).type_as(hidden_states)

            query = attn.to_q(hidden_states)
            query = attn.head_to_batch_dim(query)
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            return hidden_states

    attn_controller = HiddenStatesController()
    module_count = 0
    for n, m in unet.named_modules():
        if n.endswith('attn2'):
            m.set_processor(MyCrossAttnProcessor(attn_controller,
                                                 n))
            module_count += 1
    print(f"cross attention module count: {module_count}")
    ###

    attn_controller_1 = HiddenStatesController()
    module_count = 0
    for n, m in unet_1.named_modules():
        if n.endswith('attn2'):
            m.set_processor(MyCrossAttnProcessor(attn_controller_1,
                                                 n))
            module_count += 1
    print(f"cross attention module count: {module_count}")

    vae.requires_grad_(False)
    if not train_text_encoder:
        text_encoder.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32.")

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}")

    if train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
                         f" {low_precision_error_string}")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    if only_optimize_ca:  
        params_to_optimize = (itertools.chain(unet.parameters(), text_encoder.parameters()) if train_text_encoder else
                              [p for n, p in unet.named_parameters() if 'attn2' in n and 'to_v' not in n])
        print("only optimize cross attention...")
    else:
        params_to_optimize = (itertools.chain(unet.parameters(), text_encoder.parameters())
                              if train_text_encoder else unet.parameters())
        print("optimize unet...")

    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = COGFDDataset(tokenizer=tokenizer,
                                       size=resolution,
                                       center_crop=center_crop,
                                       use_pooler=use_pooler,
                                       task_info=task_info,
                                       concept_combination=concept_combination,
                                       labels=labels,
                                       args=args)

    if len(train_dataset) == 0:
        raise ValueError("Dataset is empty. Please check your dataset configuration.")

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_batch_size,
                                                   shuffle=True,
                                                   collate_fn=lambda examples: collate_fn(examples, with_prior_preservation),
                                                   num_workers=dataloader_num_workers,
                                                   drop_last=True)

    if len(train_dataloader) == 0:
        raise ValueError("No batches in the dataloader. Please check your batch_size.")

    # Ensure gradient_accumulation_steps is valid
    if gradient_accumulation_steps <= 0:
        logger.warning("gradient_accumulation_steps <= 0, setting to 1")
        gradient_accumulation_steps = 1

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Ensure we have at least one training step
    if num_update_steps_per_epoch == 0:
        logger.warning("No update steps per epoch, setting to 1")
        num_update_steps_per_epoch = 1
        
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    # Prepare everything with our `accelerator`.
    if train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler, unet_1 = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler, unet_1)
    else:
        unet, optimizer, train_dataloader, lr_scheduler, unet_1 = accelerator.prepare(unet, optimizer, train_dataloader,
                                                                                      lr_scheduler, unet_1)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("forgetmenot")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    debug_twice = 2
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        unet_1.to(torch.device('cuda:1'))
        if train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # show
                if debug_twice > 0:
                    print(batch["source_prompts"])
                    debug_twice -= 1
                # Convert images to latent space

                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(args.start, args.end, (bsz, ), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states_source = text_encoder(batch["source_ids"], attention_mask=batch["source_mask"])[0]

                # set concept_positions for this batch
                attn_controller.set_encoder_attn_mask(batch["source_mask"])
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states_source,
                ).sample

                # Predict the noise residual
                with torch.no_grad():
                    attn_controller_1.set_encoder_attn_mask(batch["source_mask"])
                    # Move all inputs to cuda:1
                    noisy_latents_1 = noisy_latents.to(torch.device('cuda:1'))
                    timesteps_1 = timesteps.to(torch.device('cuda:1'))
                    encoder_hidden_states_1 = encoder_hidden_states_source.to(torch.device('cuda:1'))
                    
                    model_pred_1 = unet_1(noisy_latents_1, timesteps_1, encoder_hidden_states_1).sample
                    # Move result back to cuda:0 before computing loss
                    model_pred_1 = model_pred_1.to(torch.device('cuda:0'))

                unlearn_select = batch["source_labels"] == args.p1
                retain_select = batch["source_labels"] == args.p2

                # Ensure all tensors are on the same device for loss computation
                loss_1 = F.mse_loss(model_pred[unlearn_select], model_pred_1[unlearn_select])
                loss_2 = F.mse_loss(model_pred[retain_select], model_pred_1[retain_select])

                # Compute final loss on the same device
                final_loss = 0.1 * torch.exp(-loss_1) + torch.exp(loss_2)
                accelerator.backward(final_loss)

                if accelerator.sync_gradients:
                    params_to_clip = params_to_optimize
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=set_grads_to_none)
                attn_controller.zero_attn_probs()
                attn_controller_1.zero_attn_probs()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss_1": loss_1.detach().item(),
                "loss_2": loss_2.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            revision=revision,
        )
        pipeline.save_pretrained(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default=PRETRAINED_MODEL_PATH)
    parser.add_argument('--theme', type=str)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--only_xa', action='store_true')
    parser.add_argument('--p1', type=float, default=-1)
    parser.add_argument('--p2', type=float, default=1)
    parser.add_argument('--combine_concept_x', type=str, default="A child is drinking wine")
    parser.add_argument('--combine_theme_y', type=str, default="underage drinking")
    parser.add_argument('--train_batch_size', type=int, default=20)
    parser.add_argument('--start', type=int, default=990)
    parser.add_argument('--end', type=int, default=1000)
    parser.add_argument('--iterate_n', type=int, default=1)
    parser.add_argument('--model_exist_check', action='store_true')

    args = parser.parse_args()
    combine_concept = args.combine_concept_x
    args.prepared_data_dir = PREPARED_DATA_DIR.format(concept_combination=combine_concept)
    args.model_output_dir = FINETUNE_MODEL_DIR.format(concept_combination=combine_concept)
    args.graph_output_dir = LOGICGRAPH_DIR.format(concept_combination=combine_concept)

    print('33333333333333333333333333')

    print(args.graph_output_dir)
    print(args.model_output_dir)
    print(args.prepared_data_dir)


    combine_theme = args.combine_theme_y
    task_info = [combine_concept, combine_theme]

    graph_path = os.path.join(args.graph_output_dir, f"{combine_concept}.json")
    # generate concept logic graph
    try:
        with open(graph_path, 'r') as f:
            parsed_graph = json.load(f)
    except FileNotFoundError:
        print(f"File {graph_path} not found. Generating concept logic graph...")
        combine_concept_x = args.combine_concept_x.replace("_", " ")
        combine_theme_y = args.combine_theme_y.replace("_", " ")
        parsed_graph = generate_and_save_iterative_graphs(combine_concept_x, combine_theme_y, graph_path, iterate_n=args.iterate_n)


    # extract concepts from graph
    concept_combination, sub_concept = extract_concept_from_graph(parsed_graph)

    concepts = concept_combination + sub_concept
    labels = [args.p1 for i in concept_combination] + [args.p2 for i in sub_concept]
    print(concepts)
    print(labels)
    batch_size = min(len(concepts), args.train_batch_size)
    
    
    
    if args.model_exist_check and os.path.isdir(args.model_output_dir):
            print('The model has been finetuned')
    else:
        main(task_info=task_info,
            concept_combination=concepts,
            labels=labels,
            output_dir=args.model_output_dir,
            pretrained_model_name_or_path=args.pretrained_path,
            learning_rate=args.lr,
            max_train_steps=args.max_steps,
            only_optimize_ca=args.only_xa,
            seed=42,
            train_batch_size=batch_size,
            args=args)

