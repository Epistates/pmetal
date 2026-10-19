//! `pmetal dataset`: looking at, converting and preparing training data.
//!
//! Every command reads rows through [`pmetal_data::rows`], so a dataset is a
//! JSONL, JSON, Parquet, Arrow or CSV file (or a directory of them) whichever
//! command reads it, and what a row is (a conversation, an instruction, a
//! document) is [`pmetal_data::DatasetFormat`]'s to say, the reading training
//! does. Text is formatted with the model's own chat template, and lengths
//! are counted with its tokenizer.

mod common;
mod inspect;
mod template;
mod transform;

use crate::cli::dataset::DatasetAction;

/// Run dataset subcommands.
pub(crate) async fn run_dataset_command(action: DatasetAction) -> anyhow::Result<()> {
    match action {
        DatasetAction::Analyze {
            path,
            model,
            detailed,
        } => inspect::analyze(path, model, detailed).await,
        DatasetAction::Download {
            dataset_id,
            split,
            output,
            revision,
        } => inspect::download(dataset_id, split, output, revision).await,
        DatasetAction::Convert {
            input,
            output,
            columns,
            shuffle,
            seed,
        } => transform::convert(input, output, columns, shuffle, seed),
        DatasetAction::Validate {
            path,
            model,
            max_seq_len,
        } => inspect::validate(path, model, max_seq_len).await,
        DatasetAction::Preview {
            dataset_id,
            split,
            num,
        } => inspect::preview(dataset_id, split, num).await,
        DatasetAction::Filter {
            input,
            output,
            model,
            min_tokens,
            max_tokens,
            dedup,
            pattern,
            invert,
            complete_only,
        } => {
            transform::filter(
                input,
                output,
                model,
                min_tokens,
                max_tokens,
                dedup,
                pattern,
                invert,
                complete_only,
            )
            .await
        }
        DatasetAction::Split {
            input,
            output_dir,
            val_ratio,
            test_ratio,
            seed,
            stratify,
        } => transform::split(input, output_dir, val_ratio, test_ratio, seed, stratify),
        DatasetAction::Merge {
            inputs,
            output,
            shuffle,
            seed,
            interleave,
            weights,
        } => transform::merge(inputs, output, shuffle, seed, interleave, weights),
        DatasetAction::Sample {
            input,
            output,
            num,
            seed,
        } => transform::sample(input, output, num, seed),
        DatasetAction::Template {
            input,
            output,
            template,
            system,
            model,
            add_generation_prompt,
            mask_prompt,
        } => {
            template::template(
                input,
                output,
                template,
                system,
                model,
                add_generation_prompt,
                mask_prompt,
            )
            .await
        }
        DatasetAction::Prepare {
            dataset,
            output_dir,
            model,
            template,
            max_seq_len,
            val_ratio,
            seed,
            no_dedup,
            columns,
        } => {
            template::prepare(
                dataset,
                output_dir,
                model,
                template,
                max_seq_len,
                val_ratio,
                seed,
                no_dedup,
                columns,
            )
            .await
        }
        DatasetAction::Formats => {
            template::formats();
            Ok(())
        }
    }
}
