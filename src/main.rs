use anyhow::{Result, bail};
use ckpt::cast::{CastSpec, parse_dtype};
use ckpt::ckpt::{Checkpoint, Format, OpenOpts};
use ckpt::dtype::parse_size;
use ckpt::tp::{self, Rules, Source};
use ckpt::writer::{self, SelectOpts, ShardSpec};
use ckpt::{diff, inspect};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "ckpt",
    version,
    about = "Inspect, convert, reshard and diff ML checkpoints (safetensors, HF sharded, PyTorch DCP/FSDP, Megatron-LM, DeepSpeed ZeRO, torch.save) without Python or torch"
)]
struct Cli {
    /// Worker threads (diff); default min(4, cpus)
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// Check the CRC32 of every zip entry (DCP torch_save chunks, torch.save files) that is read;
    /// a mismatch is an error. `inspect --verify` checks every archive of the checkpoint.
    #[arg(long, global = true)]
    verify: bool,
    /// Megatron: present the checkpoint with HF names/layout for this architecture
    /// (auto|llama|qwen2). `convert` from Megatron uses `auto` unless --keep-names is given.
    #[arg(long, global = true)]
    hf_arch: Option<String>,
    /// Megatron: logical vocab size; unpads word embeddings / output layer (default: args.vocab_size
    /// for the HF mapping, no unpadding otherwise)
    #[arg(long, global = true)]
    vocab_size: Option<u64>,
    /// Show files as stored: Megatron torch_dist as plain DCP (stacked layers), a DeepSpeed
    /// *_states.pt file as a plain torch.save dict
    #[arg(long, global = true)]
    raw: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum To {
    /// one .safetensors file
    Safetensors,
    /// HF-style sharded safetensors + model.safetensors.index.json
    Hf,
    /// PyTorch Distributed Checkpoint directory (.metadata + __R_0.distcp)
    Dcp,
}

#[derive(clap::Args, Clone)]
struct SelArgs {
    /// Only tensors matching this glob (repeatable; `*` and `?`)
    #[arg(long)]
    include: Vec<String>,
    /// Skip tensors matching this glob (repeatable)
    #[arg(long)]
    exclude: Vec<String>,
    /// Remove this prefix from output names (e.g. `model.` for DCP {"model": ...} checkpoints)
    #[arg(long)]
    strip_prefix: Option<String>,
    /// Add this prefix to output names
    #[arg(long)]
    add_prefix: Option<String>,
}

impl SelArgs {
    fn opts(&self) -> SelectOpts {
        SelectOpts {
            include: self.include.clone(),
            exclude: self.exclude.clone(),
            strip_prefix: self.strip_prefix.clone(),
            add_prefix: self.add_prefix.clone(),
        }
    }
}

#[derive(clap::Args, Clone)]
struct CastArgs {
    /// Convert floating-point tensors to this dtype: fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, fp64
    /// (round-to-nearest-even, bit-compatible with torch's Tensor.to; ints/bools are kept)
    #[arg(long)]
    dtype: Option<String>,
    /// Keep the dtype of tensors matching this glob (repeatable), e.g. '*norm*'
    #[arg(long)]
    keep_dtype: Vec<String>,
}

impl CastArgs {
    fn spec(&self) -> Result<Option<CastSpec>> {
        Ok(match &self.dtype {
            None => {
                if !self.keep_dtype.is_empty() {
                    bail!("--keep-dtype needs --dtype");
                }
                None
            }
            Some(d) => Some(CastSpec {
                to: parse_dtype(d)?,
                exclude: self.keep_dtype.clone(),
            }),
        })
    }
}

#[derive(clap::Args, Clone)]
struct ShardArgs {
    /// Max bytes per output shard, e.g. 5GB, 500MB, 2GiB
    #[arg(long)]
    max_shard_size: Option<String>,
    /// Number of output shards (balanced by bytes)
    #[arg(long, conflicts_with = "max_shard_size")]
    num_shards: Option<usize>,
    /// Shard file prefix
    #[arg(long, default_value = "model")]
    prefix: String,
    /// Do not copy config/tokenizer files from an HF source directory
    #[arg(long)]
    no_aux: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// Detect format and list tensors, dtypes, shapes, params and per-shard sizes
    Inspect {
        path: PathBuf,
        /// Machine-readable JSON output
        #[arg(long)]
        json: bool,
        /// Only list tensors matching this glob (repeatable)
        #[arg(long)]
        filter: Vec<String>,
        /// Summary only, no per-tensor listing
        #[arg(long)]
        summary: bool,
        /// Show DCP chunk layout (offsets/sizes/file) per tensor
        #[arg(long)]
        chunks: bool,
    },
    /// Convert any supported checkpoint to one safetensors file, HF sharded safetensors or DCP
    Convert {
        src: PathBuf,
        /// Output: a .safetensors file or a directory
        #[arg(short, long)]
        out: PathBuf,
        /// Output format (default: by extension of --out)
        #[arg(long, value_enum)]
        to: Option<To>,
        #[command(flatten)]
        sel: SelArgs,
        #[command(flatten)]
        shard: ShardArgs,
        #[command(flatten)]
        cast: CastArgs,
        /// With --to dcp: number of rank files; tensors are row-split (dim 0) across them like FSDP
        #[arg(long, default_value_t = 1)]
        dcp_ranks: usize,
        /// Megatron sources: keep Megatron tensor names (TP/PP merged) instead of mapping to HF
        #[arg(long)]
        keep_names: bool,
    },
    /// Change HF shard count/size, or split/merge tensor-parallel layouts
    Reshard {
        /// Checkpoint, or a directory of tp_rank_XX/ subdirectories
        src: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
        #[command(flatten)]
        shard: ShardArgs,
        /// Split into N tensor-parallel ranks (needs --rules unless re-splitting a tp dir)
        #[arg(long)]
        tp: Option<usize>,
        /// YAML rules: which tensors split along which dim (see README / examples/)
        #[arg(long)]
        rules: Option<PathBuf>,
        /// HF config.json supplying head counts named in the rules (default: <src>/config.json)
        #[arg(long)]
        model_config: Option<PathBuf>,
        #[command(flatten)]
        sel: SelArgs,
        #[command(flatten)]
        cast: CastArgs,
    },
    /// Compare two checkpoints tensor by tensor (exit 0 = equal, 1 = different)
    Diff {
        a: PathBuf,
        b: PathBuf,
        /// Max abs difference still considered equal
        #[arg(long, default_value_t = 0.0)]
        tolerance: f64,
        #[arg(long)]
        json: bool,
        /// Show identical tensors too
        #[arg(long)]
        all: bool,
        #[arg(long)]
        strip_prefix_a: Option<String>,
        #[arg(long)]
        strip_prefix_b: Option<String>,
        /// Only compare names (after prefix stripping) matching this glob (repeatable)
        #[arg(long)]
        include: Vec<String>,
        /// Do not fail on names present in only one side
        #[arg(long)]
        ignore_missing: bool,
    },
}

fn shard_spec(s: &ShardArgs) -> Result<ShardSpec> {
    Ok(match (&s.max_shard_size, s.num_shards) {
        (_, Some(n)) => ShardSpec::Count(n),
        (Some(m), None) => ShardSpec::MaxSize(parse_size(m)?),
        (None, None) => ShardSpec::MaxSize(5_000_000_000),
    })
}

fn is_st_path(p: &Path) -> bool {
    p.extension().is_some_and(|e| e == "safetensors")
}

fn out_kind(to: Option<To>, out: &Path) -> To {
    to.unwrap_or(if is_st_path(out) {
        To::Safetensors
    } else {
        To::Hf
    })
}

fn is_megatron(f: Format) -> bool {
    matches!(f, Format::Megatron | Format::MegatronDist)
}

/// The model config for TP rules: --model-config, else config.json next to the source.
fn model_config(explicit: &Option<PathBuf>, src: &Path) -> Result<Option<serde_json::Value>> {
    let p = match explicit {
        Some(p) => Some(p.clone()),
        None => {
            let dir = if src.is_dir() {
                src.to_path_buf()
            } else {
                src.parent().unwrap_or(Path::new(".")).to_path_buf()
            };
            [dir.join("config.json")].into_iter().find(|p| p.is_file())
        }
    };
    match p {
        None => Ok(None),
        Some(p) => Ok(Some(serde_json::from_slice(
            &std::fs::read(&p).map_err(|e| anyhow::anyhow!("reading {}: {e}", p.display()))?,
        )?)),
    }
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();
    let threads = cli.threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(4)
    });
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();
    ckpt::zipread::set_verify(cli.verify);
    let oo = OpenOpts {
        megatron_hf: cli.hf_arch.clone(),
        vocab_size: cli.vocab_size,
        raw: cli.raw,
    };
    let t0 = Instant::now();
    match cli.cmd {
        Cmd::Inspect {
            path,
            json,
            filter,
            summary,
            chunks,
        } => {
            let ck = Checkpoint::open_with(&path, &oo)?;
            let o = inspect::InspectOpts {
                filter,
                tensors: !summary,
                chunks,
            };
            let verified = if cli.verify {
                Some(ck.verify_all()?)
            } else {
                None
            };
            if json {
                let mut j = inspect::to_json(&ck, &o)?;
                if let Some((a, e)) = verified {
                    j["verified"] = serde_json::json!({"archives": a, "entries": e});
                }
                println!("{}", serde_json::to_string_pretty(&j)?);
            } else {
                inspect::print_human(&ck, &o)?;
                if let Some((a, e)) = verified {
                    println!("verify:  CRC32 OK for {e} zip entries in {a} archive(s)");
                }
            }
        }
        Cmd::Convert {
            src,
            out,
            to,
            sel,
            shard,
            cast,
            dcp_ranks,
            keep_names,
        } => {
            let mut ck = Checkpoint::open_with(&src, &oo)?;
            if is_megatron(ck.format) && oo.megatron_hf.is_none() && !keep_names {
                let mut o2 = oo.clone();
                o2.megatron_hf = Some("auto".into());
                ck = Checkpoint::open_with(&src, &o2)?;
            }
            let mut s = writer::select(&ck, &sel.opts())?;
            writer::apply_cast(&ck, &mut s, &cast.spec()?);
            if ck.format.is_dcp() && !ck.bytes_items.is_empty() {
                eprintln!(
                    "note: skipping {} non-tensor DCP entries (see `ckpt inspect`)",
                    ck.bytes_items.len()
                );
            }
            match out_kind(to, &out) {
                To::Safetensors => {
                    writer::write_single(&ck, &s, &out)?;
                }
                To::Dcp => {
                    ckpt::dcp_write::write_dcp(&ck, &s, &out, dcp_ranks)?;
                }
                To::Hf => {
                    writer::write_sharded(&ck, &s, &out, shard_spec(&shard)?, &shard.prefix)?;
                    if !shard.no_aux && matches!(ck.format, Format::Safetensors | Format::HfSharded)
                    {
                        let c = writer::copy_aux_files(&ck.root, &out)?;
                        if !c.is_empty() {
                            eprintln!("copied {}", c.join(", "));
                        }
                    }
                }
            }
            if let Some(cfg) = &ck.hf_config {
                if out_kind(to, &out) == To::Hf {
                    let p = out.join("config.json");
                    let mut cfg = cfg.clone();
                    if let Some(d) = &cast.dtype {
                        cfg["torch_dtype"] = serde_json::json!(parse_dtype(d)?.torch_name());
                    }
                    std::fs::write(&p, serde_json::to_string_pretty(&cfg)? + "\n")?;
                    eprintln!("wrote {} ({})", p.display(), cfg["architectures"][0]);
                } else {
                    eprintln!(
                        "note: HF config.json is only written for directory (--to hf) outputs"
                    );
                }
            }
            eprintln!("done in {:.2}s", t0.elapsed().as_secs_f64());
        }
        Cmd::Reshard {
            src,
            out,
            shard,
            tp,
            rules,
            model_config: mc,
            sel,
            cast,
        } => {
            let cast = cast.spec()?;
            let is_tp_dir = src.is_dir() && !tp::rank_dirs(&src)?.is_empty();
            let plain = if is_tp_dir {
                None
            } else {
                Some(Checkpoint::open_with(&src, &oo)?)
            };
            // head counts for the rules: --model-config, <src>/config.json, or the HF config a
            // Megatron source produces with --hf-arch
            let cfg = match model_config(&mc, &src)? {
                Some(c) => Some(c),
                None => plain.as_ref().and_then(|c| c.hf_config.clone()),
            };
            let rules = rules
                .as_deref()
                .map(Rules::load)
                .transpose()?
                .map(|r| match &cfg {
                    Some(c) => r.with_model_config(c),
                    None => r,
                });
            let source = match plain {
                Some(ck) => Source::Plain(ck),
                None => tp::open_tp(&src, rules.as_ref())?,
            };
            if let Some(n) = tp {
                let r = match (&rules, &source) {
                    (Some(r), _) => r.clone(),
                    (None, Source::Tp { plan, .. }) => tp::rules_from_plan(plan),
                    (None, Source::Plain(_)) => bail!("--tp needs --rules"),
                };
                tp::split(&source, &r, n, &out, &cast)?;
            } else {
                match &source {
                    Source::Plain(ck) => {
                        let mut s = writer::select(ck, &sel.opts())?;
                        writer::apply_cast(ck, &mut s, &cast);
                        if is_st_path(&out) {
                            writer::write_single(ck, &s, &out)?;
                        } else {
                            writer::write_sharded(
                                ck,
                                &s,
                                &out,
                                shard_spec(&shard)?,
                                &shard.prefix,
                            )?;
                            if !shard.no_aux
                                && matches!(ck.format, Format::Safetensors | Format::HfSharded)
                            {
                                writer::copy_aux_files(&ck.root, &out)?;
                            }
                        }
                    }
                    Source::Tp { .. } => {
                        // merge TP ranks into full tensors
                        let entries = source.entries();
                        let decl: Vec<_> = entries
                            .iter()
                            .map(|e| ckpt::safetensors::OutTensor {
                                name: e.name.clone(),
                                dtype: CastSpec::target(&cast, &e.name, e.dtype),
                                shape: e.shape.clone(),
                            })
                            .collect();
                        let produce = |i: usize,
                                       w: &mut dyn FnMut(&[u8]) -> Result<()>|
                         -> Result<()> {
                            let full = source.read_full(i)?;
                            ckpt::cast::cast_stream(entries[i].dtype, decl[i].dtype, w, &mut |w2| {
                                w2(&full)
                            })
                        };
                        if is_st_path(&out) {
                            writer::write_single_with(&decl, &out, Default::default(), &produce)?;
                        } else {
                            writer::write_sharded_with(
                                &decl,
                                &out,
                                shard_spec(&shard)?,
                                &shard.prefix,
                                &produce,
                            )?;
                        }
                    }
                }
            }
            eprintln!("done in {:.2}s", t0.elapsed().as_secs_f64());
        }
        Cmd::Diff {
            a,
            b,
            tolerance,
            json,
            all,
            strip_prefix_a,
            strip_prefix_b,
            include,
            ignore_missing,
        } => {
            let ca = Checkpoint::open_with(&a, &oo)?;
            let cb = Checkpoint::open_with(&b, &oo)?;
            let o = diff::DiffOpts {
                tolerance,
                strip_prefix_a,
                strip_prefix_b,
                include,
                ignore_missing,
            };
            let r = diff::diff(&ca, &cb, &o)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&r)?);
            } else {
                diff::print_report(&r, all);
                eprintln!("diff took {:.2}s", t0.elapsed().as_secs_f64());
            }
            return Ok(if r.equal {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            });
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}
