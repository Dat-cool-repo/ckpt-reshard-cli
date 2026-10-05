//! A restricted, inert pickle reader.
//!
//! Safety model:
//! * Nothing is ever executed. Python pickles are programs for a small stack machine;
//!   `GLOBAL`/`REDUCE`/`BUILD`/`NEWOBJ` normally *import and call* arbitrary Python code.
//!   This VM never imports or calls anything: a global is recorded as an inert
//!   `(module, name)` pair, and calling it just records an `Object { class, args }` node.
//! * Defense in depth: every global referenced by the pickle must be on an explicit
//!   allowlist (the handful of classes DCP metadata and `torch.save` tensor records use).
//!   Anything else (`os.system`, `builtins.eval`, `subprocess.Popen`, ...) makes the
//!   load fail with an error, so a tampered checkpoint is reported instead of half-read.
//! * Extension registry opcodes (EXT1/2/4) and out-of-band buffers are refused.
//! * All containers live in a flat arena addressed by index, so self-referencing or
//!   deeply nested pickles cannot cause recursive drops; all consumers that walk the
//!   graph use explicit depth limits.

use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    Bytes(Rc<[u8]>),
    Ref(usize),
}

#[derive(Debug)]
pub enum Node {
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Set(Vec<Value>),
    Global(Rc<str>, Rc<str>),
    Object {
        class: Value,
        args: Vec<Value>,
        state: Option<Value>,
        items: Vec<Value>,
        dict_items: Vec<(Value, Value)>,
    },
    Persistent(Value),
}

/// The decoded pickle: an arena of nodes plus the root value.
#[derive(Debug)]
pub struct Pickle {
    pub nodes: Vec<Node>,
    pub root: Value,
}

/// Which globals a pickle may reference. Each format gets the smallest list that its real
/// writers need; everything else is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Allow {
    /// DCP `.metadata` + `torch.save` tensor records + simple python containers.
    Checkpoint,
    /// Megatron-LM `model_optim_rng.pt` / `common.pt`: adds the `args` Namespace (with its
    /// enum and signal values) and the numpy RNG state.
    Megatron,
    /// DeepSpeed `*_model_states.pt` / `*_optim_states.pt`: adds DeepSpeed's loss-scaler,
    /// ZeRO stage enum and fragment-address records.
    DeepSpeed,
}

fn global_allowed(allow: Allow, module: &str, name: &str) -> bool {
    if base_allowed(module, name) {
        return true;
    }
    match allow {
        Allow::Checkpoint => false,
        Allow::Megatron => match module {
            "argparse" => name == "Namespace",
            // enum members stored in args (e.g. args.attention_backend, args.exit_signal)
            "megatron.core.transformer.enums" => matches!(
                name,
                "AttnBackend"
                    | "AttnMaskType"
                    | "AttnType"
                    | "LayerType"
                    | "ModelType"
                    | "CudaGraphScope"
            ),
            "megatron.core.enums" => matches!(name, "ModelType" | "Fp8Recipe"),
            "signal" => name == "Signals",
            "numpy._core.multiarray" | "numpy.core.multiarray" => {
                matches!(name, "_reconstruct" | "scalar")
            }
            "numpy" => matches!(name, "ndarray" | "dtype"),
            _ => false,
        },
        Allow::DeepSpeed => match module {
            "deepspeed.runtime.fp16.loss_scaler" => {
                matches!(name, "LossScaler" | "DynamicLossScaler")
            }
            "deepspeed.runtime.zero.config" => name == "ZeroStageEnum",
            "deepspeed.utils.tensor_fragment" => name == "fragment_address",
            "argparse" => name == "Namespace",
            _ => false,
        },
    }
}

fn base_allowed(module: &str, name: &str) -> bool {
    match module {
        "torch" => {
            name == "Size"
                || name == "Tensor"
                || name == "device"
                || name == "strided"
                || crate::dtype::DType::from_torch_name(name).is_some()
                || crate::dtype::DType::from_storage_name(name).is_some()
                || matches!(
                    name,
                    "contiguous_format" | "channels_last" | "preserve_format"
                )
        }
        "torch._utils" => matches!(
            name,
            "_rebuild_tensor_v2" | "_rebuild_tensor" | "_rebuild_parameter"
        ),
        // tensor subclass wrapper: _rebuild_from_type_v2(_rebuild_tensor_v2, torch.Tensor, args, state)
        "torch._tensor" => name == "_rebuild_from_type_v2",
        "torch.nn.parameter" => name == "Parameter",
        "torch.serialization" => matches!(name, "_get_layout"),
        "torch.distributed.checkpoint.metadata" => matches!(
            name,
            "Metadata"
                | "TensorStorageMetadata"
                | "BytesStorageMetadata"
                | "ChunkStorageMetadata"
                | "TensorProperties"
                | "MetadataIndex"
                | "StorageMeta"
                | "_MEM_FORMAT_ENCODING"
        ),
        "torch.distributed.checkpoint.filesystem" => name == "_StorageInfo",
        // planner_data of Megatron torch_dist checkpoints caches the per-rank save plans
        "torch.distributed.checkpoint.planner" => matches!(
            name,
            "WriteItemType" | "LoadItemType" | "SavePlan" | "WriteItem" | "TensorWriteData"
        ),
        "torch.distributed._shard.metadata" => name == "ShardMetadata",
        "torch.distributed._shard.sharded_tensor.metadata" => {
            matches!(
                name,
                "TensorProperties" | "ShardedTensorMetadata" | "MEM_FORMAT_ENCODING"
            )
        }
        "collections" => name == "OrderedDict",
        "builtins" | "__builtin__" => matches!(name, "set" | "frozenset" | "bytearray" | "slice"),
        "_codecs" => name == "encode",
        "copyreg" | "copy_reg" => name == "_reconstructor",
        "pathlib" => matches!(
            name,
            "PosixPath" | "WindowsPath" | "PurePosixPath" | "PureWindowsPath" | "Path"
        ),
        _ => false,
    }
}

struct Vm<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: HashMap<u32, Value>,
    nodes: Vec<Node>,
    allow: Allow,
}

const MAX_NODES: usize = 50_000_000;

pub fn load(data: &[u8], allow: Allow) -> Result<Pickle> {
    let mut vm = Vm {
        data,
        pos: 0,
        stack: Vec::new(),
        marks: Vec::new(),
        memo: HashMap::new(),
        nodes: Vec::new(),
        allow,
    };
    let root = vm
        .run()
        .with_context(|| format!("restricted pickle reader failed at byte {}", vm.pos))?;
    Ok(Pickle {
        nodes: vm.nodes,
        root,
    })
}

impl<'a> Vm<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            bail!("truncated pickle");
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn line(&mut self) -> Result<&'a str> {
        let rest = &self.data[self.pos..];
        let n = rest
            .iter()
            .position(|&c| c == b'\n')
            .ok_or_else(|| anyhow!("unterminated line"))?;
        let s = std::str::from_utf8(&rest[..n])?;
        self.pos += n + 1;
        Ok(s)
    }
    fn len_checked(&self, n: u64) -> Result<usize> {
        if n > (self.data.len() - self.pos) as u64 {
            bail!("length {n} exceeds remaining pickle data");
        }
        Ok(n as usize)
    }
    fn push_node(&mut self, n: Node) -> Result<Value> {
        if self.nodes.len() >= MAX_NODES {
            bail!("pickle too large (node limit)");
        }
        self.nodes.push(n);
        Ok(Value::Ref(self.nodes.len() - 1))
    }
    fn pop(&mut self) -> Result<Value> {
        let floor = self.marks.last().copied().unwrap_or(0);
        if self.stack.len() <= floor {
            bail!("stack underflow");
        }
        Ok(self.stack.pop().unwrap())
    }
    fn top(&self) -> Result<&Value> {
        self.stack.last().ok_or_else(|| anyhow!("stack underflow"))
    }
    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let m = self.marks.pop().ok_or_else(|| anyhow!("MARK missing"))?;
        Ok(self.stack.split_off(m))
    }
    fn global(&mut self, module: &str, name: &str) -> Result<Value> {
        if !global_allowed(self.allow, module, name) {
            bail!(
                "refusing pickle global `{module}.{name}`: not on the checkpoint allowlist \
                 (this file may be malicious or from an unsupported writer)"
            );
        }
        self.push_node(Node::Global(module.into(), name.into()))
    }
    fn global_of(&self, v: &Value) -> Option<(Rc<str>, Rc<str>)> {
        if let Value::Ref(i) = v
            && let Node::Global(m, n) = &self.nodes[*i]
        {
            return Some((m.clone(), n.clone()));
        }
        None
    }
    fn seq_items(&self, v: &Value) -> Result<Vec<Value>> {
        match v {
            Value::Ref(i) => match &self.nodes[*i] {
                Node::Tuple(x) | Node::List(x) | Node::Set(x) => Ok(x.clone()),
                _ => bail!("expected tuple/list"),
            },
            _ => bail!("expected tuple/list"),
        }
    }
    fn call(&mut self, func: Value, args: Vec<Value>) -> Result<Value> {
        let Some((m, n)) = self.global_of(&func) else {
            bail!("REDUCE on a non-global callable is not supported");
        };
        match (&*m, &*n) {
            ("collections", "OrderedDict") => {
                let mut items = Vec::new();
                if let Some(a) = args.first() {
                    for it in self.seq_items(a)? {
                        let kv = self.seq_items(&it)?;
                        if kv.len() == 2 {
                            items.push((kv[0].clone(), kv[1].clone()));
                        }
                    }
                }
                self.push_node(Node::Dict(items))
            }
            ("builtins" | "__builtin__", "set" | "frozenset") => {
                let items = match args.first() {
                    Some(a) => self.seq_items(a)?,
                    None => vec![],
                };
                self.push_node(Node::Set(items))
            }
            ("_codecs", "encode") => match args.first() {
                Some(Value::Str(s)) => {
                    // protocol-2 bytes: _codecs.encode(<latin1 str>, 'latin1')
                    let b: Vec<u8> = s.chars().map(|c| c as u32 as u8).collect();
                    Ok(Value::Bytes(b.into()))
                }
                _ => bail!("unexpected _codecs.encode args"),
            },
            ("builtins" | "__builtin__", "bytearray") => match args.first() {
                Some(Value::Bytes(b)) => Ok(Value::Bytes(b.clone())),
                None => Ok(Value::Bytes(Rc::from(Vec::new()))),
                _ => bail!("unexpected bytearray args"),
            },
            ("copyreg" | "copy_reg", "_reconstructor") => {
                let cls = args
                    .first()
                    .cloned()
                    .ok_or_else(|| anyhow!("bad _reconstructor"))?;
                self.push_node(Node::Object {
                    class: cls,
                    args: vec![],
                    state: None,
                    items: vec![],
                    dict_items: vec![],
                })
            }
            _ => self.push_node(Node::Object {
                class: func,
                args,
                state: None,
                items: vec![],
                dict_items: vec![],
            }),
        }
    }

    fn append(&mut self, target: &Value, vals: Vec<Value>) -> Result<()> {
        let Value::Ref(i) = target else {
            bail!("APPEND to non-container")
        };
        match &mut self.nodes[*i] {
            Node::List(v) | Node::Set(v) => v.extend(vals),
            Node::Object { items, .. } => items.extend(vals),
            _ => bail!("APPEND to non-list"),
        }
        Ok(())
    }
    fn setitems(&mut self, target: &Value, kvs: Vec<Value>) -> Result<()> {
        if !kvs.len().is_multiple_of(2) {
            bail!("odd number of SETITEMS values");
        }
        let Value::Ref(i) = target else {
            bail!("SETITEM on non-dict")
        };
        let mut it = kvs.into_iter();
        let mut pairs = Vec::new();
        while let (Some(k), Some(v)) = (it.next(), it.next()) {
            pairs.push((k, v));
        }
        match &mut self.nodes[*i] {
            Node::Dict(d) => d.extend(pairs),
            Node::Object { dict_items, .. } => dict_items.extend(pairs),
            _ => bail!("SETITEM on non-dict"),
        }
        Ok(())
    }

    fn run(&mut self) -> Result<Value> {
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    let p = self.u8()?;
                    if p > 5 {
                        bail!("unsupported pickle protocol {p}");
                    }
                }
                0x95 => {
                    self.u64()?;
                } // FRAME
                b'.' => {
                    return self
                        .stack
                        .pop()
                        .ok_or_else(|| anyhow!("empty stack at STOP"));
                }
                b'(' => self.marks.push(self.stack.len()),
                b'0' => {
                    self.pop()?;
                }
                b'1' => {
                    self.pop_mark()?;
                }
                b'2' => {
                    let t = self.top()?.clone();
                    self.stack.push(t)
                }
                b'N' => self.stack.push(Value::None),
                0x88 => self.stack.push(Value::Bool(true)),
                0x89 => self.stack.push(Value::Bool(false)),
                b'J' => {
                    let v = self.u32()? as i32 as i64;
                    self.stack.push(Value::Int(v))
                }
                b'K' => {
                    let v = self.u8()? as i64;
                    self.stack.push(Value::Int(v))
                }
                b'M' => {
                    let v = self.u16()? as i64;
                    self.stack.push(Value::Int(v))
                }
                0x8a | 0x8b => {
                    let n = if op == 0x8a {
                        self.u8()? as u64
                    } else {
                        self.u32()? as u64
                    };
                    let n = self.len_checked(n)?;
                    let b = self.take(n)?;
                    if n > 8 {
                        bail!("integer too large ({n} bytes)");
                    }
                    let mut buf = if n > 0 && b[n - 1] & 0x80 != 0 {
                        [0xffu8; 8]
                    } else {
                        [0u8; 8]
                    };
                    buf[..n].copy_from_slice(b);
                    self.stack.push(Value::Int(i64::from_le_bytes(buf)))
                }
                b'I' => {
                    let l = self.line()?;
                    let v = match l {
                        "00" => Value::Bool(false),
                        "01" => Value::Bool(true),
                        _ => Value::Int(l.trim().parse()?),
                    };
                    self.stack.push(v)
                }
                b'L' => {
                    let l = self.line()?.trim_end_matches('L');
                    self.stack.push(Value::Int(l.parse()?))
                }
                b'F' => {
                    let l = self.line()?;
                    self.stack.push(Value::Float(l.trim().parse()?))
                }
                b'G' => {
                    let v = f64::from_be_bytes(self.take(8)?.try_into().unwrap());
                    self.stack.push(Value::Float(v))
                }
                0x8c | b'X' | 0x8d => {
                    let n = match op {
                        0x8c => self.u8()? as u64,
                        b'X' => self.u32()? as u64,
                        _ => self.u64()?,
                    };
                    let n = self.len_checked(n)?;
                    let s = std::str::from_utf8(self.take(n)?)?;
                    self.stack.push(Value::Str(s.into()))
                }
                b'V' => {
                    let s = self.line()?;
                    self.stack.push(Value::Str(s.into()))
                }
                b'S' => {
                    let s = self.line()?.trim();
                    let s = s.trim_matches(|c| c == '\'' || c == '"');
                    self.stack.push(Value::Str(s.into()))
                }
                b'T' | b'U' => {
                    let n = if op == b'U' {
                        self.u8()? as u64
                    } else {
                        self.u32()? as u64
                    };
                    let n = self.len_checked(n)?;
                    let b = self.take(n)?;
                    // py2 str: decode as latin1
                    let s: String = b.iter().map(|&c| c as char).collect();
                    self.stack.push(Value::Str(s.into()))
                }
                b'C' | b'B' | 0x8e | 0x96 => {
                    let n = match op {
                        b'C' => self.u8()? as u64,
                        b'B' => self.u32()? as u64,
                        _ => self.u64()?,
                    };
                    let n = self.len_checked(n)?;
                    let b = self.take(n)?;
                    self.stack.push(Value::Bytes(b.into()))
                }
                b')' => {
                    let v = self.push_node(Node::Tuple(vec![]))?;
                    self.stack.push(v)
                }
                0x85..=0x87 => {
                    let k = (op - 0x84) as usize;
                    let mut items = Vec::with_capacity(k);
                    for _ in 0..k {
                        items.push(self.pop()?);
                    }
                    items.reverse();
                    let v = self.push_node(Node::Tuple(items))?;
                    self.stack.push(v)
                }
                b't' => {
                    let items = self.pop_mark()?;
                    let v = self.push_node(Node::Tuple(items))?;
                    self.stack.push(v)
                }
                b']' => {
                    let v = self.push_node(Node::List(vec![]))?;
                    self.stack.push(v)
                }
                b'l' => {
                    let items = self.pop_mark()?;
                    let v = self.push_node(Node::List(items))?;
                    self.stack.push(v)
                }
                b'}' => {
                    let v = self.push_node(Node::Dict(vec![]))?;
                    self.stack.push(v)
                }
                b'd' => {
                    let items = self.pop_mark()?;
                    let v = self.push_node(Node::Dict(vec![]))?;
                    self.setitems(&v, items)?;
                    self.stack.push(v)
                }
                0x8f => {
                    let v = self.push_node(Node::Set(vec![]))?;
                    self.stack.push(v)
                }
                0x91 => {
                    let items = self.pop_mark()?;
                    let v = self.push_node(Node::Set(items))?;
                    self.stack.push(v)
                }
                b'a' => {
                    let v = self.pop()?;
                    let t = self.top()?.clone();
                    self.append(&t, vec![v])?;
                }
                b'e' | 0x90 => {
                    let items = self.pop_mark()?;
                    let t = self.top()?.clone();
                    self.append(&t, items)?;
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    let t = self.top()?.clone();
                    self.setitems(&t, vec![k, v])?;
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    let t = self.top()?.clone();
                    self.setitems(&t, items)?;
                }
                0x94 => {
                    let t = self.top()?.clone();
                    let idx = self.memo.len() as u32;
                    self.memo.insert(idx, t);
                }
                b'q' | b'r' | b'p' => {
                    let idx = match op {
                        b'q' => self.u8()? as u32,
                        b'r' => self.u32()?,
                        _ => self.line()?.trim().parse()?,
                    };
                    let t = self.top()?.clone();
                    self.memo.insert(idx, t);
                }
                b'h' | b'j' | b'g' => {
                    let idx = match op {
                        b'h' => self.u8()? as u32,
                        b'j' => self.u32()?,
                        _ => self.line()?.trim().parse()?,
                    };
                    let v = self
                        .memo
                        .get(&idx)
                        .cloned()
                        .ok_or_else(|| anyhow!("memo key {idx} missing"))?;
                    self.stack.push(v)
                }
                b'c' => {
                    let m = self.line()?.to_string();
                    let n = self.line()?.to_string();
                    let v = self.global(&m, &n)?;
                    self.stack.push(v)
                }
                0x93 => {
                    let n = self.pop()?;
                    let m = self.pop()?;
                    let (Value::Str(m), Value::Str(n)) = (m, n) else {
                        bail!("STACK_GLOBAL needs strings")
                    };
                    let v = self.global(&m, &n)?;
                    self.stack.push(v)
                }
                b'R' => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    let args = self.seq_items(&args)?;
                    let v = self.call(func, args)?;
                    self.stack.push(v)
                }
                0x81 => {
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    let args = self.seq_items(&args)?;
                    let v = self.call(cls, args)?;
                    self.stack.push(v)
                }
                0x92 => {
                    let _kwargs = self.pop()?;
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    let args = self.seq_items(&args)?;
                    let v = self.call(cls, args)?;
                    self.stack.push(v)
                }
                b'i' => {
                    let m = self.line()?.to_string();
                    let n = self.line()?.to_string();
                    let args = self.pop_mark()?;
                    let cls = self.global(&m, &n)?;
                    let v = self.call(cls, args)?;
                    self.stack.push(v)
                }
                b'o' => {
                    let mut items = self.pop_mark()?;
                    if items.is_empty() {
                        bail!("OBJ without class");
                    }
                    let cls = items.remove(0);
                    let v = self.call(cls, items)?;
                    self.stack.push(v)
                }
                b'b' => {
                    let state = self.pop()?;
                    let t = self.top()?.clone();
                    let Value::Ref(i) = t else {
                        bail!("BUILD on non-object")
                    };
                    match &mut self.nodes[i] {
                        Node::Object { state: s, .. } => *s = Some(state),
                        Node::Dict(_) | Node::List(_) => {} // e.g. OrderedDict with __dict__; ignore
                        _ => bail!("BUILD on unsupported node"),
                    }
                }
                b'Q' => {
                    let pid = self.pop()?;
                    let v = self.push_node(Node::Persistent(pid))?;
                    self.stack.push(v)
                }
                b'P' => {
                    let s = self.line()?;
                    let v = self.push_node(Node::Persistent(Value::Str(s.into())))?;
                    self.stack.push(v)
                }
                0x82..=0x84 => bail!("pickle extension registry opcodes are not allowed"),
                0x97 | 0x98 => bail!("out-of-band pickle buffers are not allowed"),
                _ => bail!("unknown/unsupported pickle opcode 0x{op:02x}"),
            }
        }
    }
}

// ---------------------------------------------------------------- navigation helpers

impl Pickle {
    pub fn node(&self, v: &Value) -> Option<&Node> {
        match v {
            Value::Ref(i) => self.nodes.get(*i),
            _ => None,
        }
    }
    pub fn as_int(&self, v: &Value) -> Option<i64> {
        match v {
            Value::Int(i) => Some(*i),
            Value::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }
    pub fn as_str<'a>(&self, v: &'a Value) -> Option<&'a str> {
        match v {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn seq<'a>(&'a self, v: &Value) -> Option<&'a [Value]> {
        match self.node(v)? {
            Node::Tuple(x) | Node::List(x) | Node::Set(x) => Some(x),
            _ => None,
        }
    }
    pub fn dict<'a>(&'a self, v: &Value) -> Option<&'a [(Value, Value)]> {
        match self.node(v)? {
            Node::Dict(d) => Some(d),
            Node::Object { dict_items, .. } => Some(dict_items),
            _ => None,
        }
    }
    pub fn dict_get<'a>(&'a self, v: &Value, key: &str) -> Option<&'a Value> {
        self.dict(v)?
            .iter()
            .find(|(k, _)| matches!(k, Value::Str(s) if &**s == key))
            .map(|(_, v)| v)
    }
    /// (module, name) of a Global node, or of an Object's class.
    pub fn class_of(&self, v: &Value) -> Option<(&str, &str)> {
        match self.node(v)? {
            Node::Global(m, n) => Some((m, n)),
            Node::Object { class, .. } => match self.node(class)? {
                Node::Global(m, n) => Some((m, n)),
                _ => None,
            },
            _ => None,
        }
    }
    pub fn obj_args<'a>(&'a self, v: &Value) -> Option<&'a [Value]> {
        match self.node(v)? {
            Node::Object { args, .. } => Some(args),
            _ => None,
        }
    }
    pub fn obj_state<'a>(&'a self, v: &Value) -> Option<&'a Value> {
        match self.node(v)? {
            Node::Object { state, .. } => state.as_ref(),
            _ => None,
        }
    }
    /// Field of a dataclass-like object: BUILD state dict entry.
    pub fn field<'a>(&'a self, v: &Value, name: &str) -> Option<&'a Value> {
        let st = self.obj_state(v)?;
        // state can be a dict, or (dict, slotstate) tuple
        if let Some(r) = self.dict_get(st, name) {
            return Some(r);
        }
        if let Some(s) = self.seq(st) {
            for part in s {
                if let Some(r) = self.dict_get(part, name) {
                    return Some(r);
                }
            }
        }
        None
    }
    /// torch.Size(...) or plain tuple/list of ints.
    pub fn int_list(&self, v: &Value) -> Option<Vec<i64>> {
        let items = if let Some(("torch", "Size")) = self.class_of(v) {
            let a = self.obj_args(v)?;
            match a.first() {
                Some(x) => self.seq(x)?.to_vec(),
                None => vec![],
            }
        } else {
            self.seq(v)?.to_vec()
        };
        items.iter().map(|x| self.as_int(x)).collect()
    }

    /// Render an arbitrary value as JSON (for displaying small python objects).
    pub fn to_json(&self, v: &Value) -> serde_json::Value {
        self.to_json_depth(v, 0)
    }
    fn to_json_depth(&self, v: &Value, depth: usize) -> serde_json::Value {
        if depth > 32 {
            return json!("<too deep>");
        }
        match v {
            Value::None => serde_json::Value::Null,
            Value::Bool(b) => json!(b),
            Value::Int(i) => json!(i),
            Value::Float(f) => json!(f),
            Value::Str(s) => json!(&**s),
            Value::Bytes(b) => json!(format!("<{} bytes>", b.len())),
            Value::Ref(i) => match &self.nodes[*i] {
                Node::Tuple(x) | Node::List(x) | Node::Set(x) => serde_json::Value::Array(
                    x.iter().map(|e| self.to_json_depth(e, depth + 1)).collect(),
                ),
                Node::Dict(d) => {
                    let mut m = serde_json::Map::new();
                    for (k, val) in d {
                        let key = match k {
                            Value::Str(s) => s.to_string(),
                            other => self.to_json_depth(other, depth + 1).to_string(),
                        };
                        m.insert(key, self.to_json_depth(val, depth + 1));
                    }
                    serde_json::Value::Object(m)
                }
                Node::Global(m, n) => json!(format!("{m}.{n}")),
                Node::Persistent(_) => json!("<persistent>"),
                Node::Object { .. } => {
                    if let Some(l) = self.int_list(v) {
                        return json!(l);
                    }
                    let (m, n) = self.class_of(v).unwrap_or(("?", "?"));
                    if (m, n) == ("torch._utils", "_rebuild_tensor_v2") {
                        return json!("<tensor>");
                    }
                    json!(format!("<{m}.{n}>"))
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_os_system() {
        // pickle.dumps of an object whose __reduce__ returns (os.system, ('echo pwned',)), protocol 2
        let evil = b"\x80\x02cposix\nsystem\nq\x00X\n\x00\x00\x00echo pwnedq\x01\x85q\x02Rq\x03.";
        let err = load(evil, Allow::Checkpoint).unwrap_err();
        assert!(
            format!("{err:#}").contains("refusing pickle global `posix.system`"),
            "{err:#}"
        );
        let evil2 = b"\x80\x04\x95\x1c\x00\x00\x00\x00\x00\x00\x00\x8c\x08builtins\x8c\x04eval\x93\x8c\x011\x85R.";
        assert!(load(evil2, Allow::Checkpoint).is_err());
    }

    #[test]
    fn simple_containers() {
        // pickle.dumps({'a': [1, 2.5, 'x'], 'b': (True, None)}, protocol=4)
        let p = b"\x80\x04\x95\x23\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x01a\x94]\x94(K\x01G@\x04\x00\x00\x00\x00\x00\x00\x8c\x01x\x94e\x8c\x01b\x94\x88N\x86\x94u.";
        let pk = load(p, Allow::Checkpoint).unwrap();
        let j = pk.to_json(&pk.root);
        assert_eq!(
            j,
            serde_json::json!({"a": [1, 2.5, "x"], "b": [true, null]})
        );
    }

    #[test]
    fn self_reference_is_harmless() {
        // l = []; l.append(l); pickle.dumps(l, protocol=2)
        let p = b"\x80\x02]q\x00h\x00a.";
        let pk = load(p, Allow::Checkpoint).unwrap();
        let _ = pk.to_json(&pk.root); // depth-limited, must not overflow
    }
}
