//! Π.15c — NUMA topology detection + worker pinning (Linux-only).
//!
//! Gated on `cfg(target_os = "linux")`. macOS / Windows builds
//! don't compile this module; users targeting NUMA must
//! `cfg(target_os = "linux")` their own usage.
//!
//! Topology comes from sysfs (`/sys/devices/system/node/node*/cpulist`);
//! pinning uses `sched_setaffinity`. No `libnuma` dep for the
//! detection / pinning surface; the heavier NUMA-aware allocator
//! lives in Π.15d.

#![cfg(target_os = "linux")]

use std::fs;
use std::io;

use crate::error::{CodecError, Result};

/// Snapshot of the host's NUMA topology. Built once at pool-
/// construction time and consulted by `pin_current_thread_to_node`.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// CPUs grouped by NUMA node. `nodes[N] = CPU ids on node N`.
    pub nodes: Vec<Vec<u32>>,
}

impl NumaTopology {
    /// Read `/sys/devices/system/node/node*` and parse each
    /// `cpulist` file. Returns a single-node topology with every
    /// online CPU if the sysfs path is absent (e.g. minimal Docker
    /// container) — caller can treat that as "no NUMA awareness
    /// available, fall back to plain rayon".
    pub fn detect() -> Result<Self> {
        let root = "/sys/devices/system/node";
        let entries = match fs::read_dir(root) {
            Ok(e) => e,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Ok(Self::single_node_fallback());
            }
            Err(e) => return Err(io_err(e)),
        };

        let mut nodes_by_id: Vec<(u32, Vec<u32>)> = Vec::new();
        for entry in entries {
            let entry = entry.map_err(io_err)?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            // node0, node1, node2, ...
            let Some(id_str) = name.strip_prefix("node") else {
                continue;
            };
            let Ok(node_id) = id_str.parse::<u32>() else {
                continue;
            };

            let cpulist_path = entry.path().join("cpulist");
            let cpulist = fs::read_to_string(&cpulist_path).map_err(io_err)?;
            let cpus = parse_cpulist(cpulist.trim())?;
            nodes_by_id.push((node_id, cpus));
        }

        if nodes_by_id.is_empty() {
            return Ok(Self::single_node_fallback());
        }

        // Sort by node id so `nodes[N]` indexes the Nth NUMA node.
        nodes_by_id.sort_by_key(|(id, _)| *id);
        let nodes = nodes_by_id.into_iter().map(|(_, c)| c).collect();
        Ok(Self { nodes })
    }

    /// Number of NUMA nodes. `1` on non-NUMA hosts.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// All CPUs on the host (flattened across nodes), deduplicated.
    pub fn all_cpus(&self) -> Vec<u32> {
        let mut all: Vec<u32> = self.nodes.iter().flatten().copied().collect();
        all.sort_unstable();
        all.dedup();
        all
    }

    fn single_node_fallback() -> Self {
        // Best-effort: use `sched_getaffinity` to find online CPUs.
        let cpus = online_cpus().unwrap_or_else(|_| vec![0]);
        Self { nodes: vec![cpus] }
    }
}

/// Pin the calling thread's CPU affinity to the CPUs of `node`.
/// Returns `InvalidInput` if `node` is out of range, or an io error
/// if the syscall fails.
pub fn pin_current_thread_to_node(topology: &NumaTopology, node: usize) -> Result<()> {
    let cpus = topology.nodes.get(node).ok_or_else(|| {
        CodecError::InvalidInput(format!(
            "numa node {node} out of range (have {} nodes)",
            topology.num_nodes()
        ))
    })?;
    set_thread_affinity(cpus)
}

/// Parse a Linux `cpulist` line (e.g. `"0-3,8,12-15"`) into a
/// sorted Vec of CPU ids.
fn parse_cpulist(s: &str) -> Result<Vec<u32>> {
    let mut out: Vec<u32> = Vec::new();
    if s.is_empty() {
        return Ok(out);
    }
    for part in s.split(',') {
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: u32 = lo
                .trim()
                .parse()
                .map_err(|_| CodecError::InvalidInput(format!("bad cpulist range: {part}")))?;
            let hi: u32 = hi
                .trim()
                .parse()
                .map_err(|_| CodecError::InvalidInput(format!("bad cpulist range: {part}")))?;
            for c in lo..=hi {
                out.push(c);
            }
        } else {
            let c: u32 = part
                .trim()
                .parse()
                .map_err(|_| CodecError::InvalidInput(format!("bad cpulist entry: {part}")))?;
            out.push(c);
        }
    }
    out.sort_unstable();
    Ok(out)
}

/// Use the kernel's `sched_getaffinity` to enumerate the CPUs the
/// current process is allowed to run on. Falls back to `[0]` if the
/// syscall fails.
fn online_cpus() -> Result<Vec<u32>> {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // Safety: we just zeroed `set`; sched_getaffinity writes the
    // affinity mask of the current process (pid = 0) into it.
    let rc = unsafe {
        libc::sched_getaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &mut set as *mut libc::cpu_set_t,
        )
    };
    if rc != 0 {
        return Err(io_err(io::Error::last_os_error()));
    }
    let mut cpus: Vec<u32> = Vec::new();
    for cpu in 0u32..(libc::CPU_SETSIZE as u32) {
        // Safety: cpu < CPU_SETSIZE.
        if unsafe { libc::CPU_ISSET(cpu as usize, &set) } {
            cpus.push(cpu);
        }
    }
    if cpus.is_empty() {
        cpus.push(0);
    }
    Ok(cpus)
}

/// Set the current thread's CPU affinity to `cpus`.
fn set_thread_affinity(cpus: &[u32]) -> Result<()> {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &cpu in cpus {
        if (cpu as i32) >= libc::CPU_SETSIZE {
            return Err(CodecError::InvalidInput(format!(
                "cpu id {cpu} ≥ CPU_SETSIZE ({})",
                libc::CPU_SETSIZE
            )));
        }
        // Safety: bounds-checked above.
        unsafe { libc::CPU_SET(cpu as usize, &mut set) };
    }
    // pid = 0 means "the current thread on Linux" for
    // sched_setaffinity (NPTL behaviour; documented in sched(7)).
    let rc = unsafe {
        libc::sched_setaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &set as *const libc::cpu_set_t,
        )
    };
    if rc != 0 {
        return Err(io_err(io::Error::last_os_error()));
    }
    Ok(())
}

fn io_err(e: io::Error) -> CodecError {
    CodecError::InvalidInput(format!("numa: {e}"))
}

/// Build a rayon thread pool with workers pinned to NUMA nodes
/// round-robin. With N CPUs spread across K nodes, worker `w` ends
/// up pinned to the CPUs of node `w % K`.
///
/// `num_threads = 0` defaults to "one worker per available CPU".
pub fn build_numa_pinned_pool(num_threads: usize) -> Result<rayon::ThreadPool> {
    let topology = NumaTopology::detect()?;
    let topology_for_workers = topology.clone();
    let total_cpus = topology.all_cpus().len();
    let num_threads = if num_threads == 0 {
        total_cpus.max(1)
    } else {
        num_threads
    };
    let num_nodes = topology.num_nodes();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .start_handler(move |worker_idx| {
            let node = worker_idx % num_nodes;
            // Best-effort pin: log and continue if it fails (caller
            // sees the worker scheduled on its rayon thread, just not
            // NUMA-pinned). We don't have an integrated logger; the
            // ignored-error here is the documented fallback.
            let _ = pin_current_thread_to_node(&topology_for_workers, node);
        })
        .build()
        .map_err(|e| CodecError::InvalidInput(format!("rayon pool build: {e}")))?;
    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpulist_round_trip() {
        assert_eq!(parse_cpulist("0").unwrap(), vec![0]);
        assert_eq!(parse_cpulist("0-3").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpulist("0,2,4").unwrap(), vec![0, 2, 4]);
        assert_eq!(
            parse_cpulist("0-3,8,12-13").unwrap(),
            vec![0, 1, 2, 3, 8, 12, 13]
        );
        assert!(parse_cpulist("").unwrap().is_empty());
    }

    #[test]
    fn parse_cpulist_rejects_garbage() {
        assert!(parse_cpulist("notanumber").is_err());
        assert!(parse_cpulist("0-").is_err());
    }

    #[test]
    fn detect_returns_at_least_one_node() {
        // On real Linux this returns the actual topology; in
        // sysfs-less environments (rare) it falls back to a
        // single-node topology. Either way: at least one node.
        let t = NumaTopology::detect().expect("detect must not panic");
        assert!(t.num_nodes() >= 1);
        assert!(!t.all_cpus().is_empty());
    }

    #[test]
    fn build_numa_pinned_pool_returns_working_pool() {
        let pool = build_numa_pinned_pool(2).unwrap();
        let sum: i32 = pool.install(|| (1..=100).sum());
        assert_eq!(sum, 5050);
    }
}
