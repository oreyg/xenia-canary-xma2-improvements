/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/loc/memory_scan.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "xenia/base/cvar.h"
#include "xenia/base/exception_handler.h"
#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/base/platform.h"
#include "xenia/base/string.h"
#include "xenia/cpu/backend/backend.h"
#include "xenia/cpu/backend/code_cache.h"
#include "xenia/cpu/function.h"
#include "xenia/cpu/processor.h"
#include "xenia/memory.h"
#if XE_PLATFORM_WIN32
#include "xenia/base/platform_win.h"
#endif

DEFINE_string(
    scan_text, "",
    "Text (UTF-8) to search for in guest memory when pressing Scroll Lock.\n"
    "The string is encoded into several candidate encodings (Shift-JIS, "
    "EUC-JP, UTF-8, UTF-16BE/LE) and each is scanned for; matches are logged "
    "with their guest address and the encoding that hit. Used to locate game "
    "text buffers.",
    "General");

DEFINE_string(
    watch_address, "",
    "Guest address (hex, e.g. 0x41501544) to watch when pressing Home.\n"
    "Protects the page containing the address and logs the first guest "
    "function that reads/writes it, then disarms. Used to find the code that "
    "consumes a located text buffer.",
    "General");

DEFINE_string(
    scan_pointer, "",
    "Guest address (hex, e.g. 0x44588128) to search for as a 4-byte big-endian "
    "pointer when pressing Scroll Lock (in addition to scan_text).\n"
    "Logs every guest address whose memory holds this value - i.e. structures "
    "that reference the buffer. Used to find the object/vtable behind a text "
    "buffer.",
    "General");

namespace xe {
namespace loc {

using xe::cpu::Processor;


struct EncodedPattern {
  const char* name;
  std::vector<uint8_t> bytes;
};

std::string BytesToHex(const std::vector<uint8_t>& bytes) {
  std::string out;
  out.reserve(bytes.size() * 3);
  for (uint8_t b : bytes) {
    char tmp[4];
    std::snprintf(tmp, sizeof(tmp), "%02X ", b);
    out += tmp;
  }
  if (!out.empty()) {
    out.pop_back();
  }
  return out;
}

// UTF-16 byte encoding (BE/LE) - portable, used to find guest UTF-16 buffers.
std::vector<uint8_t> EncodeUtf16(const std::u16string& w, bool big_endian) {
  std::vector<uint8_t> out;
  out.reserve(w.size() * 2);
  for (char16_t c : w) {
    uint16_t u = static_cast<uint16_t>(c);
    if (big_endian) {
      out.push_back(static_cast<uint8_t>(u >> 8));
      out.push_back(static_cast<uint8_t>(u & 0xFF));
    } else {
      out.push_back(static_cast<uint8_t>(u & 0xFF));
      out.push_back(static_cast<uint8_t>(u >> 8));
    }
  }
  return out;
}

#if XE_PLATFORM_WIN32
// Shift-JIS / EUC-JP via the Win32 codepage API. No portable equivalent in
// xe::base (would need iconv/ICU), so these two encodings are Windows-only; the
// scan still runs utf-8/utf-16 on every platform.
std::vector<uint8_t> EncodeCodepage(const std::u16string& w, UINT code_page) {
  if (w.empty()) {
    return {};
  }
  const wchar_t* wdata = reinterpret_cast<const wchar_t*>(w.data());
  int n = WideCharToMultiByte(code_page, 0, wdata, static_cast<int>(w.size()),
                              nullptr, 0, nullptr, nullptr);
  if (n <= 0) {
    return {};
  }
  std::vector<uint8_t> out(n);
  WideCharToMultiByte(code_page, 0, wdata, static_cast<int>(w.size()),
                      reinterpret_cast<char*>(out.data()), n, nullptr, nullptr);
  return out;
}
#endif  // XE_PLATFORM_WIN32

std::vector<EncodedPattern> BuildSearchPatterns(const std::string& text) {
  std::vector<EncodedPattern> patterns;
  // UTF-8 is just the raw cvar bytes.
  patterns.push_back({"utf-8", std::vector<uint8_t>(text.begin(), text.end())});
  const std::u16string wide = xe::to_utf16(text);
  patterns.push_back({"utf-16be", EncodeUtf16(wide, true)});
  patterns.push_back({"utf-16le", EncodeUtf16(wide, false)});
#if XE_PLATFORM_WIN32
  patterns.push_back({"shift-jis", EncodeCodepage(wide, 932)});
  patterns.push_back({"euc-jp", EncodeCodepage(wide, 20932)});
#endif  // XE_PLATFORM_WIN32
  // Drop any encodings that failed to produce bytes.
  patterns.erase(
      std::remove_if(patterns.begin(), patterns.end(),
                     [](const EncodedPattern& p) { return p.bytes.empty(); }),
      patterns.end());
  return patterns;
}

struct ScanHit {
  int pattern_index;
  uint32_t guest_addr;
};

// Scans a single committed+readable region for every pattern, appending matches
// to out_hits. The caller only passes regions it has verified are committed and
// readable, so the in-region reads can't fault.
void FindHitsInRegion(const uint8_t* host, uint32_t region_size,
                      uint32_t region_base, const EncodedPattern* patterns,
                      size_t pattern_count, std::vector<ScanHit>* out_hits) {
  for (size_t pi = 0; pi < pattern_count; ++pi) {
    const uint8_t* pat = patterns[pi].bytes.data();
    const size_t plen = patterns[pi].bytes.size();
    if (plen == 0 || plen > region_size) {
      continue;
    }
    const uint8_t first = pat[0];
    for (uint32_t off = 0; off + plen <= region_size; ++off) {
      if (host[off] != first) {
        continue;
      }
      if (std::memcmp(host + off, pat, plen) == 0) {
        out_hits->push_back({static_cast<int>(pi), region_base + off});
      }
    }
  }
}

// Scans all committed guest memory for the given text in several encodings and
// logs every match with its guest address and encoding.
void ScanGuestMemoryForText(Memory* memory) {
  const std::string& text = cvars::scan_text;
  if (text.empty()) {
    XELOGI("[scan] scan_text is empty; set --scan_text=... to search.");
    return;
  }
  auto patterns = BuildSearchPatterns(text);
  XELOGI("[scan] Searching for '{}' ({} encodings):", text, patterns.size());
  for (const auto& p : patterns) {
    XELOGI("[scan]   {:<9} : {}", p.name, BytesToHex(p.bytes));
  }

  // Virtual base of each guest heap. Physical memory is covered by the
  // A0/C0/E0 virtual aliases.
  static const uint32_t kHeapBases[] = {0x00000000, 0x40000000, 0x80000000,
                                        0x90000000, 0xA0000000, 0xC0000000,
                                        0xE0000000};
  int total_hits = 0;
  std::vector<ScanHit> hits;
  for (uint32_t heap_base : kHeapBases) {
    BaseHeap* heap = memory->LookupHeap(heap_base);
    if (!heap) {
      continue;
    }
    const uint32_t end = heap->heap_base() + heap->heap_size();
    uint32_t cur = heap->heap_base();
    while (cur < end) {
      HeapAllocationInfo info = {};
      if (!heap->QueryRegionInfo(cur, &info)) {
        break;
      }
      uint32_t region_size = info.region_size;
      if (region_size == 0) {
        cur += heap->page_size();
        continue;
      }
      // Only scan committed, readable pages; reading committed-but-NoAccess
      // pages (guard pages, protected zero page, etc.) would fault.
      if ((info.state & kMemoryAllocationCommit) &&
          (info.protect & kMemoryProtectRead)) {
        const uint8_t* host = memory->TranslateVirtual<const uint8_t*>(cur);
        hits.clear();
        FindHitsInRegion(host, region_size, cur, patterns.data(),
                         patterns.size(), &hits);
        for (const ScanHit& h : hits) {
          XELOGI("[scan]   HIT enc={:<9} guest=0x{:08X}",
                 patterns[h.pattern_index].name, h.guest_addr);
          ++total_hits;
        }
      }
      cur += region_size;
    }
  }
  XELOGI("[scan] Done. {} match(es).", total_hits);
}

// ---------------------------------------------------------------------------
// Read-watch: protects the host page containing a guest address and logs the
// first guest function that accesses it, then disarms. Used to locate the code
// that consumes a text buffer found by the scanner.
// ---------------------------------------------------------------------------
struct MemoryWatchState {
  std::atomic<bool> armed{false};
  void* page_host = nullptr;
  size_t page_len = 0;
  uint32_t target_guest = 0;
  xe::memory::PageAccess old_access = xe::memory::PageAccess::kReadWrite;
  Memory* memory = nullptr;
  cpu::Processor* processor = nullptr;
  bool handler_installed = false;
};

MemoryWatchState g_memory_watch;

// Captured by the exception handler, drained on the UI thread. Keep this raw:
// the handler runs inside a vectored exception handler, where logging, locks,
// and code-cache lookups are unsafe. The guest pseudo-stack, however, is
// lock-free and must be sampled here while the faulting guest thread is live.
struct MemoryWatchResult {
  std::atomic<bool> pending{false};
  uint64_t host_pc = 0;
  uint64_t fault_host = 0;
  Exception::AccessViolationOperation op =
      Exception::AccessViolationOperation::kUnknown;
  bool have_stack = false;
  cpu::backend::GuestPseudoStackTrace stack{};
};

MemoryWatchResult g_memory_watch_result;

bool MemoryWatchExceptionHandler(Exception* ex, void*) {
  if (ex->code() != Exception::Code::kAccessViolation) {
    return false;
  }
  void* page = g_memory_watch.page_host;
  if (!page) {
    return false;
  }
  const uint64_t fault = ex->fault_address();
  const uint64_t page_start = reinterpret_cast<uint64_t>(page);
  if (fault < page_start || fault >= page_start + g_memory_watch.page_len) {
    return false;  // Not our watched page.
  }
  // The page is only ever protected by us, so any fault here is ours. Sample
  // ONLY raw exception fields here - no Xenia calls, no logging, no locks. The
  // UI thread maps host_pc to a guest function later.
  if (g_memory_watch.armed.exchange(false)) {
    g_memory_watch_result.host_pc = ex->pc();
    g_memory_watch_result.fault_host = fault;
    g_memory_watch_result.op = ex->access_violation_operation();
    g_memory_watch_result.have_stack = false;
    g_memory_watch_result.pending.store(true, std::memory_order_release);
  }
  xe::memory::Protect(page, g_memory_watch.page_len, g_memory_watch.old_access,
                      nullptr);
  // Hard anti-loop guard: stop matching after the first handled fault so a
  // failed restore can never spin into an infinite fault loop (emulator freeze).
  g_memory_watch.page_host = nullptr;
  return true;  // Handled: re-execute the now-accessible instruction.
}

// Drains a pending watch hit to the log (called on the UI thread, where logging
// and code-cache lookups are safe). Returns true if a result was drained.
bool DumpMemoryWatchResult(Memory* memory, cpu::Processor* processor) {
  if (!g_memory_watch_result.pending.load(std::memory_order_acquire)) {
    return false;
  }
  const uint64_t host_pc = g_memory_watch_result.host_pc;
  uint32_t fn_base = 0;
  uint32_t guest_pc = 0;
  auto* code_cache = processor->backend()->code_cache();
  auto* guest_fn = code_cache ? code_cache->LookupFunction(host_pc) : nullptr;
  if (guest_fn) {
    fn_base = guest_fn->address();
    guest_pc = guest_fn->MapMachineCodeToGuestAddress(host_pc);
  }
  const uint32_t fault_guest = memory->HostToGuestVirtual(
      reinterpret_cast<void*>(g_memory_watch_result.fault_host));
  const char* op = "unknown";
  switch (g_memory_watch_result.op) {
    case Exception::AccessViolationOperation::kRead:
      op = "read";
      break;
    case Exception::AccessViolationOperation::kWrite:
      op = "write";
      break;
    default:
      break;
  }
  XELOGI(
      "[watch] HIT access guest=0x{:08X} (target=0x{:08X}) op={} func=0x{:08X} "
      "guest_pc=0x{:08X} host_pc=0x{:016X}",
      fault_guest, g_memory_watch.target_guest, op, fn_base, guest_pc, host_pc);
  g_memory_watch_result.pending.store(false, std::memory_order_release);
  return true;
}

void ArmMemoryWatch(Memory* memory, cpu::Processor* processor) {
  const std::string& addr_str = cvars::watch_address;
  if (addr_str.empty()) {
    XELOGI("[watch] watch_address is empty; set --watch_address=0x... first.");
    return;
  }
  const uint32_t guest_addr =
      static_cast<uint32_t>(std::strtoul(addr_str.c_str(), nullptr, 16));
  if (!guest_addr) {
    XELOGI("[watch] invalid watch_address '{}'.", addr_str);
    return;
  }
  BaseHeap* heap = memory->LookupHeap(guest_addr);
  uint32_t protect = 0;
  if (!heap || !heap->QueryProtect(guest_addr, &protect) ||
      !(protect & kMemoryProtectRead)) {
    XELOGI("[watch] 0x{:08X} is not committed/readable.", guest_addr);
    return;
  }
  uint8_t* host = memory->TranslateVirtual<uint8_t*>(guest_addr);
  const size_t page_size = 4096;
  void* page = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(host) &
                                       ~(static_cast<uintptr_t>(page_size) - 1));

  if (!g_memory_watch.handler_installed) {
    ExceptionHandler::Install(MemoryWatchExceptionHandler, nullptr);
    g_memory_watch.handler_installed = true;
  }

  // Capture the current protection before arming, so an early fault (between
  // arming and protecting) can restore it correctly.
  size_t query_len = page_size;
  xe::memory::PageAccess old_access = xe::memory::PageAccess::kReadWrite;
  xe::memory::QueryProtect(page, query_len, old_access);

  g_memory_watch.memory = memory;
  g_memory_watch.processor = processor;
  g_memory_watch.target_guest = guest_addr;
  g_memory_watch.page_host = page;
  g_memory_watch.page_len = page_size;
  g_memory_watch.old_access = old_access;
  g_memory_watch.armed.store(true, std::memory_order_release);

  if (!xe::memory::Protect(page, page_size, xe::memory::PageAccess::kNoAccess,
                           nullptr)) {
    g_memory_watch.armed.store(false, std::memory_order_release);
    XELOGI("[watch] failed to protect page for 0x{:08X}.", guest_addr);
    return;
  }
  XELOGI(
      "[watch] Armed on guest=0x{:08X}. Trigger the dialog; the first accessor "
      "will be logged.",
      guest_addr);
}


}  // namespace loc
}  // namespace xe
