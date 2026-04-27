/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/xma_frame_dumper.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <system_error>

#include "xenia/base/logging.h"

namespace xe {
namespace apu {

namespace {

std::mutex g_mutex;
std::FILE* g_xframes_file = nullptr;
std::FILE* g_manifest_file = nullptr;
std::string g_xframes_path;
std::atomic<bool> g_enabled{false};
uint64_t g_seq = 0;
uint64_t g_xframes_offset = 0;
std::chrono::steady_clock::time_point g_start_time;

void CloseFilesLocked() {
  if (g_xframes_file) {
    std::fclose(g_xframes_file);
    g_xframes_file = nullptr;
  }
  if (g_manifest_file) {
    std::fclose(g_manifest_file);
    g_manifest_file = nullptr;
  }
  g_xframes_path.clear();
  g_seq = 0;
  g_xframes_offset = 0;
}

}  // namespace

bool XmaFrameDumper::IsEnabled() {
  return g_enabled.load(std::memory_order_acquire);
}

std::string XmaFrameDumper::xframes_path() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_xframes_path;
}

bool XmaFrameDumper::Enable(const std::filesystem::path& output_dir) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_enabled.load(std::memory_order_relaxed)) {
    return true;
  }

  std::error_code ec;
  std::filesystem::create_directories(output_dir, ec);
  if (ec) {
    XELOGE("XmaFrameDumper: failed to create '{}': {}", output_dir.string(),
           ec.message());
    return false;
  }

  // Timestamp: YYYY-MM-DDTHH-MM-SS — matches the screenshot dumper.
  auto now = std::time(nullptr);
  char ts[32];
  std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H-%M-%S", std::localtime(&now));
  std::string base = std::string("xma_") + ts;

  auto xframes_path = output_dir / (base + ".xframes");
  auto manifest_path = output_dir / (base + ".manifest");

  g_xframes_file = std::fopen(xframes_path.string().c_str(), "wb");
  if (!g_xframes_file) {
    XELOGE("XmaFrameDumper: cannot open '{}'", xframes_path.string());
    return false;
  }
  g_manifest_file = std::fopen(manifest_path.string().c_str(), "wb");
  if (!g_manifest_file) {
    XELOGE("XmaFrameDumper: cannot open '{}'", manifest_path.string());
    std::fclose(g_xframes_file);
    g_xframes_file = nullptr;
    return false;
  }

  // Manifest header.
  std::fprintf(
      g_manifest_file,
      "# xframes capture: %s\n"
      "# columns (tab-separated):\n"
      "#   seq         monotonic submission order across contexts\n"
      "#   context_id  XmaContext id (0..n)\n"
      "#   time_us     microseconds since session start\n"
      "#   offset      byte offset in .xframes of [u32 size][bytes]\n"
      "#   size        AVPacket payload size (frame bytes, varies)\n"
      "#   rate_hz     decoder sample rate at submission\n"
      "#   channels    decoder channel count (1 or 2)\n"
      "#   buf_idx     XMA_CONTEXT_DATA.current_buffer (0 or 1)\n"
      "#   pkt_idx     packet index within that input buffer\n"
      "#   read_off    XMA_CONTEXT_DATA.input_buffer_read_offset (bits)\n"
      "seq\tcontext_id\ttime_us\toffset\tsize\trate_hz\tchannels\t"
      "buf_idx\tpkt_idx\tread_off\n",
      xframes_path.filename().string().c_str());
  std::fflush(g_manifest_file);

  g_xframes_path = xframes_path.string();
  g_seq = 0;
  g_xframes_offset = 0;
  g_start_time = std::chrono::steady_clock::now();
  g_enabled.store(true, std::memory_order_release);

  XELOGI("XmaFrameDumper: capturing to '{}'", g_xframes_path);
  return true;
}

void XmaFrameDumper::Disable() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_enabled.store(false, std::memory_order_release);
  XELOGI("XmaFrameDumper: stopped after {} frames ({} bytes)", g_seq,
         g_xframes_offset);
  CloseFilesLocked();
}

void XmaFrameDumper::RecordFrame(uint32_t context_id, uint32_t rate_hz,
                                 uint8_t channels, uint8_t buf_idx,
                                 uint16_t packet_index, uint32_t read_off_bits,
                                 const uint8_t* frame, int frame_size) {
  if (!g_enabled.load(std::memory_order_acquire) || frame_size <= 0 || !frame) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_enabled.load(std::memory_order_relaxed) || !g_xframes_file ||
      !g_manifest_file) {
    return;
  }

  const uint64_t time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - g_start_time)
          .count();

  const uint32_t size_le = static_cast<uint32_t>(frame_size);
  const uint64_t record_offset = g_xframes_offset;
  std::fwrite(&size_le, sizeof(size_le), 1, g_xframes_file);
  std::fwrite(frame, 1, static_cast<size_t>(frame_size), g_xframes_file);
  std::fflush(g_xframes_file);

  std::fprintf(
      g_manifest_file, "%llu\t%u\t%llu\t%llu\t%d\t%u\t%u\t%u\t%u\t%u\n",
      static_cast<unsigned long long>(g_seq), context_id,
      static_cast<unsigned long long>(time_us),
      static_cast<unsigned long long>(record_offset), frame_size, rate_hz,
      static_cast<unsigned>(channels), static_cast<unsigned>(buf_idx),
      static_cast<unsigned>(packet_index), read_off_bits);
  std::fflush(g_manifest_file);

  g_xframes_offset += sizeof(size_le) + static_cast<uint64_t>(frame_size);
  ++g_seq;
}

}  // namespace apu
}  // namespace xe
