/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APU_XMA_FRAME_DUMPER_H_
#define XENIA_APU_XMA_FRAME_DUMPER_H_

#include <cstdint>
#include <filesystem>
#include <string>

namespace xe {
namespace apu {

// Captures the per-frame AVPackets that XmaContextNew assembles for
// AV_CODEC_ID_XMAFRAMES, in the order the runtime decodes them.  Lets
// xma-replay feed the same bytes through a fresh codec context per
// xenia-side context_id and reconstruct each context's PCM
// independently of the runtime's output-ring-buffer scheduling.
//
// Two files are written per session:
//
//   xma_<timestamp>.xframes  - sequence of [u32 size][bytes] records,
//                              one per recorded frame.  Bytes are the
//                              AVPacket payload the runtime hands to
//                              avcodec_send_packet:
//                                  byte[0]    = ((padding_start & 7) << 5)
//                                             | ((padding_end   & 7) << 2)
//                                  byte[1..]  = packed frame bits
//                              i.e. ready to be replayed as-is.
//
//   xma_<timestamp>.manifest - tab-separated index, one row per record:
//                                seq  context_id  time_us  offset
//                                size  rate_hz  channels  buf_idx
//                                pkt_idx  read_off
//                              `time_us` is microseconds since the
//                              session started; `offset` is the byte
//                              position of the size prefix in the
//                              .xframes file; the trailing `buf_idx`
//                              / `pkt_idx` / `read_off` triple is
//                              diagnostic state captured at the
//                              moment of frame selection.
class XmaFrameDumper {
 public:
  static bool IsEnabled();
  static std::string xframes_path();
  static bool Enable(const std::filesystem::path& output_dir);
  static void Disable();

  // Record one frame's AVPacket payload (xma_frame_ buffer) the
  // runtime is about to hand to avcodec_send_packet. No-op when the
  // dumper is disabled.
  static void RecordFrame(uint32_t context_id, uint32_t rate_hz,
                          uint8_t channels, uint8_t buf_idx,
                          uint16_t packet_index, uint32_t read_off_bits,
                          const uint8_t* frame, int frame_size);

 private:
  XmaFrameDumper() = delete;
};

}  // namespace apu
}  // namespace xe

#endif  // XENIA_APU_XMA_FRAME_DUMPER_H_
