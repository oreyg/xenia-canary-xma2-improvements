/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APU_XMA_FRAME_WALKER_H_
#define XENIA_APU_XMA_FRAME_WALKER_H_

#include <cstdint>

#include "xenia/apu/xma_helpers.h"

namespace xe {
namespace apu {
namespace xma {

// Maximum valid bit position for a frame to BEGIN within a 2KB packet.
// Any packet whose first-frame-offset header exceeds this is corrupt.
static constexpr uint32_t kMaxFrameSizeinBits =
    kBitsPerPacket - kBitsPerPacketHeader;

// Result of walking the frame headers within a single 2KB packet.
// Mirrors the runtime kPacketInfo struct used by xma_context_new /
// xma_context_v3 so consumers can keep their existing field names.
struct PacketWalk {
  uint8_t frame_count = 0;
  uint8_t current_frame = 0;
  // Bit length of the frame at frame_offset_bits. 0 when the frame's
  // header straddles a packet boundary - caller must call
  // ResolveSplitFrameSize() with the next packet to recover the real size.
  uint32_t current_frame_size = 0;

  bool IsLastFrameInPacket() const {
    return frame_count == 0 || current_frame == frame_count - 1;
  }
};

// Cheap check: is this packet's first-frame-offset header in range?
// Lets a state machine fast-forward past obviously-corrupt packets
// without trying to walk them.
inline bool IsPacketHeaderValid(const uint8_t* packet) {
  return GetPacketFrameOffset(packet) <= kMaxFrameSizeinBits;
}

// Walk the frame headers in a single 2KB packet using V3 hybrid
// semantics: trailing-bit walk first, then accept packet[0]>>2 as the
// authoritative frame count (XMA2 packets only) when the walk
// under-counts. frame_offset_bits is the bit offset within the packet
// of the frame the caller wants to identify; if it lies before the
// first frame in the packet (i.e. mid-stream split-frame tail), the
// returned current_frame_size is the gap to skip.
PacketWalk InspectPacket(const uint8_t* packet, uint32_t frame_offset_bits);

// Resolve a 15-bit frame_size header that straddles a packet boundary
// by combining current+next packet payloads and peeking the prefix at
// frame_offset_bits. Returns 0 if the resolved value is the reserved
// "invalid" marker (kMaxFrameLength).
uint32_t ResolveSplitFrameSize(const uint8_t* current_packet,
                               const uint8_t* next_packet,
                               uint32_t frame_offset_bits);

// Build the AVPacket payload that AV_CODEC_ID_XMAFRAMES eats:
//   byte[0]    = ((padding_start & 7) << 5) | ((padding_end & 7) << 2)
//   byte[1..]  = packed frame bits
// Returns the AVPacket byte size on success, 0 on:
//   - frame_size_bits == 0
//   - frame extends past current_packet but next_packet is nullptr
//   - resulting AVPacket would exceed out_avpacket_size
// next_packet may be nullptr when the caller knows the frame fits
// entirely in current_packet's payload.
int BuildAvPacketPayload(const uint8_t* current_packet,
                         const uint8_t* next_packet, uint32_t frame_offset_bits,
                         uint32_t frame_size_bits, uint8_t* out_avpacket,
                         int out_avpacket_size);

}  // namespace xma
}  // namespace apu
}  // namespace xe

#endif  // XENIA_APU_XMA_FRAME_WALKER_H_
