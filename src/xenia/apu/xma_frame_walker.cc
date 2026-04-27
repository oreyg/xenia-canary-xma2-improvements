/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/xma_frame_walker.h"

#include <cstring>

#include "xenia/base/bit_stream.h"

namespace xe {
namespace apu {
namespace xma {

namespace {

// Combined two-packet payload buffer (current + next, headers stripped).
// Used as scratch by ResolveSplitFrameSize / BuildAvPacketPayload to run
// a contiguous BitStream across the boundary. Fits comfortably on the
// stack at audio rates.
constexpr size_t kCombinedPayloadBytes = kBytesPerPacketData * 2;
constexpr size_t kCombinedPayloadBits = kCombinedPayloadBytes * 8;

}  // namespace

PacketWalk InspectPacket(const uint8_t* packet, uint32_t frame_offset_bits) {
  PacketWalk pw{};

  const uint32_t first_frame_offset = GetPacketFrameOffset(packet);
  // BitStream takes a non-const pointer; we never write through it here.
  BitStream stream(const_cast<uint8_t*>(packet), kBitsPerPacket);
  stream.SetOffset(first_frame_offset);

  // Caller's offset is mid-stream split-frame tail: report the gap to skip.
  if (frame_offset_bits < first_frame_offset) {
    pw.current_frame = 0;
    pw.current_frame_size = first_frame_offset - frame_offset_bits;
  }

  while (true) {
    if (stream.BitsRemaining() < kBitsPerFrameHeader) break;

    const uint64_t fs = stream.Peek(kBitsPerFrameHeader);
    if (fs == 0 || fs == kMaxFrameLength) break;

    if (stream.offset_bits() == frame_offset_bits) {
      pw.current_frame = pw.frame_count;
      pw.current_frame_size = static_cast<uint32_t>(fs);
    }

    pw.frame_count++;

    if (fs > stream.BitsRemaining()) break;
    stream.Advance(fs - 1);
    if (stream.Read(1) == 0) break;
  }

  // XMA2 fallback: if packet[0]>>2 exceeds what the trailing-bit walk
  // counted, accept the header value. Lets us recover frames that the
  // walk terminated on prematurely.
  if (IsPacketXma2Type(packet)) {
    const uint8_t xma2_count = GetPacketFrameCount(packet);
    if (xma2_count > pw.frame_count) {
      if (pw.current_frame_size == 0) {
        pw.current_frame = pw.frame_count;
      }
      pw.frame_count = xma2_count;
    }
  }
  return pw;
}

uint32_t ResolveSplitFrameSize(const uint8_t* current_packet,
                               const uint8_t* next_packet,
                               uint32_t frame_offset_bits) {
  uint8_t scratch[kCombinedPayloadBytes];
  std::memcpy(scratch, current_packet + kBytesPerPacketHeader,
              kBytesPerPacketData);
  std::memcpy(scratch + kBytesPerPacketData,
              next_packet + kBytesPerPacketHeader, kBytesPerPacketData);

  BitStream src(scratch, kCombinedPayloadBits);
  src.SetOffset(frame_offset_bits - kBitsPerPacketHeader);

  const uint64_t fs = src.Peek(kBitsPerFrameHeader);
  if (fs == kMaxFrameLength) return 0;
  return static_cast<uint32_t>(fs);
}

int BuildAvPacketPayload(const uint8_t* current_packet,
                         const uint8_t* next_packet, uint32_t frame_offset_bits,
                         uint32_t frame_size_bits, uint8_t* out_avpacket,
                         int out_avpacket_size) {
  if (frame_size_bits == 0 || out_avpacket_size <= 0) return 0;

  // Frame extends past current_packet's payload?  We need next_packet
  // to source the overflow bits.
  const uint32_t available_in_current_payload =
      kBitsPerPacket - frame_offset_bits;
  const bool needs_next = frame_size_bits > available_in_current_payload;
  if (needs_next && !next_packet) return 0;

  // Combine payloads (header bytes stripped) into scratch so the
  // BitStream copy is contiguous across the boundary.
  uint8_t scratch[kCombinedPayloadBytes];
  std::memcpy(scratch, current_packet + kBytesPerPacketHeader,
              kBytesPerPacketData);
  if (needs_next) {
    std::memcpy(scratch + kBytesPerPacketData,
                next_packet + kBytesPerPacketHeader, kBytesPerPacketData);
  } else {
    std::memset(scratch + kBytesPerPacketData, 0, kBytesPerPacketData);
  }

  BitStream src(scratch, kCombinedPayloadBits);
  src.SetOffset(frame_offset_bits - kBitsPerPacketHeader);

  // Extra safety: refuse if frame would read past the combined buffer.
  if (src.BitsRemaining() < frame_size_bits) return 0;

  // Zero the output, copy frame bits starting at byte[1]. Copy returns
  // the leading bit offset within byte[1] (= padding_start, 0..7).
  std::memset(out_avpacket, 0, out_avpacket_size);
  const uint32_t padding_start =
      static_cast<uint32_t>(src.Copy(out_avpacket + 1, frame_size_bits));

  // Total AVPacket size = header byte + ceil((padding_start + frame)/8).
  const size_t pkt_size = 1 + ((padding_start + frame_size_bits) / 8) +
                          (((padding_start + frame_size_bits) % 8) ? 1 : 0);
  if (static_cast<int>(pkt_size) > out_avpacket_size) return 0;

  const size_t padding_end =
      pkt_size * 8 - (8 + padding_start + frame_size_bits);
  out_avpacket[0] = static_cast<uint8_t>(((padding_start & 7) << 5) |
                                         ((padding_end & 7) << 2));

  return static_cast<int>(pkt_size);
}

}  // namespace xma
}  // namespace apu
}  // namespace xe
