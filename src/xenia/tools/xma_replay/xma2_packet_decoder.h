/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_TOOLS_XMA_REPLAY_XMA2_PACKET_DECODER_H_
#define XENIA_TOOLS_XMA_REPLAY_XMA2_PACKET_DECODER_H_

#include <cstddef>
#include <cstdint>

struct AVCodecContext;
struct AVPacket;
struct AVFrame;

namespace xe {
namespace tools {

// Default Windows speaker mask for `channels` channels. Mirrors
// vgmstream's ffmpeg_make_riff_xma2 mapping. Returns 0 if `channels`
// is outside the supported 1..8 range; caller should fall back to
// FFmpeg's default channel layout in that case.
uint32_t DefaultChannelMask(int channels);

// Fill the 34-byte XMA2WAVEFORMATEX-style extradata block that
// FFmpeg's AV_CODEC_ID_XMA2 init reads. Only num_streams (LE u16 @ 0)
// and channel_mask (LE u32 @ 2) are consumed; the remaining 28 bytes
// are zero-padded so the size check passes.
void BuildXma2Extradata(uint8_t buf[34], uint16_t num_streams,
                        uint32_t channel_mask);

// Owns an AV_CODEC_ID_XMA2 decoder context plus its working AVPacket
// and AVFrame, and exposes a SendPacket / ReceiveFrame loop matching
// FFmpeg's send/receive API.  One instance decodes one logical XMA
// stream (1..N internal multi-channel streams as encoded in the file
// header); the codec demultiplexes streams internally based on the
// per-packet skip_packets field.
class Xma2PacketDecoder {
 public:
  // Output frame view. Pointers are borrowed from the owned AVFrame
  // and remain valid until the next SendPacket / ReceiveFrame / Close.
  struct Frame {
    // [channels] pointers, each pointing to [nb_samples] floats.
    const float* const* planar;
    int nb_samples;
    int channels;
    int sample_rate;
  };

  Xma2PacketDecoder();
  ~Xma2PacketDecoder();

  Xma2PacketDecoder(const Xma2PacketDecoder&) = delete;
  Xma2PacketDecoder& operator=(const Xma2PacketDecoder&) = delete;

  // total_channels is summed across all streams; num_streams is the
  // XMA2 stream count from the file header (each carrying 1 or 2
  // channels). Returns 0 on success or a negative AVERROR.
  int Open(int sample_rate, int total_channels, int num_streams);

  // Releases all owned FFmpeg objects. Safe to call repeatedly.
  void Close();

  bool is_open() const { return ctx_ != nullptr; }
  int sample_rate() const;
  int channels() const;

  // Feed one XMA2 physical packet (typically 2KB). Internally copies
  // into a padded scratch buffer so callers don't need to honor
  // AV_INPUT_BUFFER_PADDING_SIZE. Returns 0 / negative AVERROR.
  int SendPacket(const uint8_t* data, size_t size);

  // End-of-stream marker. After this, ReceiveFrame() drains queued
  // output frames and eventually returns AVERROR_EOF.
  int SendEof();

  // Drop all internal codec state without reallocating. Use when the
  // input position is rewound (e.g. XMA loop wrap). Buffered output
  // samples that haven't been consumed yet are lost.
  void Flush();

  // Pull next output frame. Returns:
  //   0               - frame populated in *out
  //   AVERROR(EAGAIN) - need more input via SendPacket
  //   AVERROR_EOF     - no more output (after SendEof)
  //   negative        - fatal error (codec state may be unusable)
  int ReceiveFrame(Frame* out);

 private:
  AVCodecContext* ctx_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVFrame* frame_ = nullptr;
};

}  // namespace tools
}  // namespace xe

#endif  // XENIA_TOOLS_XMA_REPLAY_XMA2_PACKET_DECODER_H_
