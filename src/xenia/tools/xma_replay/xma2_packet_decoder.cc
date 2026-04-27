/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/tools/xma_replay/xma2_packet_decoder.h"

#include <cstring>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/mem.h>
#include <libavutil/samplefmt.h>
}

namespace xe {
namespace tools {

namespace {

constexpr size_t kXma2ExtradataSize = 34;
constexpr size_t kXma2PacketSize = 2048;

}  // namespace

uint32_t DefaultChannelMask(int channels) {
  switch (channels) {
    case 1:
      return 0x04;  // FC
    case 2:
      return 0x01 | 0x02;  // FL FR
    case 3:
      return 0x01 | 0x02 | 0x08;  // FL FR LF
    case 4:
      return 0x01 | 0x02 | 0x10 | 0x20;  // FL FR BL BR
    case 5:
      return 0x01 | 0x02 | 0x08 | 0x10 | 0x20;  // FL FR LF BL BR
    case 6:
      return 0x01 | 0x02 | 0x04 | 0x08 | 0x10 | 0x20;  // 5.1
    case 7:
      return 0x01 | 0x02 | 0x04 | 0x08 | 0x10 | 0x20 | 0x100;
    case 8:
      return 0x01 | 0x02 | 0x04 | 0x08 | 0x10 | 0x20 | 0x40 | 0x80;
    default:
      return 0;
  }
}

void BuildXma2Extradata(uint8_t buf[34], uint16_t num_streams,
                        uint32_t channel_mask) {
  std::memset(buf, 0, kXma2ExtradataSize);
  buf[0] = uint8_t(num_streams & 0xFF);
  buf[1] = uint8_t((num_streams >> 8) & 0xFF);
  buf[2] = uint8_t(channel_mask & 0xFF);
  buf[3] = uint8_t((channel_mask >> 8) & 0xFF);
  buf[4] = uint8_t((channel_mask >> 16) & 0xFF);
  buf[5] = uint8_t((channel_mask >> 24) & 0xFF);
}

Xma2PacketDecoder::Xma2PacketDecoder() = default;

Xma2PacketDecoder::~Xma2PacketDecoder() { Close(); }

int Xma2PacketDecoder::Open(int sample_rate, int total_channels,
                            int num_streams) {
  Close();

  const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_XMA2);
  if (!codec) {
    return AVERROR_DECODER_NOT_FOUND;
  }
  ctx_ = avcodec_alloc_context3(codec);
  if (!ctx_) {
    return AVERROR(ENOMEM);
  }

  ctx_->sample_rate = sample_rate;
  ctx_->block_align = int(kXma2PacketSize);

  const uint32_t mask = DefaultChannelMask(total_channels);
  if (mask) {
    av_channel_layout_from_mask(&ctx_->ch_layout, mask);
  } else {
    av_channel_layout_default(&ctx_->ch_layout, total_channels);
  }
  ctx_->request_sample_fmt = AV_SAMPLE_FMT_FLTP;

  ctx_->extradata = static_cast<uint8_t*>(
      av_mallocz(kXma2ExtradataSize + AV_INPUT_BUFFER_PADDING_SIZE));
  if (!ctx_->extradata) {
    Close();
    return AVERROR(ENOMEM);
  }
  ctx_->extradata_size = int(kXma2ExtradataSize);
  BuildXma2Extradata(ctx_->extradata, uint16_t(num_streams), mask);

  int ret = avcodec_open2(ctx_, codec, nullptr);
  if (ret < 0) {
    Close();
    return ret;
  }

  pkt_ = av_packet_alloc();
  frame_ = av_frame_alloc();
  if (!pkt_ || !frame_) {
    Close();
    return AVERROR(ENOMEM);
  }
  return 0;
}

void Xma2PacketDecoder::Close() {
  if (frame_) av_frame_free(&frame_);
  if (pkt_) av_packet_free(&pkt_);
  if (ctx_) avcodec_free_context(&ctx_);
}

int Xma2PacketDecoder::sample_rate() const {
  return ctx_ ? ctx_->sample_rate : 0;
}

int Xma2PacketDecoder::channels() const {
  return ctx_ ? ctx_->ch_layout.nb_channels : 0;
}

int Xma2PacketDecoder::SendPacket(const uint8_t* data, size_t size) {
  if (!ctx_ || !pkt_) return AVERROR(EINVAL);

  // FFmpeg requires AV_INPUT_BUFFER_PADDING_SIZE bytes of zeroed
  // memory past `size`. Use av_packet's owned buffer to satisfy that
  // without burdening callers.
  av_packet_unref(pkt_);
  int ret = av_new_packet(pkt_, int(size));
  if (ret < 0) return ret;
  std::memcpy(pkt_->data, data, size);
  std::memset(pkt_->data + size, 0, AV_INPUT_BUFFER_PADDING_SIZE);

  return avcodec_send_packet(ctx_, pkt_);
}

int Xma2PacketDecoder::SendEof() {
  if (!ctx_) return AVERROR(EINVAL);
  return avcodec_send_packet(ctx_, nullptr);
}

void Xma2PacketDecoder::Flush() {
  if (ctx_) {
    avcodec_flush_buffers(ctx_);
  }
  if (frame_) {
    av_frame_unref(frame_);
  }
  if (pkt_) {
    av_packet_unref(pkt_);
  }
}

int Xma2PacketDecoder::ReceiveFrame(Frame* out) {
  if (!ctx_ || !frame_) return AVERROR(EINVAL);
  av_frame_unref(frame_);
  int ret = avcodec_receive_frame(ctx_, frame_);
  if (ret < 0) return ret;
  out->planar = reinterpret_cast<const float* const*>(frame_->extended_data);
  out->nb_samples = frame_->nb_samples;
  out->channels = ctx_->ch_layout.nb_channels;
  out->sample_rate = ctx_->sample_rate;
  return 0;
}

}  // namespace tools
}  // namespace xe
