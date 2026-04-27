/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// xma-replay: feed XMA bitstream frames through xenia's vendored FFmpeg
// (third_party/FFmpeg) and write decoded PCM to a WAV file. Used to test
// changes to the XMA decode path offline without booting a game.
//
// Default per-stream RIFF path (DecodeStream) mirrors XmaContextV3's
// frame-walk and AVPacket assembly: same GetPacketInfo (trailing-bit
// walk + XMA2 frame_count fallback), same cross-packet split-frame
// handling, same [padding_start][frame_bits][padding_end] layout fed
// to AV_CODEC_ID_XMAFRAMES. Replay output should match the runtime's
// per-context output for the same packets.
//
// Inputs (auto-detected):
//
//   1. RIFF/WAVE container (.xma) — sample rate / channel count come
//      from the file's `fmt ` / `XMA2` chunk; CLI -r / -c are ignored.
//      With --codec xma2 the target stream's packets are fed through
//      Xma2PacketDecoder (single-stream AV_CODEC_ID_XMA2). With
//      -S all the full file is fed to AV_CODEC_ID_XMA2 (vgmstream-
//      style).
//
//   2. .manifest from XmaFrameDumper (per-frame runtime capture) —
//      replays each context's stream through its own
//      AV_CODEC_ID_XMAFRAMES decoder; produces one <out>_ctx<id>.wav
//      per context.
//
// Output is RIFF/WAVE int16 PCM at the codec's reported sample rate /
// channel count.

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "xenia/apu/xma_frame_walker.h"
#include "xenia/apu/xma_helpers.h"
#include "xenia/base/bit_stream.h"
#include "xenia/base/console_app_main.h"
#include "xenia/tools/xma_replay/xma2_packet_decoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/samplefmt.h>
}

namespace {

struct Args {
  std::string input_path;
  std::string output_path;
  std::string compare_path;  // reference WAV to diff against (post-decode)
  int sample_rate = 48000;
  int channels = 2;
  int verbose = 0;
  // RIFF/multi-stream: which stream to decode. -1 means "all streams,
  // interleaved" (3 stereo streams → one 6-ch WAV).
  int stream = 0;
  uint64_t skip_frames = 0;
  uint64_t max_frames = UINT64_MAX;
  // Single-stream codec selection (only applies to -S <n>, not -S all):
  //   xmaframes  : extract per-frame, feed AV_CODEC_ID_XMAFRAMES (default)
  //   xma2       : demux per-stream packets, feed AV_CODEC_ID_XMA2 with
  //                num_streams=1
  std::string codec = "xmaframes";
};

void usage() {
  std::fprintf(
      stderr,
      "usage: xma-replay [options] <input> <output.wav>\n"
      "\n"
      "Inputs (auto-detected):\n"
      "  RIFF/WAVE (.xma)      : decoded via xenia's frame-walk\n"
      "                          (per-stream XMAFRAMES or full-file XMA2)\n"
      "  .manifest             : runtime capture from XmaFrameDumper —\n"
      "                          replays each context's stream through its\n"
      "                          own decoder; writes one <out>_ctx<id>.wav\n"
      "                          per context at the recorded rate/channels\n"
      "                          (the .xframes file lives next to it).\n"
      "\n"
      "options:\n"
      "  -S, --stream <n|all>  which stream to decode for multi-stream\n"
      "                        XMA2 files, default 0. \"all\" interleaves\n"
      "                        every stream into one multi-channel WAV.\n"
      "      --codec <name>    single-stream codec: xmaframes (default)\n"
      "                        or xma2 (AV_CODEC_ID_XMA2, num_streams=1).\n"
      "  -s, --skip <n>        skip first N input frames (RIFF only)\n"
      "  -n, --max <n>         decode at most N input frames (RIFF only)\n"
      "      --compare <wav>   after decoding, diff output against this\n"
      "                        reference WAV (per-channel RMS, first diff)\n"
      "  -v, --verbose         per-frame logging to stderr\n"
      "  -h, --help            this help\n");
}

int parse_uint(const char* s, uint64_t* out) {
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0') return -1;
  *out = static_cast<uint64_t>(v);
  return 0;
}

int parse_args(const std::vector<std::string>& argv, Args* a) {
  std::vector<std::string> positional;
  // argv[0] is the program name (xenia's ParseWin32LaunchArguments
  // includes it). Skip it.
  for (size_t i = 1; i < argv.size(); ++i) {
    const std::string& arg = argv[i];
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argv.size()) {
        std::fprintf(stderr, "error: %s requires a value\n", name);
        return nullptr;
      }
      return argv[++i].c_str();
    };
    if (arg == "-h" || arg == "--help") {
      usage();
      std::exit(0);
    } else if (arg == "-v" || arg == "--verbose") {
      a->verbose = 1;
    } else if (arg == "-r" || arg == "--rate") {
      const char* v = need_value("--rate");
      if (!v) return -1;
      a->sample_rate = std::atoi(v);
    } else if (arg == "-c" || arg == "--channels") {
      const char* v = need_value("--channels");
      if (!v) return -1;
      a->channels = std::atoi(v);
    } else if (arg == "-S" || arg == "--stream") {
      const char* v = need_value("--stream");
      if (!v) return -1;
      if (std::strcmp(v, "all") == 0) {
        a->stream = -1;
      } else {
        a->stream = std::atoi(v);
      }
    } else if (arg == "-s" || arg == "--skip") {
      const char* v = need_value("--skip");
      if (!v || parse_uint(v, &a->skip_frames) < 0) return -1;
    } else if (arg == "-n" || arg == "--max") {
      const char* v = need_value("--max");
      if (!v || parse_uint(v, &a->max_frames) < 0) return -1;
    } else if (arg == "--compare") {
      const char* v = need_value("--compare");
      if (!v) return -1;
      a->compare_path = v;
    } else if (arg == "--codec") {
      const char* v = need_value("--codec");
      if (!v) return -1;
      a->codec = v;
      if (a->codec != "xmaframes" && a->codec != "xma2") {
        std::fprintf(stderr, "error: --codec must be 'xmaframes' or 'xma2'\n");
        return -1;
      }
    } else if (!arg.empty() && arg[0] == '-') {
      std::fprintf(stderr, "error: unknown option %s\n", arg.c_str());
      return -1;
    } else {
      positional.push_back(arg);
    }
  }
  if (positional.size() != 2) {
    usage();
    return -1;
  }
  a->input_path = positional[0];
  a->output_path = positional[1];
  if (a->sample_rate <= 0 || a->channels <= 0 || a->channels > 8) {
    std::fprintf(stderr, "error: invalid sample_rate or channels\n");
    return -1;
  }
  return 0;
}

// Minimal RIFF/WAVE writer for 16-bit signed little-endian interleaved PCM.
class WavWriter {
 public:
  bool Open(const std::string& path, int sample_rate, int channels) {
    Close();
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) return false;
    sample_rate_ = sample_rate;
    channels_ = channels;
    samples_written_ = 0;
    WriteHeader(0);
    return true;
  }

  void WriteInterleavedS16(const int16_t* data, size_t frames) {
    if (!f_) return;
    std::fwrite(data, sizeof(int16_t), frames * channels_, f_);
    samples_written_ += frames;
  }

  void Close() {
    if (!f_) return;
    std::fseek(f_, 0, SEEK_SET);
    WriteHeader(samples_written_);
    std::fclose(f_);
    f_ = nullptr;
  }

  uint64_t samples_written() const { return samples_written_; }

 private:
  static void Write32(FILE* f, uint32_t v) {
    uint8_t b[4] = {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8),
                    static_cast<uint8_t>(v >> 16),
                    static_cast<uint8_t>(v >> 24)};
    std::fwrite(b, 1, 4, f);
  }
  static void Write16(FILE* f, uint16_t v) {
    uint8_t b[2] = {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8)};
    std::fwrite(b, 1, 2, f);
  }

  void WriteHeader(uint64_t total_samples) {
    const uint32_t data_bytes =
        static_cast<uint32_t>(total_samples * channels_ * sizeof(int16_t));
    const uint32_t fmt_chunk_size = 16;
    const uint32_t riff_size = 4 + (8 + fmt_chunk_size) + (8 + data_bytes);

    std::fwrite("RIFF", 1, 4, f_);
    Write32(f_, riff_size);
    std::fwrite("WAVE", 1, 4, f_);

    std::fwrite("fmt ", 1, 4, f_);
    Write32(f_, fmt_chunk_size);
    Write16(f_, 1);             // PCM format
    Write16(f_, channels_);     // num channels
    Write32(f_, sample_rate_);  // sample rate
    Write32(f_, sample_rate_ * channels_ * sizeof(int16_t));
    Write16(f_, channels_ * sizeof(int16_t));
    Write16(f_, 16);

    std::fwrite("data", 1, 4, f_);
    Write32(f_, data_bytes);
  }

  FILE* f_ = nullptr;
  int sample_rate_ = 0;
  int channels_ = 0;
  uint64_t samples_written_ = 0;
};

// Write a planar-float frame view (as produced by Xma2PacketDecoder)
// as int16 PCM into `wav`.
size_t WritePlanarFloatToWav(WavWriter& wav, const float* const* planar,
                             int nb_samples, int channels, int wav_channels) {
  if (nb_samples <= 0) return 0;
  std::vector<int16_t> interleaved(
      static_cast<size_t>(nb_samples) * wav_channels, 0);
  const int copy_channels = std::min(channels, wav_channels);
  for (int s = 0; s < nb_samples; ++s) {
    for (int c = 0; c < copy_channels; ++c) {
      float v = planar[c][s];
      if (v > 1.0f)
        v = 1.0f;
      else if (v < -1.0f)
        v = -1.0f;
      interleaved[s * wav_channels + c] = static_cast<int16_t>(v * 32767.0f);
    }
  }
  wav.WriteInterleavedS16(interleaved.data(), static_cast<size_t>(nb_samples));
  return static_cast<size_t>(nb_samples);
}

size_t WriteFrameToWav(WavWriter& wav, AVFrame* frame, int wav_channels) {
  const int n = frame->nb_samples;
  const int src_channels = frame->ch_layout.nb_channels;
  const int out_channels = wav_channels;
  if (n <= 0) return 0;

  std::vector<int16_t> interleaved(static_cast<size_t>(n) * out_channels, 0);

  if (frame->format == AV_SAMPLE_FMT_FLTP) {
    const int copy_channels = std::min(src_channels, out_channels);
    for (int s = 0; s < n; ++s) {
      for (int c = 0; c < copy_channels; ++c) {
        const float* src = reinterpret_cast<const float*>(frame->data[c]);
        float v = src[s];
        if (v > 1.0f)
          v = 1.0f;
        else if (v < -1.0f)
          v = -1.0f;
        interleaved[s * out_channels + c] = static_cast<int16_t>(v * 32767.0f);
      }
    }
  } else if (frame->format == AV_SAMPLE_FMT_S16) {
    const int16_t* src = reinterpret_cast<const int16_t*>(frame->data[0]);
    const int copy_channels = std::min(src_channels, out_channels);
    for (int s = 0; s < n; ++s) {
      for (int c = 0; c < copy_channels; ++c) {
        interleaved[s * out_channels + c] = src[s * src_channels + c];
      }
    }
  }

  wav.WriteInterleavedS16(interleaved.data(), static_cast<size_t>(n));
  return static_cast<size_t>(n);
}

const char* AvErr(int err) {
  static thread_local char buf[256];
  av_strerror(err, buf, sizeof(buf));
  return buf;
}

// Open and configure the AV_CODEC_ID_XMAFRAMES decoder.
AVCodecContext* OpenXmaCodec(int sample_rate, int channels) {
  const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_XMAFRAMES);
  if (!codec) {
    std::fprintf(stderr,
                 "error: AV_CODEC_ID_XMAFRAMES decoder not available\n");
    return nullptr;
  }
  AVCodecContext* ctx = avcodec_alloc_context3(codec);
  if (!ctx) return nullptr;
  ctx->sample_rate = sample_rate;
  av_channel_layout_default(&ctx->ch_layout, channels);
  ctx->request_sample_fmt = AV_SAMPLE_FMT_FLTP;
  int ret = avcodec_open2(ctx, codec, nullptr);
  if (ret < 0) {
    std::fprintf(stderr, "error: avcodec_open2 failed: %s\n", AvErr(ret));
    avcodec_free_context(&ctx);
    return nullptr;
  }
  return ctx;
}

// Build the avpacket the same way XmaContextNew::PreparePacket does:
// [padding_byte][frame_bytes][trailing_padding]. xma_frame_buf must be
// sized to hold 1 + ceil((frame_size_bits + padding_start)/8) bytes.
void BuildXmaFramePacket(uint8_t* xma_frame_buf, int xma_frame_buf_size,
                         AVPacket* pkt, uint32_t frame_size_bits,
                         uint32_t padding_start) {
  pkt->data = xma_frame_buf;
  pkt->size =
      static_cast<int>(1 + ((padding_start + frame_size_bits) / 8) +
                       (((padding_start + frame_size_bits) % 8) ? 1 : 0));
  if (pkt->size > xma_frame_buf_size) pkt->size = xma_frame_buf_size;
  uint32_t padding_end = pkt->size * 8 - (8 + padding_start + frame_size_bits);
  xma_frame_buf[0] = static_cast<uint8_t>(((padding_start & 7) << 5) |
                                          ((padding_end & 7) << 2));
}

// Send pkt to ctx, drain frames into wav. Side-effect updates stats.
struct DecodeStats {
  uint64_t input_frames = 0;
  uint64_t decode_ok = 0;    // avcodec_send_packet returned >= 0
  uint64_t decode_fail = 0;  // avcodec_send_packet returned < 0
  uint64_t frames_out = 0;   // frames produced by avcodec_receive_frame
  // input_frames - frames_out = frames the decoder accepted but produced
  // no output for (silent skip due to internal warmup, mid-stream errors,
  // bit-misalignment recovery, etc.)
};

void FeedAndDrain(AVCodecContext* ctx, AVPacket* pkt, AVFrame* frame,
                  WavWriter& wav, int wav_channels, bool verbose,
                  uint64_t input_idx, int diag_size_bytes, DecodeStats* stats) {
  int ret = avcodec_send_packet(ctx, pkt);
  if (ret < 0) {
    ++stats->decode_fail;
    if (verbose) {
      std::fprintf(stderr, "frame %" PRIu64 " (size %d): send_packet: %s\n",
                   input_idx, diag_size_bytes, AvErr(ret));
    }
  } else {
    ++stats->decode_ok;
  }
  while ((ret = avcodec_receive_frame(ctx, frame)) >= 0) {
    WriteFrameToWav(wav, frame, wav_channels);
    ++stats->frames_out;
    if (verbose) {
      std::fprintf(stderr,
                   "frame %" PRIu64 " (in_size %d): %d samples * %d ch\n",
                   input_idx, diag_size_bytes, frame->nb_samples,
                   frame->ch_layout.nb_channels);
    }
    av_frame_unref(frame);
  }
}

bool DetectRiff(const std::vector<uint8_t>& bytes) {
  return bytes.size() >= 4 && bytes[0] == 'R' && bytes[1] == 'I' &&
         bytes[2] == 'F' && bytes[3] == 'F';
}

bool ReadWholeFile(const std::string& path, std::vector<uint8_t>* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (sz < 0) {
    std::fclose(f);
    return false;
  }
  out->resize(static_cast<size_t>(sz));
  size_t got = std::fread(out->data(), 1, out->size(), f);
  std::fclose(f);
  return got == out->size();
}

// Parse a RIFF/WAVE XMA2 file and locate the codec params + data chunk.
// Parsed RIFF/XMA2 metadata. Source: GDK's xma2defs.h.
//
// "fmt " chunk holds an XMA2WAVEFORMATEX (52 bytes, little-endian)
// when wFormatTag == 0x0166. "XMA2" chunk holds an XMA2WAVEFORMAT
// (legacy, big-endian DWORDs).
struct StreamFmt {
  uint8_t channels;
};

struct WaveInfo {
  uint16_t codec_id = 0;  // 0x0166 = WAVE_FORMAT_XMA2
  uint8_t num_streams = 0;
  uint16_t total_channels = 0;  // sum of per-stream channels
  uint32_t sample_rate = 0;
  uint32_t block_size_bytes = 0;
  uint32_t block_count = 0;
  StreamFmt streams[8] = {};
  size_t data_offset = 0;
  size_t data_size = 0;
};

static uint32_t ReadBE32(const uint8_t* p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
         (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}
static uint16_t ReadBE16(const uint8_t* p) {
  return (uint16_t(p[0]) << 8) | uint16_t(p[1]);
}
static uint32_t ReadLE32(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}
static uint16_t ReadLE16(const uint8_t* p) {
  return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

// Parse a legacy "XMA2" chunk per xma2defs.h XMA2WAVEFORMAT (big-endian).
bool ParseXMA2Chunk(const uint8_t* p, size_t size, WaveInfo* info) {
  if (size < 40) return false;
  // p[0]: Version, p[1]: NumStreams, p[2]: RESERVED, p[3]: LoopCount.
  info->num_streams = p[1];
  if (info->num_streams == 0 || info->num_streams > 8) return false;
  info->sample_rate = ReadBE32(p + 12);
  info->block_size_bytes = ReadBE32(p + 24);
  info->block_count = ReadBE32(p + 36);
  if (size < 40u + 4u * info->num_streams) return false;
  uint16_t total = 0;
  for (uint8_t i = 0; i < info->num_streams; ++i) {
    info->streams[i].channels = p[40 + 4 * i];
    total += info->streams[i].channels;
  }
  info->total_channels = total;
  info->codec_id = 0x0166;  // WAVE_FORMAT_XMA2
  return true;
}

// Parse a "fmt " chunk per XMA2WAVEFORMATEX (little-endian).
bool ParseFmtChunk(const uint8_t* p, size_t size, WaveInfo* info) {
  if (size < 18) return false;
  info->codec_id = ReadLE16(p);
  if (info->codec_id != 0x0166) return false;  // not XMA2
  info->total_channels = ReadLE16(p + 2);
  info->sample_rate = ReadLE32(p + 4);
  if (size < 52) return false;  // need cbSize=34 extension
  info->num_streams = ReadLE16(p + 18);
  info->block_size_bytes = ReadLE32(p + 28);
  // XMA2WAVEFORMATEX doesn't enumerate per-stream channel breakdowns,
  // so we approximate: fan total_channels across num_streams as
  // 2-2-2-... with mono trailer if odd. Matches the standard XMA2
  // multichannel encoding convention.
  if (info->num_streams == 0 || info->num_streams > 8) return false;
  uint16_t remaining = info->total_channels;
  for (uint8_t i = 0; i < info->num_streams; ++i) {
    uint8_t ch = (remaining >= 2) ? 2 : remaining;
    info->streams[i].channels = ch;
    remaining -= ch;
  }
  if (remaining != 0) return false;  // mismatch between total and streams
  return true;
}

bool ParseRiffWave(const std::vector<uint8_t>& bytes, WaveInfo* info) {
  if (bytes.size() < 12) return false;
  if (std::memcmp(bytes.data(), "RIFF", 4) != 0) return false;
  if (std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) return false;

  bool got_fmt = false;
  size_t pos = 12;
  while (pos + 8 <= bytes.size()) {
    char id[5] = {0};
    std::memcpy(id, bytes.data() + pos, 4);
    const uint32_t size = ReadLE32(bytes.data() + pos + 4);
    pos += 8;
    if (pos + size > bytes.size()) return false;
    if (std::memcmp(id, "fmt ", 4) == 0) {
      got_fmt = ParseFmtChunk(bytes.data() + pos, size, info) || got_fmt;
    } else if (std::memcmp(id, "XMA2", 4) == 0) {
      // Don't override a fully-parsed fmt chunk; otherwise prefer XMA2
      // since it carries authoritative per-stream channel breakdown.
      if (!got_fmt) {
        ParseXMA2Chunk(bytes.data() + pos, size, info);
      }
    } else if (std::memcmp(id, "data", 4) == 0) {
      info->data_offset = pos;
      info->data_size = size;
      return info->codec_id == 0x0166;
    }
    pos += (size + 1) & ~1u;  // pad to even
  }
  return false;
}

// Decode one stream from a RIFF/XMA2 file using the same frame-extraction
// logic as XmaContextV3 (apu/xma_context_v3.cc). Mirrors V3's runtime
// state machine so xma-replay exercises the same packet-walk + frame-
// extraction code path the in-emulator decoder takes:
//
//   - GetPacketInfo (trailing-bit walk first, XMA2 packet[0]>>2 as
//     fallback when the walk under-counts — V3's hybrid)
//   - cross-packet split-frame handling: combine current packet payload
//     with the next packet's payload, re-Peek the 15-bit frame size
//   - [padding_start][frame_bits][padding_end] layout for AV_CODEC_ID_XMAFRAMES
//
// Strategy: demux the target stream's packets into a contiguous buffer
// (zeroing each PacketSkipCount byte so consecutive entries are walked
// one packet at a time), then run V3's walker over that buffer with a
// single read_offset (no current_buffer dual-buffer state — replay only
// has one stream of packets).
//
// XMA standard: every input frame decodes to exactly this many samples
// per channel.
static constexpr int kSamplesPerXmaFrame = 512;

template <typename OnFrameFn, typename OnSilenceFn>
int DecodeStream(const Args& args, const std::vector<uint8_t>& bytes,
                 const WaveInfo& info, uint8_t stream_idx, DecodeStats* stats,
                 OnFrameFn on_frame, OnSilenceFn on_silence) {
  const uint8_t stream_channels = info.streams[stream_idx].channels;
  if (stream_channels < 1 || stream_channels > 2) {
    std::fprintf(stderr, "error: stream %u has %u channels (must be 1 or 2)\n",
                 stream_idx, stream_channels);
    return 1;
  }

  AVCodecContext* ctx = OpenXmaCodec(info.sample_rate, stream_channels);
  if (!ctx) return 1;
  AVPacket* pkt = av_packet_alloc();
  AVFrame* frame = av_frame_alloc();

  using namespace xe::apu;
  static constexpr uint32_t kBytesPerPacket = xma::kBytesPerPacket;
  static constexpr uint32_t kBitsPerPacket = xma::kBitsPerPacket;
  static constexpr uint32_t kBitsPerPacketHeader = xma::kBitsPerPacketHeader;
  static constexpr uint32_t kMaxFrameSizeinBits = xma::kMaxFrameSizeinBits;

  const size_t packet_count = info.data_size / kBytesPerPacket;

  // Extract this stream's packets into a contiguous buffer. Walk packets
  // via PacketSkipCount + 1 (xma2defs.h §"all frames beginning in
  // packet 0 belong to stream 0"). The first NumStreams packets are in
  // stream order, so packet[stream_idx] is the first one for us.
  // PacketSkipCount of each copy is zeroed so the V3 walker, which
  // reads packet[3] as `skip_count`, advances to the immediate next
  // entry in stream_data instead of jumping back into multi-stream space.
  std::vector<uint8_t> stream_data;
  stream_data.reserve(packet_count * kBytesPerPacket);
  size_t pkt_i = stream_idx;
  while (pkt_i < packet_count) {
    const uint8_t* p =
        bytes.data() + info.data_offset + pkt_i * kBytesPerPacket;
    const uint8_t skip = xma::GetPacketSkipCount(p);
    if (skip == 0xFF) {
      pkt_i += 1;
      continue;
    }
    const size_t off = stream_data.size();
    stream_data.resize(off + kBytesPerPacket);
    std::memcpy(stream_data.data() + off, p, kBytesPerPacket);
    stream_data[off + 3] = 0;
    pkt_i += size_t(skip) + 1;
  }

  if (stream_data.empty()) {
    std::fprintf(stderr, "warning: stream %u has no decodable packets\n",
                 stream_idx);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&ctx);
    return 0;
  }

  // AVPacket payload buffer, identically sized to XmaContextV3::xma_frame_.
  std::vector<uint8_t> xma_frame(1 + 4096, 0);

  auto feed_and_drain = [&](uint64_t idx, uint32_t frame_size_bits) {
    int ret = avcodec_send_packet(ctx, pkt);
    if (ret < 0) {
      ++stats->decode_fail;
      if (args.verbose) {
        std::fprintf(stderr,
                     "stream %u frame %" PRIu64
                     " (size %u bits): "
                     "send_packet: %s\n",
                     stream_idx, idx, frame_size_bits, AvErr(ret));
      }
    } else {
      ++stats->decode_ok;
    }
    int got = 0;
    while ((ret = avcodec_receive_frame(ctx, frame)) >= 0) {
      on_frame(frame);
      ++stats->frames_out;
      ++got;
      av_frame_unref(frame);
    }
    if (got == 0) {
      // Packet accepted but produced no output (warmup / mid-stream
      // recovery). Emit one frame of silence so per-stream timeline
      // matches input frame count, exactly as before.
      on_silence(kSamplesPerXmaFrame);
    }
  };

  const uint32_t stream_bytes = static_cast<uint32_t>(stream_data.size());
  const uint32_t stream_packet_count = stream_bytes / kBytesPerPacket;

  // V3-style walker: track input_buffer_read_offset (bits from start of
  // stream_data). Single contiguous buffer, no SwapInputBuffer.
  uint32_t read_offset = kBitsPerPacketHeader;
  uint64_t input_idx = 0;

  while (true) {
    if (input_idx - args.skip_frames >= args.max_frames &&
        input_idx >= args.skip_frames)
      break;

    const uint32_t packet_index = read_offset / kBitsPerPacket;
    if (packet_index >= stream_packet_count) break;

    uint8_t* packet = stream_data.data() + packet_index * kBytesPerPacket;
    const uint32_t packet_first_frame_offset =
        xma::GetPacketFrameOffset(packet);

    // skip_corrupt-style fast-forward to the next packet that has a valid
    // frame offset — V3 does this via skip_corrupt_packet().
    if (packet_first_frame_offset > kMaxFrameSizeinBits) {
      const uint32_t next = packet_index + 1;
      if (next >= stream_packet_count) break;
      read_offset = next * kBitsPerPacket +
                    xma::GetPacketFrameOffset(stream_data.data() +
                                              next * kBytesPerPacket);
      if ((read_offset % kBitsPerPacket) > kMaxFrameSizeinBits) break;
      continue;
    }

    uint32_t relative_offset = read_offset % kBitsPerPacket;
    if (relative_offset < packet_first_frame_offset) {
      read_offset = packet_index * kBitsPerPacket + packet_first_frame_offset;
      relative_offset = packet_first_frame_offset;
    }

    const uint8_t skip_count = xma::GetPacketSkipCount(packet);  // 0 (zeroed)
    if (skip_count == 0xFF) {
      read_offset = (packet_index + 1) * kBitsPerPacket + kBitsPerPacketHeader;
      continue;
    }

    xma::PacketWalk packet_info = xma::InspectPacket(packet, relative_offset);
    const uint32_t next_packet_index = packet_index + skip_count + 1;
    const uint8_t* next_packet =
        next_packet_index < stream_packet_count
            ? stream_data.data() + next_packet_index * kBytesPerPacket
            : nullptr;

    // Cross-packet split frame: header straddles packet boundary,
    // walker reported frame_size == 0.
    if (packet_info.current_frame_size == 0) {
      if (!next_packet) break;
      packet_info.current_frame_size =
          xma::ResolveSplitFrameSize(packet, next_packet, relative_offset);
      if (packet_info.current_frame_size == 0) break;
    }

    // Frame must fit within the combined current+next payload window.
    const uint32_t combined_payload_bits = xma::kBytesPerPacketData * 2 * 8;
    const uint32_t combined_relative_offset =
        relative_offset - kBitsPerPacketHeader;
    if (combined_relative_offset > combined_payload_bits ||
        packet_info.current_frame_size >
            (combined_payload_bits - combined_relative_offset)) {
      // Out-of-range frame size: fast-forward to next packet (matches
      // XmaContextV3::skip_corrupt_packet).
      const uint32_t next = packet_index + 1;
      if (next >= stream_packet_count) break;
      read_offset = next * kBitsPerPacket +
                    xma::GetPacketFrameOffset(stream_data.data() +
                                              next * kBytesPerPacket);
      continue;
    }

    // Build the AVPacket payload via shared helper - byte-identical to
    // what XmaContextV3::Decode hands to AV_CODEC_ID_XMAFRAMES.
    const int avpacket_size = xma::BuildAvPacketPayload(
        packet, next_packet, relative_offset, packet_info.current_frame_size,
        xma_frame.data(), static_cast<int>(xma_frame.size()));
    if (avpacket_size <= 0) break;

    if (input_idx >= args.skip_frames) {
      ++stats->input_frames;
      pkt->data = xma_frame.data();
      pkt->size = avpacket_size;
      feed_and_drain(input_idx, packet_info.current_frame_size);
    }
    ++input_idx;

    // Advance read_offset.
    if (!packet_info.IsLastFrameInPacket()) {
      const uint32_t next_frame_offset =
          (read_offset + packet_info.current_frame_size) % kBitsPerPacket;
      read_offset = packet_index * kBitsPerPacket + next_frame_offset;
    } else {
      // Jump to next packet's first frame.
      if (next_packet_index >= stream_packet_count) break;
      const uint32_t next_first = xma::GetPacketFrameOffset(
          stream_data.data() + next_packet_index * kBytesPerPacket);
      if (next_first > kMaxFrameSizeinBits) break;
      read_offset = next_packet_index * kBitsPerPacket + next_first;
    }
  }

  // Drain.
  avcodec_send_packet(ctx, nullptr);
  while (avcodec_receive_frame(ctx, frame) >= 0) {
    on_frame(frame);
    ++stats->frames_out;
    av_frame_unref(frame);
  }
  av_frame_free(&frame);
  av_packet_free(&pkt);
  avcodec_free_context(&ctx);
  return 0;
}

// Walk the disk packets the same way wmaprodec.c's xma_decode_packet
// scheduler does, returning the disk-packet indices that belong to
// `target_stream`.  This lets us feed a single-stream Xma2PacketDecoder
// the same packet sequence the multi-stream codec would route to that
// stream — one stream per context, the shape the runtime sees.
//
// Scheduler state:
//   - skip_packets[i] decremented each disk packet
//   - on schedule, current_stream = stream with min skip_packets
//   - after processing, skip_packets[current] = packet header skip-count
std::vector<size_t> DemuxStreamPackets(const std::vector<uint8_t>& bytes,
                                       const WaveInfo& info,
                                       uint8_t target_stream) {
  using namespace xe::apu;
  static constexpr uint32_t kBytesPerPacket = xma::kBytesPerPacket;
  std::vector<size_t> indices;
  if (target_stream >= info.num_streams) return indices;
  const size_t packet_count = info.data_size / kBytesPerPacket;

  std::array<int, 8> skip_packets = {};
  uint8_t current_stream = 0;
  bool first_call = true;

  for (size_t pkt_i = 0; pkt_i < packet_count; ++pkt_i) {
    if (!first_call && skip_packets[current_stream] != 0) {
      // Find stream with min skip_packets (ties → smallest index).
      int min_val = skip_packets[0];
      uint8_t min_idx = 0;
      for (uint8_t i = 1; i < info.num_streams; ++i) {
        if (skip_packets[i] < min_val) {
          min_val = skip_packets[i];
          min_idx = i;
        }
      }
      current_stream = min_idx;
    }
    if (!first_call) {
      for (uint8_t i = 0; i < info.num_streams; ++i) {
        if (skip_packets[i] > 0) skip_packets[i]--;
      }
    }
    first_call = false;

    if (current_stream == target_stream) {
      indices.push_back(pkt_i);
    }

    const uint8_t* packet =
        bytes.data() + info.data_offset + pkt_i * kBytesPerPacket;
    skip_packets[current_stream] = xma::GetPacketSkipCount(packet);
  }
  return indices;
}

// Single-stream decode via AV_CODEC_ID_XMA2 with num_streams=1.
// Demuxes the target stream's packets out of the multi-stream input via
// the scheduler simulation, then feeds them sequentially to
// Xma2PacketDecoder and drains output.
int RunRiffSingleStreamXma2(const Args& args, const std::vector<uint8_t>& bytes,
                            const WaveInfo& info, WavWriter* wav,
                            DecodeStats* stats, int* out_sample_rate,
                            int* out_channels) {
  *out_sample_rate = info.sample_rate;
  if (args.stream < 0 || args.stream >= info.num_streams) {
    std::fprintf(stderr,
                 "error: --stream %d out of range (file has %u streams)\n",
                 args.stream, info.num_streams);
    return 1;
  }
  const uint8_t stream_idx = static_cast<uint8_t>(args.stream);
  const uint8_t stream_channels = info.streams[stream_idx].channels;
  *out_channels = stream_channels;

  wav->Close();
  if (!wav->Open(args.output_path, info.sample_rate, stream_channels)) {
    std::fprintf(stderr, "error: cannot open output\n");
    return 1;
  }

  const std::vector<size_t> packet_indices =
      DemuxStreamPackets(bytes, info, stream_idx);
  std::fprintf(stderr,
               "xma-replay: RIFF input, %u Hz, stream %u (%u ch) — "
               "demuxed %zu packets via scheduler simulation, decoding via "
               "AV_CODEC_ID_XMA2 (num_streams=1)\n",
               info.sample_rate, stream_idx, stream_channels,
               packet_indices.size());

  using namespace xe::apu;
  xe::tools::Xma2PacketDecoder decoder;
  int ret = decoder.Open(int(info.sample_rate), int(stream_channels),
                         /*num_streams=*/1);
  if (ret < 0) {
    std::fprintf(stderr, "error: Xma2PacketDecoder::Open failed: %s\n",
                 AvErr(ret));
    return 1;
  }

  static constexpr uint32_t kBytesPerPacket = xma::kBytesPerPacket;
  uint64_t input_pkts = 0;
  uint64_t decode_ok = 0;
  uint64_t decode_fail = 0;
  uint64_t frames_out = 0;

  auto drain = [&]() {
    xe::tools::Xma2PacketDecoder::Frame f{};
    while (decoder.ReceiveFrame(&f) == 0) {
      WritePlanarFloatToWav(*wav, f.planar, f.nb_samples, f.channels,
                            stream_channels);
      ++frames_out;
    }
  };

  for (size_t pkt_i : packet_indices) {
    if (input_pkts - args.skip_frames >= args.max_frames &&
        input_pkts >= args.skip_frames)
      break;
    if (input_pkts < args.skip_frames) {
      ++input_pkts;
      continue;
    }
    const uint8_t* p =
        bytes.data() + info.data_offset + pkt_i * kBytesPerPacket;
    int sret = decoder.SendPacket(p, kBytesPerPacket);
    if (sret < 0) {
      ++decode_fail;
      if (args.verbose) {
        std::fprintf(stderr, "packet %zu: send_packet: %s\n", pkt_i,
                     AvErr(sret));
      }
    } else {
      ++decode_ok;
    }
    drain();
    ++input_pkts;
  }
  decoder.SendEof();
  drain();

  stats->input_frames += input_pkts;
  stats->decode_ok += decode_ok;
  stats->decode_fail += decode_fail;
  stats->frames_out += frames_out;
  return 0;
}

int RunRiff(const Args& args, const std::vector<uint8_t>& bytes,
            const WaveInfo& info, WavWriter* wav, DecodeStats* stats,
            int* out_sample_rate, int* out_channels) {
  *out_sample_rate = info.sample_rate;

  if (args.stream < 0 || args.stream >= info.num_streams) {
    std::fprintf(stderr,
                 "error: --stream %d out of range (file has %u streams)\n",
                 args.stream, info.num_streams);
    return 1;
  }
  const uint8_t stream_idx = static_cast<uint8_t>(args.stream);
  const uint8_t stream_channels = info.streams[stream_idx].channels;

  *out_channels = stream_channels;

  // (Re-)open WAV at codec's actual rate / channel count.
  wav->Close();
  if (!wav->Open(args.output_path, info.sample_rate, stream_channels)) {
    std::fprintf(stderr, "error: cannot open output\n");
    return 1;
  }
  std::fprintf(stderr,
               "xma-replay: RIFF input, codec_id=0x%04X, %u Hz, "
               "num_streams=%u, total_channels=%u, block_size=%u, "
               "block_count=%u; selected stream %u (%u ch)\n",
               info.codec_id, info.sample_rate, info.num_streams,
               info.total_channels, info.block_size_bytes, info.block_count,
               stream_idx, stream_channels);

  return DecodeStream(
      args, bytes, info, stream_idx, stats,
      [&](AVFrame* frame) { WriteFrameToWav(*wav, frame, stream_channels); },
      [&](int n_samples) {
        // Pad missed frames with silence so per-stream
        // sample timeline matches input frame timeline.
        std::vector<int16_t> silence(size_t(n_samples) * stream_channels, 0);
        wav->WriteInterleavedS16(silence.data(), size_t(n_samples));
      });
}

// Decode every stream in `info` independently into per-stream float
// PCM buffers, then interleave them into one multi-channel WAV.
// Channel order: stream 0's channels first, then stream 1, etc.
// vgmstream-style decode: feed the full data chunk's 2KB packets to
// FFmpeg's AV_CODEC_ID_XMA2 codec, which walks frames and demultiplexes
// streams internally. Output frames are already multi-channel
// interleaved (per the channel_mask in extradata), so no per-stream
// extraction or post-interleave step is needed.
int RunRiffAll(const Args& args, const std::vector<uint8_t>& bytes,
               const WaveInfo& info, WavWriter* wav, DecodeStats* stats,
               int* out_sample_rate, int* out_channels) {
  *out_sample_rate = info.sample_rate;
  *out_channels = info.total_channels;

  std::fprintf(stderr,
               "xma-replay: RIFF input, %u Hz, %u streams, %u total ch — "
               "decoding via AV_CODEC_ID_XMA2 (vgmstream-style 2KB packet "
               "feeding) into one %u-channel WAV\n",
               info.sample_rate, info.num_streams, info.total_channels,
               info.total_channels);

  using namespace xe::apu;
  xe::tools::Xma2PacketDecoder decoder;
  int ret = decoder.Open(int(info.sample_rate), int(info.total_channels),
                         int(info.num_streams));
  if (ret < 0) {
    std::fprintf(stderr, "error: Xma2PacketDecoder::Open failed: %s\n",
                 AvErr(ret));
    return 1;
  }

  // Re-open WAV at codec's actual channel count.
  wav->Close();
  if (!wav->Open(args.output_path, info.sample_rate, info.total_channels)) {
    std::fprintf(stderr, "error: cannot open output\n");
    return 1;
  }

  static constexpr uint32_t kBytesPerPacket = xma::kBytesPerPacket;
  const size_t packet_count = info.data_size / kBytesPerPacket;

  uint64_t input_pkts = 0;
  uint64_t decode_ok = 0;
  uint64_t decode_fail = 0;
  uint64_t frames_out = 0;

  auto drain = [&]() {
    xe::tools::Xma2PacketDecoder::Frame f{};
    while (true) {
      int r = decoder.ReceiveFrame(&f);
      if (r < 0) break;
      WritePlanarFloatToWav(*wav, f.planar, f.nb_samples, f.channels,
                            info.total_channels);
      ++frames_out;
    }
  };

  for (size_t pkt_i = 0; pkt_i < packet_count; ++pkt_i) {
    if (input_pkts - args.skip_frames >= args.max_frames &&
        input_pkts >= args.skip_frames)
      break;
    const uint8_t* p =
        bytes.data() + info.data_offset + pkt_i * kBytesPerPacket;
    if (input_pkts < args.skip_frames) {
      ++input_pkts;
      continue;
    }

    int sret = decoder.SendPacket(p, kBytesPerPacket);
    if (sret < 0) {
      ++decode_fail;
      if (args.verbose) {
        std::fprintf(stderr, "packet %zu: send_packet: %s\n", pkt_i,
                     AvErr(sret));
      }
    } else {
      ++decode_ok;
    }
    drain();
    ++input_pkts;
  }

  decoder.SendEof();
  drain();

  // Stuff our local stats counters into the shared DecodeStats so the
  // summary line still works. (XMA2 codec produces multiple output
  // frames per input packet — the input/output ratio isn't 1:1, so
  // "missed" semantics differ from XMAFRAMES.)
  stats->input_frames += input_pkts;
  stats->decode_ok += decode_ok;
  stats->decode_fail += decode_fail;
  stats->frames_out += frames_out;
  return 0;
}

// One entry parsed from the .manifest TSV (per-packet capture).
//   seq context_id time_us offset size rate_hz channels buf_idx pkt_idx
//   read_off
struct ManifestRow {
  uint64_t seq;
  uint32_t context_id;
  uint64_t time_us;
  uint64_t offset;
  uint32_t size;
  uint32_t rate_hz;
  uint8_t channels;
  uint8_t buf_idx;
  uint16_t pkt_idx;
  uint32_t read_off_bits;
};

// Read the .manifest file (xma_<ts>.manifest produced by
// XmaFrameDumper). Header lines start with '#' and the column-name
// row starts with "seq\t". All other lines are TSV rows.
bool ParseManifest(const std::string& path, std::vector<ManifestRow>* rows) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "error: cannot open manifest %s\n", path.c_str());
    return false;
  }
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
    if (std::strncmp(line, "seq\t", 4) == 0) continue;
    ManifestRow r{};
    unsigned channels = 0;
    unsigned buf_idx = 0;
    unsigned pkt_idx = 0;
    unsigned read_off = 0;
    int n = std::sscanf(
        line, "%llu\t%u\t%llu\t%llu\t%u\t%u\t%u\t%u\t%u\t%u",
        reinterpret_cast<unsigned long long*>(&r.seq), &r.context_id,
        reinterpret_cast<unsigned long long*>(&r.time_us),
        reinterpret_cast<unsigned long long*>(&r.offset), &r.size, &r.rate_hz,
        &channels, &buf_idx, &pkt_idx, &read_off);
    if (n != 10) {
      std::fprintf(stderr, "warning: skipping malformed manifest row: %s",
                   line);
      continue;
    }
    r.channels = static_cast<uint8_t>(channels);
    r.buf_idx = static_cast<uint8_t>(buf_idx);
    r.pkt_idx = static_cast<uint16_t>(pkt_idx);
    r.read_off_bits = read_off;
    rows->push_back(r);
  }
  std::fclose(f);
  return true;
}

// Per-context decode state for manifest replay. Each captured frame
// is the AVPacket payload xma_context_new built (header byte + frame
// bits + padding), fed straight into AV_CODEC_ID_XMAFRAMES.  Mirrors
// the runtime decode path, so A/B against the live output is direct.
struct ContextDecoder {
  uint32_t context_id = 0;
  uint32_t rate_hz = 0;
  uint8_t channels = 0;
  AVCodecContext* ctx = nullptr;
  AVPacket* pkt = nullptr;
  AVFrame* frame = nullptr;
  std::unique_ptr<WavWriter> wav;
  uint64_t frames_in = 0;
  uint64_t frames_out = 0;
  uint64_t decode_fail = 0;
  // For mis-walk detection: previous packet index (within buf_idx).
  int last_buf_idx = -1;
  int last_pkt_idx = -1;

  ~ContextDecoder() {
    if (frame) av_frame_free(&frame);
    if (pkt) av_packet_free(&pkt);
    if (ctx) avcodec_free_context(&ctx);
  }
  ContextDecoder() = default;
  ContextDecoder(ContextDecoder&& other) noexcept
      : context_id(other.context_id),
        rate_hz(other.rate_hz),
        channels(other.channels),
        ctx(other.ctx),
        pkt(other.pkt),
        frame(other.frame),
        wav(std::move(other.wav)),
        frames_in(other.frames_in),
        frames_out(other.frames_out),
        decode_fail(other.decode_fail),
        last_buf_idx(other.last_buf_idx),
        last_pkt_idx(other.last_pkt_idx) {
    other.ctx = nullptr;
    other.pkt = nullptr;
    other.frame = nullptr;
  }
  ContextDecoder& operator=(ContextDecoder&&) = delete;
  ContextDecoder(const ContextDecoder&) = delete;
  ContextDecoder& operator=(const ContextDecoder&) = delete;
};

// Replay a per-frame capture via its .manifest + .xframes pair. Each
// context_id gets its own AV_CODEC_ID_XMAFRAMES decoder at the
// recorded rate_hz / channels; AVPacket payloads are fed in seq
// order. Output: one .wav per context. Logs packet-index
// discontinuities so manual review can flag mis-walks.
int RunManifest(const Args& args) {
  std::vector<ManifestRow> rows;
  if (!ParseManifest(args.input_path, &rows)) return 1;
  if (rows.empty()) {
    std::fprintf(stderr, "error: manifest has no rows\n");
    return 1;
  }

  std::string xframes_path = args.input_path;
  const auto dot = xframes_path.find_last_of('.');
  if (dot == std::string::npos) {
    std::fprintf(stderr, "error: manifest path needs an extension\n");
    return 1;
  }
  xframes_path.replace(dot, std::string::npos, ".xframes");
  FILE* xframes = std::fopen(xframes_path.c_str(), "rb");
  if (!xframes) {
    std::fprintf(stderr, "error: cannot open %s\n", xframes_path.c_str());
    return 1;
  }

  std::string out_base = args.output_path;
  const auto out_dot = out_base.find_last_of('.');
  if (out_dot != std::string::npos) {
    out_base.resize(out_dot);
  }

  std::fprintf(stderr,
               "xma-replay: manifest mode (per-frame), %zu rows, "
               "xframes='%s'\n",
               rows.size(), xframes_path.c_str());

  std::map<uint32_t, ContextDecoder> contexts;
  std::vector<uint8_t> scratch;
  uint64_t total_in = 0;
  uint64_t total_out = 0;
  uint64_t total_fail = 0;

  for (const ManifestRow& r : rows) {
    auto it = contexts.find(r.context_id);
    if (it == contexts.end()) {
      ContextDecoder cd;
      cd.context_id = r.context_id;
      cd.rate_hz = r.rate_hz;
      cd.channels = r.channels ? r.channels : 2;
      cd.ctx = OpenXmaCodec(int(cd.rate_hz), int(cd.channels));
      if (!cd.ctx) {
        std::fprintf(stderr,
                     "error: AV_CODEC_ID_XMAFRAMES open failed for ctx %u\n",
                     r.context_id);
        std::fclose(xframes);
        return 1;
      }
      cd.pkt = av_packet_alloc();
      cd.frame = av_frame_alloc();
      cd.wav = std::make_unique<WavWriter>();
      const std::string out_path =
          out_base + "_ctx" + std::to_string(r.context_id) + ".wav";
      if (!cd.wav->Open(out_path, int(cd.rate_hz), int(cd.channels))) {
        std::fprintf(stderr, "error: cannot open output %s\n",
                     out_path.c_str());
        std::fclose(xframes);
        return 1;
      }
      std::fprintf(stderr, "xma-replay: ctx %u → %s (%u Hz, %u ch)\n",
                   r.context_id, out_path.c_str(), r.rate_hz,
                   unsigned(cd.channels));
      it = contexts.emplace(r.context_id, std::move(cd)).first;
    }
    ContextDecoder& cd = it->second;

    // Mis-walk diagnostics: report unexpected packet-index transitions.
    // Within the same buf_idx, pkt_idx should advance monotonically by
    // 1 (or more if the runtime hit a 0xFF skip-packet).  A drop or a
    // jump back to a previous index signals a runtime bug.
    if (cd.last_buf_idx == int(r.buf_idx) && cd.last_pkt_idx >= 0) {
      const int delta = int(r.pkt_idx) - cd.last_pkt_idx;
      if (delta <= 0 || delta > 32) {
        std::fprintf(stderr,
                     "ctx %u seq %llu: pkt_idx %d -> %d (buf=%u, delta=%+d) "
                     "— review for mis-walk\n",
                     r.context_id, static_cast<unsigned long long>(r.seq),
                     cd.last_pkt_idx, int(r.pkt_idx), unsigned(r.buf_idx),
                     delta);
      }
    }
    cd.last_buf_idx = int(r.buf_idx);
    cd.last_pkt_idx = int(r.pkt_idx);

    if (std::fseek(xframes, long(r.offset), SEEK_SET) != 0) {
      std::fprintf(stderr, "error: seek to offset %llu failed\n",
                   static_cast<unsigned long long>(r.offset));
      break;
    }
    uint8_t hdr[4];
    if (std::fread(hdr, 1, 4, xframes) != 4) {
      std::fprintf(stderr, "error: truncated record header at seq %llu\n",
                   static_cast<unsigned long long>(r.seq));
      break;
    }
    const uint32_t sz_le = uint32_t(hdr[0]) | (uint32_t(hdr[1]) << 8) |
                           (uint32_t(hdr[2]) << 16) | (uint32_t(hdr[3]) << 24);
    if (sz_le != r.size) {
      std::fprintf(stderr,
                   "warning: manifest size %u != xframes size %u at seq %llu\n",
                   r.size, sz_le, static_cast<unsigned long long>(r.seq));
    }
    scratch.assign(r.size, 0);
    if (std::fread(scratch.data(), 1, r.size, xframes) != r.size) {
      std::fprintf(stderr, "error: truncated payload at seq %llu\n",
                   static_cast<unsigned long long>(r.seq));
      break;
    }

    av_packet_unref(cd.pkt);
    if (av_new_packet(cd.pkt, static_cast<int>(r.size)) < 0) {
      std::fprintf(stderr, "ctx %u seq %llu: av_new_packet failed\n",
                   r.context_id, static_cast<unsigned long long>(r.seq));
      ++cd.decode_fail;
      ++total_fail;
      continue;
    }
    std::memcpy(cd.pkt->data, scratch.data(), r.size);
    int sret = avcodec_send_packet(cd.ctx, cd.pkt);
    ++cd.frames_in;
    ++total_in;
    if (sret < 0) {
      ++cd.decode_fail;
      ++total_fail;
      if (args.verbose) {
        std::fprintf(stderr, "ctx %u seq %llu: send_packet: %s\n", r.context_id,
                     static_cast<unsigned long long>(r.seq), AvErr(sret));
      }
    }
    while (avcodec_receive_frame(cd.ctx, cd.frame) >= 0) {
      WriteFrameToWav(*cd.wav, cd.frame, cd.channels);
      ++cd.frames_out;
      ++total_out;
      av_frame_unref(cd.frame);
    }
  }

  for (auto& kv : contexts) {
    ContextDecoder& cd = kv.second;
    avcodec_send_packet(cd.ctx, nullptr);
    while (avcodec_receive_frame(cd.ctx, cd.frame) >= 0) {
      WriteFrameToWav(*cd.wav, cd.frame, cd.channels);
      ++cd.frames_out;
      ++total_out;
      av_frame_unref(cd.frame);
    }
    cd.wav->Close();
  }
  std::fclose(xframes);

  std::fprintf(stderr,
               "xma-replay: manifest done — %zu contexts, %llu frames in, "
               "%llu frames out, %llu decode_fail\n",
               contexts.size(), static_cast<unsigned long long>(total_in),
               static_cast<unsigned long long>(total_out),
               static_cast<unsigned long long>(total_fail));
  return 0;
}

// Read a 16-bit RIFF/WAVE PCM file into an interleaved int16 buffer.
// Returns false on parse failure or unsupported format.
bool ReadPcmWav(const std::string& path, std::vector<int16_t>* out,
                int* channels, int* sample_rate) {
  std::vector<uint8_t> bytes;
  if (!ReadWholeFile(path, &bytes)) return false;
  if (bytes.size() < 44) return false;
  if (std::memcmp(bytes.data(), "RIFF", 4) != 0) return false;
  if (std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) return false;
  size_t pos = 12;
  uint16_t fmt_codec = 0, fmt_ch = 0, fmt_bps = 0;
  uint32_t fmt_rate = 0;
  size_t data_off = 0, data_size = 0;
  while (pos + 8 <= bytes.size()) {
    char id[4];
    std::memcpy(id, bytes.data() + pos, 4);
    const uint32_t size = ReadLE32(bytes.data() + pos + 4);
    pos += 8;
    if (pos + size > bytes.size()) return false;
    if (std::memcmp(id, "fmt ", 4) == 0 && size >= 16) {
      fmt_codec = ReadLE16(bytes.data() + pos);
      fmt_ch = ReadLE16(bytes.data() + pos + 2);
      fmt_rate = ReadLE32(bytes.data() + pos + 4);
      fmt_bps = ReadLE16(bytes.data() + pos + 14);
    } else if (std::memcmp(id, "data", 4) == 0) {
      data_off = pos;
      data_size = size;
      break;
    }
    pos += (size + 1) & ~1u;
  }
  if (fmt_codec != 1 || fmt_bps != 16 || fmt_ch == 0 || data_size == 0)
    return false;
  *channels = fmt_ch;
  *sample_rate = fmt_rate;
  out->resize(data_size / 2);
  std::memcpy(out->data(), bytes.data() + data_off, data_size);
  return true;
}

// Compare decoded output against a reference WAV. Both must be 16-bit
// PCM. Reports per-channel RMS, sample-count mismatch, first sample
// where the two differ, and average abs error over the overlap.
int CompareToRef(const std::string& output_path, const std::string& ref_path) {
  std::vector<int16_t> our, ref;
  int our_ch = 0, our_sr = 0, ref_ch = 0, ref_sr = 0;
  if (!ReadPcmWav(output_path, &our, &our_ch, &our_sr)) {
    std::fprintf(stderr, "compare: cannot read output %s\n",
                 output_path.c_str());
    return 2;
  }
  if (!ReadPcmWav(ref_path, &ref, &ref_ch, &ref_sr)) {
    std::fprintf(stderr, "compare: cannot read reference %s\n",
                 ref_path.c_str());
    return 2;
  }
  if (our_ch != ref_ch || our_sr != ref_sr) {
    std::fprintf(
        stderr,
        "compare: format mismatch — output %dch@%dHz vs ref %dch@%dHz\n",
        our_ch, our_sr, ref_ch, ref_sr);
    return 2;
  }

  const int ch = our_ch;
  const size_t our_samples = our.size() / ch;
  const size_t ref_samples = ref.size() / ch;
  const size_t common = std::min(our_samples, ref_samples);

  std::fprintf(stderr,
               "\ncompare: ref=%zu samples (%.3fs), ours=%zu (%.3fs), "
               "common=%zu (%.3fs)\n",
               ref_samples, double(ref_samples) / our_sr, our_samples,
               double(our_samples) / our_sr, common, double(common) / our_sr);
  if (our_samples != ref_samples) {
    int64_t diff = int64_t(ref_samples) - int64_t(our_samples);
    std::fprintf(stderr,
                 "compare: length differs by %" PRId64 " samples (%.3fs)\n",
                 diff, double(diff) / our_sr);
  }

  // Per-channel RMS and avg-abs-error over the common region.
  std::vector<double> sumsq_ref(ch, 0.0), sumsq_our(ch, 0.0),
      sum_abserr(ch, 0.0);
  size_t first_diff_idx = SIZE_MAX;
  int first_diff_ch = -1;
  int16_t first_diff_ref = 0, first_diff_our = 0;
  for (size_t i = 0; i < common; ++i) {
    for (int c = 0; c < ch; ++c) {
      int16_t r = ref[i * ch + c];
      int16_t o = our[i * ch + c];
      sumsq_ref[c] += double(r) * r;
      sumsq_our[c] += double(o) * o;
      sum_abserr[c] += std::abs(int(r) - int(o));
      if (first_diff_idx == SIZE_MAX && r != o) {
        first_diff_idx = i;
        first_diff_ch = c;
        first_diff_ref = r;
        first_diff_our = o;
      }
    }
  }
  std::fprintf(stderr, "\ncompare: per-channel RMS / avg-abs-error:\n");
  for (int c = 0; c < ch; ++c) {
    const double r_rms =
        common > 0 ? std::sqrt(sumsq_ref[c] / double(common)) : 0.0;
    const double o_rms =
        common > 0 ? std::sqrt(sumsq_our[c] / double(common)) : 0.0;
    const double avg_err = common > 0 ? sum_abserr[c] / double(common) : 0.0;
    std::fprintf(stderr,
                 "  ch%d: ref_rms=%.0f our_rms=%.0f ratio=%.3f "
                 "avg|err|=%.0f (%.2f%% of ref_rms)\n",
                 c, r_rms, o_rms, r_rms > 0 ? o_rms / r_rms : 0.0, avg_err,
                 r_rms > 0 ? 100.0 * avg_err / r_rms : 0.0);
  }
  if (first_diff_idx == SIZE_MAX) {
    std::fprintf(stderr, "compare: bit-identical over the overlap\n");
  } else {
    std::fprintf(stderr,
                 "compare: first differing sample at idx=%zu (%.4fs), "
                 "ch=%d, ref=%d, ours=%d\n",
                 first_diff_idx, double(first_diff_idx) / our_sr, first_diff_ch,
                 first_diff_ref, first_diff_our);
  }
  return 0;
}

// Heuristic: a manifest is a text file whose first non-blank line
// starts with '#' (the auto-generated comment block) or "seq\t".
bool DetectManifest(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char buf[8] = {};
  size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
  std::fclose(f);
  if (n == 0) return false;
  if (buf[0] == '#') return true;
  if (n >= 4 && std::memcmp(buf, "seq\t", 4) == 0) return true;
  return false;
}

int Run(const Args& args) {
  // Manifest replay: separate dispatch path because output is one
  // WAV per context, not a single combined WAV.
  if (DetectManifest(args.input_path)) {
    return RunManifest(args);
  }

  WavWriter wav;
  if (!wav.Open(args.output_path, args.sample_rate, args.channels)) {
    std::fprintf(stderr, "error: cannot open output %s\n",
                 args.output_path.c_str());
    return 1;
  }

  std::vector<uint8_t> bytes;
  if (!ReadWholeFile(args.input_path, &bytes)) {
    std::fprintf(stderr, "error: cannot read input %s\n",
                 args.input_path.c_str());
    return 1;
  }

  DecodeStats stats;
  int rc;
  bool is_riff = DetectRiff(bytes);
  int sample_rate = args.sample_rate;
  int channels = args.channels;

  if (!is_riff) {
    std::fprintf(stderr,
                 "error: input is neither a RIFF/WAVE container nor a "
                 ".manifest file. The legacy raw .xframes mode is no longer "
                 "supported — use xma_<ts>.manifest from XmaFrameDumper.\n");
    return 1;
  }

  WaveInfo info;
  if (!ParseRiffWave(bytes, &info)) {
    std::fprintf(stderr, "error: not a recognized RIFF/WAVE file\n");
    return 1;
  }
  if (args.stream == -1) {
    rc = RunRiffAll(args, bytes, info, &wav, &stats, &sample_rate, &channels);
  } else if (args.codec == "xma2") {
    rc = RunRiffSingleStreamXma2(args, bytes, info, &wav, &stats, &sample_rate,
                                 &channels);
  } else {
    rc = RunRiff(args, bytes, info, &wav, &stats, &sample_rate, &channels);
  }

  wav.Close();

  // input_frames     = packets fed via avcodec_send_packet
  // decode_ok / fail = send_packet return codes
  // frames_out       = avcodec_receive_frame successes
  // missed           = packets that returned no frame (silent skip,
  //                    bit-misalignment recovery, mid-stream errors)
  const uint64_t missed = stats.input_frames > stats.frames_out
                              ? stats.input_frames - stats.frames_out
                              : 0;
  std::fprintf(stderr,
               "xma-replay: riff mode, %" PRIu64 " frames in / %" PRIu64
               " decode_ok / %" PRIu64 " decode_fail / %" PRIu64
               " frames out / %" PRIu64 " missed / %" PRIu64
               " samples (%.2fs at %d Hz, %d ch)\n",
               stats.input_frames, stats.decode_ok, stats.decode_fail,
               stats.frames_out, missed, wav.samples_written(),
               static_cast<double>(wav.samples_written()) / sample_rate,
               sample_rate, channels);
  if (rc == 0 && !args.compare_path.empty()) {
    CompareToRef(args.output_path, args.compare_path);
  }
  return rc;
}

int Main(const std::vector<std::string>& argv) {
  Args args;
  if (parse_args(argv, &args) < 0) return 2;
  return Run(args);
}

}  // namespace

XE_DEFINE_CONSOLE_APP_TRANSPARENT("xma-replay", Main);
