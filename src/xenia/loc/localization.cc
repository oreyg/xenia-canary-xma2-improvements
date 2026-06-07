/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/loc/localization.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "third_party/fmt/include/fmt/format.h"
#include "third_party/rapidjson/include/rapidjson/document.h"
#include "third_party/rapidjson/include/rapidjson/stringbuffer.h"
#include "third_party/rapidjson/include/rapidjson/writer.h"
#include "xenia/base/byte_order.h"
#include "xenia/base/cvar.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/base/platform.h"
#include "xenia/base/socket.h"
#include "xenia/base/string.h"
#include "xenia/base/system.h"
#include "xenia/cpu/backend/backend.h"
#include "xenia/cpu/backend/guest_prolog_hook.h"
#include "xenia/cpu/function.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/processor.h"
#include "xenia/memory.h"
#if XE_ARCH_AMD64
// The stackpoint-walk diagnostics ([sp] dump and buffer discovery) read the x64
// backend's per-frame guest stack pointers, which the generic Backend interface
// does not expose. Everything else in this module is backend-agnostic.
#include "xenia/cpu/backend/x64/x64_backend.h"
#endif  // XE_ARCH_AMD64

DEFINE_uint64(
    jit_argscan_lo, 0,
    "Lower bound for JIT argument scanning. When jit_argscan_hi is greater "
    "than this value, every guest function entry is instrumented to log the "
    "first time it is called with an argument register (r3-r10) pointing into "
    "[jit_argscan_lo, jit_argscan_hi). Used to locate text-processing code.",
    "Localization");
DEFINE_uint64(jit_argscan_hi, 0,
              "Upper bound (exclusive) for jit_argscan_lo scanning.", "Localization");

DEFINE_string(
    text_hook_config, "",
    "Path to a JSON file describing the text hooks to install. Each entry sets "
    "a guest function address and its per-hook registers and translation "
    "parameters; every listed hook is active simultaneously. Empty = no hooks. "
    "Schema: {\"hooks\":[{\"address\":\"82274208\",\"source_register\":4,"
    "\"length_register\":5,\"dest_register\":32,\"cooldown_ms\":3000,"
    "\"replace\":\"\",\"dest_min\":\"0\",\"dest_max\":\"0\","
    "\"translate\":{\"max_chars\":0,\"wrap\":60,\"wrap_slack\":16,"
    "\"clamp_to_original\":false,\"stack_max\":0}}]}. dest_min/dest_max (hex "
    "guest addresses) mark a destination range large enough for full-length "
    "text, exempt from clamp_to_original; 0/0 = none.",
    "Localization");
DEFINE_string(
    text_hook_output, "inplace",
    "How hooked/translated text is delivered, as a '|'-combinable set: "
    "\"inplace\" rewrites the guest string/length registers so the game renders "
    "it; \"clipboard[:jp|en|both]\" publishes the active on-screen lines to the "
    "Windows clipboard (default content \"jp\"). e.g. \"inplace|clipboard:en\". "
    "Empty = translate/extract only, no output.",
    "Localization");
DEFINE_bool(
    text_hook_dump_stack, false,
    "Log the destination buffer and guest call stack for each distinct hooked "
    "line (debugging aid to find a length-safe hook point / the owning frame).",
    "Localization");
DEFINE_bool(
    text_hook_probe, false,
    "On entry to each hooked function, log r3-r10 with a preview of any "
    "UTF-16BE string they point at (once per function). Use to discover which "
    "register holds the source string for a candidate hook address.",
    "Localization");
DEFINE_bool(
    text_hook_buffer_discover, false,
    "Log the owning guest function and buffer offset for every stack "
    "destination buffer seen by a text hook, derived from the PPC "
    "back-chain/stackpoint walk. Each unique result is printed as a ready-to-"
    "use text_hook_buffer_map entry.",
    "Localization");

DEFINE_bool(
    text_translate, false,
    "Live-translate hooked text via an OpenAI-compatible chat endpoint (e.g. a "
    "local Ollama server). Translation is asynchronous and cached; the original "
    "is shown until a line is ready. Per-hook rendering (wrap/clamp/limits) is "
    "configured in text_hook_config.",
    "Localization");
DEFINE_string(text_translate_host, "127.0.0.1",
              "Hostname of the translation server.", "Localization");
DEFINE_uint64(text_translate_port, 11434, "Port of the translation server.",
              "Localization");
DEFINE_string(text_translate_path, "/v1/chat/completions",
              "Request path of the translation endpoint.", "Localization");
DEFINE_string(text_translate_model, "translategemma:4b",
              "Model name to request from the translation server.",
              "Localization");
DEFINE_string(
    text_translate_prompt,
    "You are a professional Japanese (ja) to English (en) translator. Your goal "
    "is to accurately convey the meaning and nuances of the original Japanese "
    "text while adhering to English grammar, vocabulary, and cultural "
    "sensitivities.\nProduce only the English translation, without any "
    "additional explanations or commentary. Please translate the following "
    "Japanese text into English:\n\n\n{TEXT}",
    "Prompt template sent to the model. The substring {TEXT} is replaced with "
    "the source line (if absent, the line is appended).",
    "Localization");

namespace xe {
namespace loc {

namespace ppc = xe::cpu::ppc;
using xe::cpu::Function;
using xe::cpu::backend::GuestPseudoStackTrace;
#if XE_ARCH_AMD64
using xe::cpu::backend::x64::X64Backend;
using xe::cpu::backend::x64::X64BackendContext;
using xe::cpu::backend::x64::X64BackendStackpoint;
#endif  // XE_ARCH_AMD64

namespace {
std::mutex jit_argscan_mutex;
std::unordered_set<uint64_t> jit_argscan_seen;
}  // namespace


// Reads up to `max` UTF-16BE units from the guest string at `addr` into `out`,
// stopping at NUL. Returns the count read (0 if `addr` isn't readable guest
// memory).
static uint32_t SafeReadString16(ppc::PPCContext* ctx, uint32_t addr,
                                 char16_t* out, uint32_t max) {
  uint32_t i = 0;
  uint32_t checked_page = ~0u;
  while (i < max) {
    const uint32_t cur = addr + i * 2;
    // Validate readability once per guest page so a register that isn't a valid
    // string pointer can't fault - portable, no SEH needed.
    const uint32_t page = cur >> 12;
    if (page != checked_page) {
      auto* heap = ctx->processor->memory()->LookupHeap(cur);
      uint32_t protect = 0;
      if (!heap || !heap->QueryProtect(cur, &protect) ||
          !(protect & kMemoryProtectRead)) {
        break;
      }
      checked_page = page;
    }
    const uint16_t c = *ctx->TranslateVirtualBE<uint16_t>(cur);
    if (c == 0) {
      break;
    }
    out[i++] = static_cast<char16_t>(c);
  }
  return i;
}

// True if the string has at least two kana (hiragana/katakana). Binary data
// often lands in the CJK ideograph range but rarely yields a run of kana, so
// this distinguishes a real Japanese source string from garbage.
static bool HasKana(const std::u16string& s) {
  int kana = 0;
  for (char16_t c : s) {
    const uint32_t u = static_cast<uint32_t>(c);
    if ((u >= 0x3040 && u <= 0x30FF) || (u >= 0xFF66 && u <= 0xFF9D)) {
      if (++kana >= 2) {
        return true;
      }
    }
  }
  return false;
}

// Discovery hook called at guest function entry when JIT arg scanning is on.
// Logs the first time a function is entered with an argument register pointing
// into the configured guest range. With text_hook_probe on, only reports
// registers pointing at a real Japanese (kana) string, cutting the noise.
static uint64_t JitArgScanHook(void* raw_context, uint64_t guest_func) {
  auto* ctx = reinterpret_cast<ppc::PPCContext*>(raw_context);
  const uint64_t lo = cvars::jit_argscan_lo;
  const uint64_t hi = cvars::jit_argscan_hi;
  const bool require_text = cvars::text_hook_probe;
  for (int i = 3; i <= 10; ++i) {
    const uint32_t value = static_cast<uint32_t>(ctx->r[i]);
    if (value < lo || value >= hi) {
      continue;
    }
    std::u16string s;
    if (require_text) {
      char16_t buf[49];
      const uint32_t n =
          SafeReadString16(ctx, value, buf, 48);
      s.assign(buf, n);
      if (!HasKana(s)) {
        continue;
      }
    }
    const uint64_t key = (guest_func << 8) | static_cast<uint64_t>(i);
    {
      std::lock_guard<std::mutex> lock(jit_argscan_mutex);
      if (!jit_argscan_seen.insert(key).second) {
        continue;
      }
    }
    if (require_text) {
      XELOGI("[jitscan] func=0x{:08X} r{}=0x{:08X} '{}'",
             static_cast<uint32_t>(guest_func), i, value, xe::to_utf8(s));
      // Log the caller chain so the fill (the one called from 0x821996B8) is
      // identifiable among all the string-handling functions.
      GuestPseudoStackTrace st;
      if (ctx->processor->backend()->PopulatePseudoStacktrace(&st)) {
        std::string callers;
        for (uint32_t k = 0; k < st.count && k < 6; ++k) {
          callers += fmt::format(" {:08X}", st.return_addrs[k]);
        }
        XELOGI("[jitscan]   callers:{}", callers);
      }
      continue;
    }
    XELOGI("[jitscan] func=0x{:08X} r{}=0x{:08X}",
           static_cast<uint32_t>(guest_func), i, value);
  }
  return 0;
}

namespace {
std::mutex text_hook_mutex;
// One on-screen line: when it was last seen (for cooldown pruning) and the order
// it first appeared (for stable clipboard ordering).
struct ActiveLine {
  std::chrono::steady_clock::time_point seen;
  uint64_t order = 0;
  uint64_t cooldown_ms = 3000;  // From the hook that last refreshed this line.
};
// Strings currently on screen ("active set"): entries that stop appearing for
// longer than the cooldown are dropped so they can be published again if they
// later reappear.
std::unordered_map<std::u16string, ActiveLine> text_hook_active;
uint64_t text_hook_order_counter = 0;
std::u16string text_hook_clipboard_last;  // Last text pushed to the clipboard.
}  // namespace

namespace {
// Persistent guest buffers built by TextHookGuestString, keyed by source text.
std::mutex g_guest_string_mutex;
std::unordered_map<std::u16string, std::pair<uint32_t, uint32_t>>
    g_guest_string_cache;
}  // namespace

// Returns a persistent guest buffer holding `text` as a NUL-terminated UTF-16BE
// string, allocating it (and never freeing it) on first use. Returns
// {guest_address, char_count}, or {0, 0} on failure. Cached per string so a
// repeated line reuses one allocation.
[[maybe_unused]] static std::pair<uint32_t, uint32_t> TextHookGuestString(
    ppc::PPCContext* ctx, const std::u16string& text) {
  std::lock_guard<std::mutex> lock(g_guest_string_mutex);
  auto it = g_guest_string_cache.find(text);
  if (it != g_guest_string_cache.end()) {
    return it->second;
  }
  const uint32_t count = static_cast<uint32_t>(text.size());
  const uint32_t addr =
      ctx->processor->memory()->SystemHeapAlloc((count + 1) * 2);
  if (!addr) {
    return {0, 0};
  }
  auto* dest = ctx->TranslateVirtualBE<uint16_t>(addr);
  for (uint32_t i = 0; i < count; ++i) {
    dest[i] = static_cast<uint16_t>(text[i]);
  }
  dest[count] = 0;
  const std::pair<uint32_t, uint32_t> entry = {addr, count};
  g_guest_string_cache[text] = entry;
  return entry;
}


// Greedy word-wrap: inserts newlines so no line exceeds max_width characters.
// Breaks at spaces where possible, hard-breaks within over-long words, and
// leaves existing newlines intact.
[[maybe_unused]] static std::u16string WrapText(const std::u16string& text,
                                             size_t max_width, size_t slack) {
  if (max_width == 0) {
    return text;
  }
  // Whether a newline already appears within `slack` characters from `pos`.
  const auto newline_soon = [&](size_t pos) {
    const size_t end = std::min(text.size(), pos + slack);
    for (size_t j = pos; j < end; ++j) {
      if (text[j] == u'\n') {
        return true;
      }
    }
    return false;
  };
  std::u16string out;
  out.reserve(text.size() + text.size() / max_width + 1);
  size_t col = 0;
  for (size_t i = 0; i < text.size();) {
    const char16_t c = text[i];
    if (c == u'\r') {
      ++i;  // Drop CR; the following LF resets the column.
      continue;
    }
    if (c == u'\n') {
      out.push_back(c);
      col = 0;
      ++i;
      // Collapse several consecutive line breaks into one.
      while (i < text.size() && (text[i] == u'\n' || text[i] == u'\r')) {
        ++i;
      }
      continue;
    }
    if (c == u' ') {
      // Measure the next word; if it won't fit, break here instead of spacing.
      size_t k = i + 1;
      while (k < text.size() && text[k] != u' ' && text[k] != u'\n') {
        ++k;
      }
      const size_t word = k - i - 1;
      if (col != 0 && col + 1 + word > max_width && !newline_soon(i)) {
        out.push_back(u'\n');
        col = 0;
      } else {
        out.push_back(u' ');
        ++col;
      }
      ++i;
      continue;
    }
    out.push_back(c);
    ++col;
    ++i;
    if (col >= max_width && i < text.size() && text[i] != u' ' &&
        text[i] != u'\n' && !newline_soon(i)) {
      out.push_back(u'\n');  // Hard-break an over-long word.
      col = 0;
    }
  }
  return out;
}

// --- Asynchronous live translation with a per-string cache ---
struct TranslationEntry {
  enum Status { kPending, kDone, kFailed };
  Status status = kPending;
  std::u16string text;  // The raw (unwrapped) English; rendered per-hook later.
};
namespace {
std::mutex g_translate_mutex;
std::unordered_map<std::u16string, TranslationEntry> g_translate_cache;
std::deque<std::u16string> g_translate_queue;
std::condition_variable g_translate_cv;
std::atomic<bool> g_translate_worker_started{false};

// Exact strings we have rendered into a guest buffer. A redirected buffer is
// both the game's source and our render target, so our output is read back as
// "source" on the next frame; if a hooked string is one we emitted, we skip it
// (it already holds the translation) to avoid a re-translation feedback loop.
std::mutex g_emitted_mutex;
std::unordered_set<std::u16string> g_emitted_outputs;
}  // namespace

[[maybe_unused]] static void RememberEmittedOutput(const std::u16string& text) {
  std::lock_guard<std::mutex> lock(g_emitted_mutex);
  g_emitted_outputs.insert(text);
}

[[maybe_unused]] static bool IsEmittedOutput(const std::u16string& text) {
  std::lock_guard<std::mutex> lock(g_emitted_mutex);
  return g_emitted_outputs.count(text) != 0;
}

// Trims surrounding whitespace and quotes from a model reply.
static std::string TrimReply(const std::string& s) {
  const char* trim_set = " \t\r\n\"";
  const size_t first = s.find_first_not_of(trim_set);
  if (first == std::string::npos) {
    return std::string();
  }
  const size_t last = s.find_last_not_of(trim_set);
  return s.substr(first, last - first + 1);
}

// Sends one chat-completion request and returns the reply text, or false.
static bool TranslateViaOllama(const std::string& source_utf8,
                               std::string* out_utf8) {
  // Build the request body with rapidjson so the text is escaped correctly.
  std::string prompt = cvars::text_translate_prompt;
  const std::string token = "{TEXT}";
  const size_t token_pos = prompt.find(token);
  if (token_pos != std::string::npos) {
    prompt.replace(token_pos, token.size(), source_utf8);
  } else {
    prompt += "\n\n" + source_utf8;
  }
  rapidjson::Document req(rapidjson::kObjectType);
  auto& al = req.GetAllocator();
  req.AddMember("model",
                rapidjson::Value(cvars::text_translate_model.c_str(), al), al);
  req.AddMember("stream", false, al);
  rapidjson::Value messages(rapidjson::kArrayType);
  rapidjson::Value message(rapidjson::kObjectType);
  message.AddMember("role", "user", al);
  message.AddMember("content", rapidjson::Value(prompt.c_str(), al), al);
  messages.PushBack(message, al);
  req.AddMember("messages", messages, al);
  rapidjson::StringBuffer json;
  rapidjson::Writer<rapidjson::StringBuffer> writer(json);
  req.Accept(writer);
  const std::string body(json.GetString(), json.GetSize());

  auto socket = xe::Socket::Connect(
      cvars::text_translate_host,
      static_cast<uint16_t>(cvars::text_translate_port));
  if (!socket) {
    return false;
  }
  const std::string request =
      "POST " + cvars::text_translate_path + " HTTP/1.0\r\n" +
      "Host: " + cvars::text_translate_host + "\r\n" +
      "Content-Type: application/json\r\n" +
      "Content-Length: " + std::to_string(body.size()) + "\r\n" +
      "Connection: close\r\n\r\n" + body;
  if (!socket->Send(request)) {
    return false;
  }

  // HTTP/1.0 + Connection: close -> read until the server closes the socket.
  std::string response;
  char chunk[8192];
  auto last_data = std::chrono::steady_clock::now();
  for (;;) {
    const size_t received = socket->Receive(chunk, sizeof(chunk));
    if (received == static_cast<size_t>(-1)) {
      break;  // Closed by server.
    }
    if (received == 0) {
      if (std::chrono::steady_clock::now() - last_data >
          std::chrono::seconds(30)) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    response.append(chunk, received);
    last_data = std::chrono::steady_clock::now();
  }
  socket->Close();

  const size_t header_end = response.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    return false;
  }
  rapidjson::Document doc;
  doc.Parse(response.c_str() + header_end + 4);
  if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("choices") ||
      !doc["choices"].IsArray() || doc["choices"].Empty()) {
    return false;
  }
  const auto& choice = doc["choices"][rapidjson::SizeType(0)];
  if (!choice.HasMember("message") || !choice["message"].IsObject() ||
      !choice["message"].HasMember("content") ||
      !choice["message"]["content"].IsString()) {
    return false;
  }
  *out_utf8 = TrimReply(choice["message"]["content"].GetString());
  return !out_utf8->empty();
}

// Background worker: drains the queue and caches the raw English per source.
// Wrapping/rendering is done per-hook at render time, not here.
static void TextTranslateWorker() {
  for (;;) {
    std::u16string source;
    {
      std::unique_lock<std::mutex> lock(g_translate_mutex);
      g_translate_cv.wait(lock, [] { return !g_translate_queue.empty(); });
      source = g_translate_queue.front();
      g_translate_queue.pop_front();
    }
    std::string reply_utf8;
    const bool ok = TranslateViaOllama(xe::to_utf8(source), &reply_utf8);
    std::u16string reply = ok ? xe::to_utf16(reply_utf8) : std::u16string();
    {
      std::lock_guard<std::mutex> lock(g_translate_mutex);
      TranslationEntry& entry = g_translate_cache[source];
      if (ok && !reply.empty()) {
        entry.status = TranslationEntry::kDone;
        entry.text = reply;
      } else {
        entry.status = TranslationEntry::kFailed;
      }
    }
    if (ok) {
      XELOGI("[translate] {} -> {}", xe::to_utf8(source), reply_utf8);
    } else {
      XELOGI("[translate] failed: {}", xe::to_utf8(source));
    }
  }
}

// If the translation of `source` is ready, copies the raw English into *out_en
// and returns true; otherwise queues it and returns false (render the original).
static bool TextTranslateLookup(const std::u16string& source,
                                std::u16string* out_en) {
  out_en->clear();
  if (!g_translate_worker_started.exchange(true)) {
    std::thread(TextTranslateWorker).detach();
  }
  std::lock_guard<std::mutex> lock(g_translate_mutex);
  auto it = g_translate_cache.find(source);
  if (it == g_translate_cache.end()) {
    g_translate_cache.emplace(source, TranslationEntry{});
    g_translate_queue.push_back(source);
    g_translate_cv.notify_one();
    return false;
  }
  if (it->second.status == TranslationEntry::kDone) {
    *out_en = it->second.text;
    return true;
  }
  return false;
}

// --- Per-hook configuration (loaded from text_hook_config JSON) -------------

// One configured text hook. All fields are per-hook; multiple hooks run at once.
struct HookConfig {
  uint32_t address = 0;
  uint32_t source_register = 4;   // GPR holding the source string pointer.
  uint32_t length_register = 5;   // GPR holding the char count (32 = none).
  uint32_t dest_register = 32;    // GPR to redirect to a scratch (32 = none).
  uint64_t cooldown_ms = 3000;    // Clipboard active-set retention.
  std::u16string replace;           // Fixed replacement text (empty = translate).
  uint64_t dest_min = 0;          // Destinations in [dest_min, dest_max] are
  uint64_t dest_max = 0;          // full-length-safe (exempt from clamping).
  uint64_t translate_max_chars = 0;
  uint64_t translate_wrap = 0;
  uint64_t translate_wrap_slack = 16;
  bool translate_clamp_to_original = true;
  uint64_t translate_stack_max = 0;
  // If non-empty, the hook only acts when one of these functions is in the guest
  // call stack. Gates a shared/generic routine (e.g. a late display primitive)
  // to a single call path so unrelated text isn't read, translated, or replaced.
  std::vector<uint32_t> callers;
  // Write the replacement directly into the source buffer (guest_ptr) instead of
  // swapping the source pointer register. Needed when the renderer dereferences a
  // fixed/stored buffer pointer rather than the argument register, so a pointer
  // swap is invisible to it. Capped to the original length unless
  // clamp_to_original is false (which then assumes a redirected/large buffer).
  bool overwrite_in_place = false;
};

namespace {
std::unordered_map<uint32_t, HookConfig> g_hooks;  // guest address -> config
std::mutex g_hook_output_mutex;
std::unordered_map<uint32_t, uint32_t> g_hook_output;  // address -> guest scratch
}  // namespace

// Reads text_hook_config (a JSON file path) into g_hooks. Called once at setup.
// A relative path is resolved against `config_folder` (same root as the TOML).
static void LoadHooks(const std::filesystem::path& config_folder) {
  if (cvars::text_hook_config.empty()) {
    return;
  }
  std::filesystem::path path = xe::to_path(cvars::text_hook_config);
  if (path.is_relative() && !config_folder.empty()) {
    path = config_folder / path;
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    XELOGW("[texthook] could not open text_hook_config '{}'.",
           xe::path_to_utf8(path));
    return;
  }
  const std::string json((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  rapidjson::Document doc;
  doc.Parse(json.c_str());
  if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("hooks") ||
      !doc["hooks"].IsArray()) {
    XELOGW("[texthook] text_hook_config '{}' is not valid JSON with a 'hooks' "
           "array.",
           path);
    return;
  }
  auto get_u = [](const rapidjson::Value& o, const char* k,
                  uint64_t def) -> uint64_t {
    if (!o.HasMember(k)) return def;
    const rapidjson::Value& v = o[k];
    if (v.IsUint64()) return v.GetUint64();
    if (v.IsInt64()) return static_cast<uint64_t>(v.GetInt64());
    if (v.IsString()) return std::strtoull(v.GetString(), nullptr, 0);
    return def;
  };
  auto get_b = [](const rapidjson::Value& o, const char* k, bool def) -> bool {
    return (o.HasMember(k) && o[k].IsBool()) ? o[k].GetBool() : def;
  };
  // Guest addresses: hex string (bare or 0x-prefixed) or number.
  auto get_addr = [](const rapidjson::Value& o, const char* k) -> uint64_t {
    if (!o.HasMember(k)) return 0;
    const rapidjson::Value& v = o[k];
    if (v.IsString()) return std::strtoull(v.GetString(), nullptr, 16);
    if (v.IsUint64()) return v.GetUint64();
    return 0;
  };
  for (const auto& h : doc["hooks"].GetArray()) {
    if (!h.IsObject() || !h.HasMember("address")) {
      continue;
    }
    HookConfig c;
    const rapidjson::Value& a = h["address"];
    if (a.IsString()) {
      c.address = static_cast<uint32_t>(std::strtoull(a.GetString(), nullptr, 16));
    } else if (a.IsUint64()) {
      c.address = static_cast<uint32_t>(a.GetUint64());
    }
    if (!c.address) {
      continue;
    }
    c.source_register = static_cast<uint32_t>(get_u(h, "source_register", 4));
    c.length_register = static_cast<uint32_t>(get_u(h, "length_register", 5));
    c.dest_register = static_cast<uint32_t>(get_u(h, "dest_register", 32));
    c.cooldown_ms = get_u(h, "cooldown_ms", 3000);
    c.dest_min = get_addr(h, "dest_min");
    c.dest_max = get_addr(h, "dest_max");
    if (h.HasMember("replace") && h["replace"].IsString()) {
      c.replace = xe::to_utf16(h["replace"].GetString());
    }
    if (h.HasMember("translate") && h["translate"].IsObject()) {
      const rapidjson::Value& t = h["translate"];
      c.translate_max_chars = get_u(t, "max_chars", 0);
      c.translate_wrap = get_u(t, "wrap", 0);
      c.translate_wrap_slack = get_u(t, "wrap_slack", 16);
      c.translate_clamp_to_original = get_b(t, "clamp_to_original", true);
      c.translate_stack_max = get_u(t, "stack_max", 0);
    }
    c.overwrite_in_place = get_b(h, "overwrite_in_place", false);
    if (h.HasMember("callers") && h["callers"].IsArray()) {
      for (const auto& cv : h["callers"].GetArray()) {
        uint32_t f = 0;
        if (cv.IsString()) {
          f = static_cast<uint32_t>(std::strtoull(cv.GetString(), nullptr, 16));
        } else if (cv.IsUint64()) {
          f = static_cast<uint32_t>(cv.GetUint64());
        }
        if (f) {
          c.callers.push_back(f);
        }
      }
    }
    g_hooks[c.address] = c;
    XELOGI(
        "[texthook] hook {:08X}: src=r{} len=r{} dest=r{} cooldown={}ms wrap={}",
        c.address, c.source_register, c.length_register, c.dest_register,
        c.cooldown_ms, c.translate_wrap);
  }
  XELOGI("[texthook] loaded {} hook(s) from '{}'.", g_hooks.size(),
         xe::path_to_utf8(path));
}

static const HookConfig* FindHook(uint32_t address) {
  auto it = g_hooks.find(address);
  return it == g_hooks.end() ? nullptr : &it->second;
}

// Lazily allocates (and reuses) the per-hook guest scratch for rendered text.
// The consumer reads it before the hook is next called, so reuse is safe.
[[maybe_unused]] static uint32_t GetHookOutputBuffer(Memory* memory,
                                                     uint32_t address) {
  std::lock_guard<std::mutex> lock(g_hook_output_mutex);
  auto it = g_hook_output.find(address);
  if (it != g_hook_output.end()) {
    return it->second;
  }
  uint32_t addr = memory->SystemHeapAlloc(8192 * 2);  // up to 8192 UTF-16 chars
  g_hook_output[address] = addr;
  return addr;
}

// How hooked text is delivered, parsed from text_hook_output.
struct OutputMode {
  bool inplace = false;
  int clipboard = 0;  // 0=off, 1=jp, 2=en, 3=both
};
[[maybe_unused]] static OutputMode ParseOutputMode() {
  OutputMode m;
  const std::string& s = cvars::text_hook_output;
  size_t pos = 0;
  while (pos < s.size()) {
    size_t bar = s.find('|', pos);
    std::string tok = s.substr(
        pos, bar == std::string::npos ? std::string::npos : bar - pos);
    pos = (bar == std::string::npos) ? s.size() : bar + 1;
    size_t b = tok.find_first_not_of(" \t");
    if (b == std::string::npos) {
      continue;
    }
    tok = tok.substr(b, tok.find_last_not_of(" \t") - b + 1);
    if (tok == "inplace") {
      m.inplace = true;
    } else if (tok == "clipboard" || tok == "clipboard:jp") {
      m.clipboard = 1;
    } else if (tok == "clipboard:en") {
      m.clipboard = 2;
    } else if (tok == "clipboard:both") {
      m.clipboard = 3;
    }
  }
  return m;
}

// Refreshes the active set with `source`, then rebuilds the clipboard from every
// currently-active line and publishes it if it changed. mode: 1=jp, 2=en,
// 3=both. en falls back to the Japanese until each translation is ready.
[[maybe_unused]] static void TextHookPublishClipboard(const std::u16string& source,
                                                      int mode,
                                                      uint64_t cooldown_ms) {
  std::u16string publish;
  {
    std::lock_guard<std::mutex> lock(text_hook_mutex);
    const auto now = std::chrono::steady_clock::now();
    // Drop lines not seen within their own hook's cooldown window.
    for (auto it = text_hook_active.begin(); it != text_hook_active.end();) {
      if (now - it->second.seen >
          std::chrono::milliseconds(static_cast<int64_t>(it->second.cooldown_ms))) {
        it = text_hook_active.erase(it);
      } else {
        ++it;
      }
    }
    // Refresh / insert the current line, assigning a stable first-seen order.
    ActiveLine& line = text_hook_active[source];
    if (line.order == 0) {
      line.order = ++text_hook_order_counter;
    }
    line.seen = now;
    line.cooldown_ms = cooldown_ms;
    // Collect the active lines in first-seen order.
    std::vector<std::pair<uint64_t, const std::u16string*>> ordered;
    ordered.reserve(text_hook_active.size());
    for (auto& kv : text_hook_active) {
      ordered.emplace_back(kv.second.order, &kv.first);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    // Build the published text. For en/both, read each translation from cache.
    std::lock_guard<std::mutex> tlock(g_translate_mutex);
    for (auto& item : ordered) {
      const std::u16string& jp = *item.second;
      const std::u16string* en = nullptr;
      if (mode != 1) {
        auto it = g_translate_cache.find(jp);
        if (it != g_translate_cache.end() &&
            it->second.status == TranslationEntry::kDone &&
            !it->second.text.empty()) {
          en = &it->second.text;
        }
      }
      if (!publish.empty()) {
        publish += u"\n";
      }
      if (mode == 1) {  // jp
        publish += jp;
      } else if (mode == 2) {  // en (fall back to jp until ready)
        publish += en ? *en : jp;
      } else {  // both
        publish += jp;
        if (en) {
          publish += u"\n";
          publish += *en;
        }
      }
    }
    if (publish == text_hook_clipboard_last) {
      return;  // Active set unchanged - leave the clipboard alone.
    }
    text_hook_clipboard_last = publish;
  }
  // Push to the clipboard on a detached thread so a busy clipboard can never
  // stall the guest thread that produced the text.
  std::thread([utf8 = xe::to_utf8(publish)] {
    xe::SetClipboardText(utf8);
  }).detach();
}

// Logs `count` raw guest bytes at `addr` as hex + ASCII rows. Used to inspect
// what lies after the NUL we read (control codes / scene markers a redirect can
// drop), to debug cases where replacing a line breaks the game's flow.
[[maybe_unused]] static void DumpGuestBytes(ppc::PPCContext* ctx, uint32_t addr,
                                            uint32_t count, const char* tag) {
  const uint8_t* p = ctx->TranslateVirtual<uint8_t*>(addr);
  for (uint32_t off = 0; off < count; off += 16) {
    std::string hex;
    std::string asc;
    for (uint32_t i = 0; i < 16 && off + i < count; ++i) {
      const uint8_t b = p[off + i];
      hex += fmt::format("{:02X} ", b);
      asc += (b >= 0x20 && b < 0x7F) ? static_cast<char>(b) : '.';
    }
    XELOGI("[{}] {:08X}: {:<48} {}", tag, addr + off, hex, asc);
  }
}

// True if the hook may act on this call. A hook with a `callers` allowlist only
// fires when its immediate guest caller (the function containing the link
// register's return address on entry) is listed. This restricts a shared, hot
// routine to one call path in O(1), before any string read - essential when
// hooking a per-line/per-glyph primitive that would otherwise tank the frame
// rate scanning every call's source register.
static bool HookCallerAllowed(ppc::PPCContext* ctx, const HookConfig* hook) {
  if (hook->callers.empty()) {
    return true;
  }
  const uint32_t ret = static_cast<uint32_t>(ctx->lr);
  uint32_t caller = ret;
  if (Function* f = ctx->processor->LookupFunction(ret)) {
    caller = f->address();
  }
  for (uint32_t allowed : hook->callers) {
    if (caller == allowed || ret == allowed) {
      return true;
    }
  }
  // Discovery aid: surface each distinct immediate caller we reject, with a
  // sample of its source string, so the real (dialogue) path can be identified
  // and added to the allowlist. One line per distinct (hook, caller).
  if (cvars::text_hook_dump_stack) {
    static std::mutex m;
    static std::unordered_set<uint64_t> seen;
    const uint64_t key = (static_cast<uint64_t>(hook->address) << 32) | caller;
    bool log_it = false;
    {
      std::lock_guard<std::mutex> lock(m);
      if (seen.size() < 128 && seen.insert(key).second) {
        log_it = true;
      }
    }
    if (log_it) {
      char16_t b[49];
      const uint32_t pr = hook->source_register <= 31
                              ? static_cast<uint32_t>(ctx->r[hook->source_register])
                              : 0;
      const uint32_t n =
          pr ? SafeReadString16(ctx, pr, b, 48) : 0;
      XELOGI("[callgate] hook=0x{:08X} caller=0x{:08X} src=r{}=0x{:08X} '{}'",
             hook->address, caller, hook->source_register, pr,
             xe::to_utf8(std::u16string(b, n)));
    }
  }
  return false;
}

// Prolog hook for any guest function listed in text_hook_config. Looks the hook
// up by address, reads the UTF-16BE source string from its source register, then
// translates/replaces it in-place and/or exports the active set to the clipboard
// per text_hook_output.
static uint64_t TextHookCallback(void* raw_context, uint64_t guest_func) {
  auto* ctx = reinterpret_cast<ppc::PPCContext*>(raw_context);
  const HookConfig* hook = FindHook(static_cast<uint32_t>(guest_func));
  if (!hook) {
    return 0;
  }
  // Cheapest possible early-out for a hot routine: bail before probe/dump/read.
  if (!HookCallerAllowed(ctx, hook)) {
    return 0;
  }

  // Register probe: when any of r3-r10 points at a UTF-16BE string containing
  // Japanese, log all of them (once per distinct line) to find which register
  // carries the source string at a candidate hook address.
  if (cvars::text_hook_probe) {
    std::u16string strs[11];
    std::u16string jp_key;
    for (uint32_t r = 3; r <= 10; ++r) {
      const uint32_t p = static_cast<uint32_t>(ctx->r[r]);
      char16_t buf[49];
      const uint32_t n =
          p ? SafeReadString16(ctx, p, buf, 48) : 0;
      strs[r].assign(buf, n);
      if (jp_key.empty() && HasKana(strs[r])) {
        jp_key = strs[r];
      }
    }
    if (!jp_key.empty()) {
      static std::mutex probe_mutex;
      static std::unordered_set<std::u16string> probe_seen;
      bool do_probe = false;
      {
        std::lock_guard<std::mutex> lock(probe_mutex);
        if (probe_seen.size() < 64 && probe_seen.insert(jp_key).second) {
          do_probe = true;
        }
      }
      if (do_probe) {
        XELOGI("[probe] func=0x{:08X}:", static_cast<uint32_t>(guest_func));
        for (uint32_t r = 3; r <= 10; ++r) {
          XELOGI("[probe]   r{:<2}=0x{:08X} '{}'", r,
                 static_cast<uint32_t>(ctx->r[r]), xe::to_utf8(strs[r]));
        }
      }
    }
  }

  const uint32_t reg = hook->source_register;
  if (reg > 31) {
    return 0;
  }
  const uint32_t guest_ptr = static_cast<uint32_t>(ctx->r[reg]);
  if (!guest_ptr) {
    return 0;
  }
  // Read the NUL-terminated UTF-16BE string: a hooked general-purpose function
  // may hold a non-string value in the source register, so SafeReadString16
  // validates each guest page and can't fault on a bogus pointer.
  char16_t buf[4096];
  const uint32_t n = SafeReadString16(ctx, guest_ptr, buf, 4095);
  std::u16string text(buf, n);
  if (text.empty()) {
    return 0;
  }

  // Dump the destination buffer, its length, and the guest call stack for each
  // distinct destination (up to 32), to find where the per-line buffer is
  // allocated. r3 is the destination, r5 the count (before any redirect).
  if (cvars::text_hook_dump_stack) {
    static std::mutex dump_mutex;
    static std::unordered_set<uint32_t> dumped_dests;
    const uint32_t dest = static_cast<uint32_t>(ctx->r[3]);
    const uint32_t count = static_cast<uint32_t>(ctx->r[5]);
    bool do_dump = false;
    {
      std::lock_guard<std::mutex> lock(dump_mutex);
      if (dumped_dests.size() < 32 && dumped_dests.insert(dest).second) {
        do_dump = true;
      }
    }
    if (do_dump) {
      const std::string preview = xe::to_utf8(text.substr(0, 48));
      XELOGI("[texthook] line dest=0x{:08X} count={} text='{}' stack:", dest,
             count, preview);
      GuestPseudoStackTrace st;
      if (ctx->processor->backend()->PopulatePseudoStacktrace(&st)) {
        for (uint32_t i = 0; i < st.count; ++i) {
          // Resolve the return address to its containing function so the start
          // address (the hookable entry) is visible, not just the return point.
          Function* fn = ctx->processor->LookupFunction(st.return_addrs[i]);
          XELOGI("[texthook]   #{:02} ret=0x{:08X} func=0x{:08X}", i,
                 st.return_addrs[i], fn ? fn->address() : 0u);
        }
      }
    }
  }

#if XE_ARCH_AMD64
  // One-shot: dump the guest stackpoints (r1 + return address + function per
  // frame) for a stack destination, to identify the frame/function that owns
  // the dialogue buffer. (x64-only: needs the backend's per-frame guest r1.)
  if (cvars::text_hook_dump_stack) {
    static std::mutex sp_mutex;
    static std::unordered_set<uint32_t> sp_seen;
    const uint32_t sp_dest = static_cast<uint32_t>(ctx->r[3]);
    const uint32_t sp_now = static_cast<uint32_t>(ctx->r[1]);
    bool sp_dump = false;
    if (sp_dest >= sp_now) {
      std::lock_guard<std::mutex> lock(sp_mutex);
      if (sp_seen.size() < 16 && sp_seen.insert(sp_dest).second) {
        sp_dump = true;
      }
    }
    if (sp_dump) {
      auto* be = static_cast<X64Backend*>(ctx->processor->backend());
      X64BackendContext* bctx = be->BackendContextForGuestContext(ctx);
      XELOGI("[sp] dest=0x{:08X} r1=0x{:08X} depth={}", sp_dest, sp_now,
             bctx->current_stackpoint_depth);
      for (uint32_t i = 0; i < bctx->current_stackpoint_depth && i < 48; ++i) {
        const uint32_t r1 = bctx->stackpoints[i].guest_stack_;
        const uint32_t ret = bctx->stackpoints[i].guest_return_address_;
        Function* fn = ctx->processor->LookupFunction(ret);
        XELOGI("[sp]   #{:02} r1=0x{:08X} ret=0x{:08X} func=0x{:08X}{}", i, r1,
               ret, fn ? fn->address() : 0u, (r1 <= sp_dest) ? "  <=dest" : "");
      }
    }
  }

  // One-shot discovery: for a stack destination buffer, identify the function
  // that owns its frame and the buffer's offset within that frame, then print a
  // ready-to-use text_hook_buffer_map entry. The owner is the deepest frame
  // whose r1 is still at/below the buffer; its return address (the resume point
  // inside the owner, right after the call it made) resolves back to it.
  if (cvars::text_hook_buffer_discover) {
    const uint32_t disc_dest = static_cast<uint32_t>(ctx->r[3]);
    const uint32_t disc_sp = static_cast<uint32_t>(ctx->r[1]);
    if (disc_dest >= disc_sp) {
      auto* be = static_cast<X64Backend*>(ctx->processor->backend());
      X64BackendContext* bctx = be->BackendContextForGuestContext(ctx);
      uint32_t frame_base = 0;  // largest stackpoint r1 that is <= dest.
      uint32_t owner_ret = 0;
      for (uint32_t i = 0; i < bctx->current_stackpoint_depth; ++i) {
        const uint32_t r1 = bctx->stackpoints[i].guest_stack_;
        if (r1 <= disc_dest && r1 >= frame_base) {
          frame_base = r1;
          owner_ret = bctx->stackpoints[i].guest_return_address_;
        }
      }
      if (frame_base && owner_ret) {
        Function* owner = ctx->processor->LookupFunction(owner_ret);
        const uint32_t owner_addr = owner ? owner->address() : 0u;
        const uint32_t offset = disc_dest - frame_base;
        const uint64_t key = (static_cast<uint64_t>(owner_addr) << 32) | offset;
        static std::mutex disc_mutex;
        static std::unordered_set<uint64_t> disc_seen;
        bool log_it = false;
        {
          std::lock_guard<std::mutex> lock(disc_mutex);
          if (disc_seen.size() < 64 && disc_seen.insert(key).second) {
            log_it = true;
          }
        }
        if (log_it && owner_addr) {
          XELOGI(
              "[bufdisc] owner=0x{:08X} offset=0x{:X} dest=0x{:08X} "
              "frame=0x{:08X}  ->  text_hook_buffer_map += {:08X}:{:X}",
              owner_addr, offset, disc_dest, frame_base, owner_addr, offset);
        }
      }
    }
  }
#endif  // XE_ARCH_AMD64

  // --- Output the translation/replacement ---
  const OutputMode out = ParseOutputMode();
  // Queue a translation so clipboard:en/both can pick it up once it's ready; the
  // worker fills the cache asynchronously and TextHookPublishClipboard reads it
  // there. Only Japanese lines are queued (a hooked general-purpose routine sees
  // a lot of non-JP text that would otherwise flood the queue).
  if (cvars::text_translate && HasKana(text)) {
    std::u16string en;
    TextTranslateLookup(text, &en);
  }

  // Export the active on-screen lines to the clipboard.
  if (out.clipboard) {
    TextHookPublishClipboard(text, out.clipboard, hook->cooldown_ms);
  }

  return 0;
}

// --- Registration -----------------------------------------------------------

static bool ShouldArgScanHook(uint32_t /*guest_address*/) {
  return cvars::jit_argscan_hi > cvars::jit_argscan_lo;
}

static bool ShouldTextHook(uint32_t guest_address) {
  return FindHook(guest_address) != nullptr;
}

void Initialize(const std::filesystem::path& config_folder) {
  using xe::cpu::backend::RegisterGuestPrologHook;
  LoadHooks(config_folder);
  RegisterGuestPrologHook({&ShouldArgScanHook, &JitArgScanHook});
  RegisterGuestPrologHook({&ShouldTextHook, &TextHookCallback});
}

}  // namespace loc
}  // namespace xe
