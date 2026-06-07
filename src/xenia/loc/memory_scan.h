/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_LOC_MEMORY_SCAN_H_
#define XENIA_LOC_MEMORY_SCAN_H_

namespace xe {
class Memory;
namespace cpu {
class Processor;
}  // namespace cpu
}  // namespace xe

namespace xe {
namespace loc {

// Scans guest memory for the cvar `scan_text` (encoded into several candidate
// encodings) and the `scan_pointer` big-endian pointer, logging every hit.
// Debugging aid for locating game text buffers (bound to Scroll Lock).
void ScanGuestMemoryForText(Memory* memory);

// Drains a pending memory-watch fault captured by the watch handler, logging
// the faulting guest function. Returns true if a result was consumed.
bool DumpMemoryWatchResult(Memory* memory, cpu::Processor* processor);

// Arms a read/write watch on the page of the cvar `watch_address` so the next
// guest access to it is logged (bound to Home).
void ArmMemoryWatch(Memory* memory, cpu::Processor* processor);

}  // namespace loc
}  // namespace xe

#endif  // XENIA_LOC_MEMORY_SCAN_H_
