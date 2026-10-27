// SPDX-License-Identifier: Apache-2.0
#include <stddef.h>

// Freestanding compiler support for metadata copies. Consumers may supply
// their own implementation; examples do not need to define libc shims.
__attribute__((weak)) void *memcpy(void *dst, const void *src, size_t count)
{
    unsigned char *out = dst;
    const unsigned char *in = src;
    for (size_t i = 0; i < count; ++i) out[i] = in[i];
    return dst;
}
