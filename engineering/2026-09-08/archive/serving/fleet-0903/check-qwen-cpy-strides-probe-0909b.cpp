#include "ops.h"
#include "ggml-cpu-impl.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    constexpr int64_t columns = 262144, snapshots = 5, guard = 16;
    constexpr float sentinel = -717.125f;
    int cases = 0;
    for (int mode = 0; mode < 3; ++mode) {
        const int64_t src_stride = columns + (mode == 1 ? 16 : 0);
        const int64_t dst_stride = columns + (mode == 2 ? 32 : 0);
        std::vector<float> input(2 * guard + src_stride * snapshots, sentinel);
        std::vector<float> output(2 * guard + dst_stride * snapshots, sentinel);
        std::vector<float> expected = output;
        for (int64_t row = 0; row < snapshots; ++row) {
            for (int64_t col = 0; col < columns; ++col) {
                float value = float((row * columns + col) % 1024 - 512);
                input[guard + row * src_stride + col] = value;
                expected[guard + row * dst_stride + col] = value;
            }
        }
        const auto original = input;
        ggml_init_params init{1 << 20, nullptr, true};
        ggml_context * ctx = ggml_init(init);
        GGML_ASSERT(ctx);
        ggml_tensor * src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, columns, 1, snapshots);
        ggml_tensor * dst = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, columns, 1, snapshots);
        src->data = input.data() + guard;
        dst->data = output.data() + guard;
        src->nb[2] = src_stride * sizeof(float);
        src->nb[3] = src->nb[2] * snapshots;
        dst->nb[2] = dst_stride * sizeof(float);
        dst->nb[3] = dst->nb[2] * snapshots;
        dst->src[0] = src;
        for (int ith = 0; ith < 15; ++ith) {
            ggml_compute_params params{};
            params.ith = ith;
            params.nth = 15;
            ggml_compute_forward_cpy(&params, dst);
        }
        GGML_ASSERT(std::memcmp(output.data(), expected.data(), output.size() * sizeof(float)) == 0);
        GGML_ASSERT(std::memcmp(input.data(), original.data(), input.size() * sizeof(float)) == 0);
        ++cases;
        ggml_free(ctx);
    }
    std::printf("CPY_PROBE_CHECK {\"cases\":%d,\"exact\":true,\"guards_preserved\":true,\"inputs_preserved\":true}\n", cases);
}
