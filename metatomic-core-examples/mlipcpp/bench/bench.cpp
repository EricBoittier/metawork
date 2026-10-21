// Time a GGUF model through mlipcpp::Predictor directly and through the
// metatomic plugin (mta_execute_model), on diamond silicon supercells.
//
//   bench <plugin.so> <model.gguf> [cells...]
//
// Prints one CSV line per (wrapper, n_atoms, mode): mean milliseconds per call.
// MLIPCPP_BACKEND=cpu|cuda|... selects the ggml backend (default cpu).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <metatensor.h>
#include <metatensor/dlpack/dlpack.h>
#include <metatomic.h>

#include "mlipcpp/mlipcpp.hpp"

namespace {

void check(mta_status_t status, const char *what) {
    if (status != MTA_SUCCESS) {
        const char *message = nullptr;
        mta_last_error(&message, nullptr, nullptr);
        std::fprintf(stderr, "error: %s: %s\n", what, message);
        std::exit(EXIT_FAILURE);
    }
}

struct Structure {
    std::vector<int32_t> types;
    std::vector<double> positions;
    double cell[9] = {};
};

Structure silicon(int n) {
    const double a = 5.43, basis[8][3] = {{0, 0, 0}, {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0},
                                          {.25, .25, .25}, {.25, .75, .75}, {.75, .25, .75}, {.75, .75, .25}};
    auto s = Structure();
    std::srand(7);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                for (auto &b : basis) {
                    s.types.push_back(14);
                    for (int d = 0; d < 3; d++) {
                        auto shift = d == 0 ? i : d == 1 ? j : k;
                        s.positions.push_back(a * (shift + b[d]) + 0.05 * (std::rand() / double(RAND_MAX) - 0.5));
                    }
                }
    for (int d = 0; d < 3; d++) s.cell[4 * d] = a * n;
    return s;
}

DLManagedTensorVersioned *tensor(const void *values, size_t size, std::vector<int64_t> shape, DLDataType dtype) {
    struct Context {
        std::vector<int64_t> shape, strides;
        std::vector<char> data;
    };
    auto *ctx = new Context{shape, std::vector<int64_t>(shape.size(), 1),
                            std::vector<char>((const char *)values, (const char *)values + size)};
    for (int i = int(shape.size()) - 2; i >= 0; i--) ctx->strides[i] = ctx->strides[i + 1] * shape[i + 1];
    auto *t = new DLManagedTensorVersioned{};
    t->version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    t->manager_ctx = ctx;
    t->deleter = [](DLManagedTensorVersioned *self) {
        delete static_cast<Context *>(self->manager_ctx);
        delete self;
    };
    t->dl_tensor = DLTensor{ctx->data.data(), {kDLCPU, 0}, int32_t(shape.size()), dtype,
                            ctx->shape.data(), ctx->strides.data(), 0};
    return t;
}

mta_system_t *make_system(const Structure &s) {
    auto n = int64_t(s.types.size());
    const bool pbc[3] = {true, true, true};
    mta_system_t *system = nullptr;
    check(mta_system_create("angstrom", tensor(s.types.data(), 4 * n, {n}, {kDLInt, 32, 1}),
                            tensor(s.positions.data(), 8 * 3 * n, {n, 3}, {kDLFloat, 64, 1}),
                            tensor(s.cell, sizeof(s.cell), {3, 3}, {kDLFloat, 64, 1}),
                            tensor(pbc, sizeof(pbc), {3}, {kDLBool, 8, 1}), &system),
          "mta_system_create");
    return system;
}

const char *REQUESTS[] = {
    // energy
    R"([{"type": "metatomic_quantity", "name": "energy", "unit": "eV", "sample_kind": "system", "gradients": []}])",
    // forces
    R"([{"type": "metatomic_quantity", "name": "energy", "unit": "eV", "sample_kind": "system", "gradients": ["positions"]}])",
    // nc
    R"([{"type": "metatomic_quantity", "name": "energy", "unit": "eV", "sample_kind": "system", "gradients": []},
        {"type": "metatomic_quantity", "name": "non_conservative_force", "unit": "eV/Angstrom", "sample_kind": "atom", "gradients": []}])",
};
const char *MODES[] = {"energy", "forces", "nc"};

double time_ms(const std::function<void()> &run) {
    for (int i = 0; i < 3; i++) run(); // warmup (and TorchScript profiling runs)
    auto start = std::chrono::steady_clock::now();
    int calls = 0;
    double elapsed = 0.0;
    while (calls < 3 || (elapsed < 2000.0 && calls < 50)) {
        run();
        calls++;
        elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    return elapsed / calls;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <plugin.so> <model.gguf> [cells...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    auto cells = std::vector<int>();
    for (int i = 3; i < argc; i++) cells.push_back(std::atoi(argv[i]));
    if (cells.empty()) cells = {1, 2, 3, 4};

    auto backend_name = std::string(std::getenv("MLIPCPP_BACKEND") ? std::getenv("MLIPCPP_BACKEND") : "cpu");
    auto backend = backend_name == "cuda" ? mlipcpp::Backend::CUDA : mlipcpp::Backend::CPU;
    auto label = "ggml-" + backend_name;

    mlipcpp::suppress_logging();
    auto predictor = mlipcpp::Predictor(argv[2], {backend});
    check(mta_load_plugin(argv[1]), "mta_load_plugin");
    mta_model_t model = {};
    auto options = R"({"backend": ")" + backend_name + R"("})";
    check(mta_load_model(argv[2], options.c_str(), nullptr, &model), "mta_load_model");

    auto threads = std::getenv("MLIPCPP_NUM_THREADS");
    for (auto n : cells) {
        auto s = silicon(n);
        auto n_atoms = int32_t(s.types.size());
        auto positions = std::vector<float>(s.positions.begin(), s.positions.end());
        auto cell = std::vector<float>(s.cell, s.cell + 9);
        const bool pbc[3] = {true, true, true};

        for (int mode = 0; mode < 3; mode++) {
            auto options = mlipcpp::PredictOptions();
            options.compute_forces = mode != 0;
            options.use_nc_forces = mode == 2;
            auto direct = time_ms([&] {
                predictor.predict(n_atoms, positions.data(), s.types.data(), cell.data(), pbc, options);
            });

            auto plugin = time_ms([&] {
                auto *system = make_system(s);
                mts_tensormap_t *outputs[2] = {nullptr, nullptr};
                auto count = mode == 2 ? 2u : 1u;
                check(mta_execute_model(model, &system, 1, nullptr, REQUESTS[mode], false, outputs, count),
                      "mta_execute_model");
                for (unsigned i = 0; i < count; i++) mts_tensormap_free(outputs[i]);
                mta_system_free(system);
            });
            std::printf("mlipcpp-direct,%s,%s,%d,%s,%.2f\n", label.c_str(), threads ? threads : "4", n_atoms,
                        MODES[mode], direct);
            std::printf("plugin-c-api,%s,%s,%d,%s,%.2f\n", label.c_str(), threads ? threads : "4", n_atoms,
                        MODES[mode], plugin);
            std::fflush(stdout);
        }
    }
    check(model.unload(model.data), "unload");
}
