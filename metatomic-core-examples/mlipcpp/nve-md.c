// A minimal MD engine on the metatomic C API: velocity Verlet for bulk silicon
// with a model loaded from a plugin. Nothing here is specific to mlipcpp,
// except the plugin and model paths given on the command line.
//
//   nve-md <plugin.so> <model> [--nc] [--steps N] [--dt FS] [--temperature K]
//
// With --nc the forces come from the model's "non_conservative_force" output
// (net force removed), otherwise from the gradient of the energy. Prints the
// potential, kinetic and total energy; the drift of the total energy is a
// direct check of the conservative forces.
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatensor.h>
#include <metatensor/dlpack/dlpack.h>
#include <metatomic.h>

#define N_CELLS 2
#define N_ATOMS (8 * N_CELLS * N_CELLS * N_CELLS)

static const double LATTICE = 5.43;          // Angstrom
static const double MASS = 28.0855;          // amu
static const double ACCELERATION = 9.648533e-3; // eV/(Angstrom amu) in Angstrom/fs^2
static const double BOLTZMANN = 8.617333e-5; // eV/K

static void check(mta_status_t status, const char *what) {
    if (status != MTA_SUCCESS) {
        const char *message = NULL;
        mta_last_error(&message, NULL, NULL);
        fprintf(stderr, "error: %s: %s\n", what, message);
        exit(EXIT_FAILURE);
    }
}

static void check_mts(mts_status_t status, const char *what) {
    if (status != MTS_SUCCESS) {
        const char *message = NULL;
        mts_last_error(&message, NULL, NULL);
        fprintf(stderr, "error: %s: %s\n", what, message);
        exit(EXIT_FAILURE);
    }
}

// DLPack tensor owning a copy of `values`, freed with the system
typedef struct {
    int64_t shape[2];
    int64_t strides[2];
} Layout;

static void owned_deleter(DLManagedTensorVersioned *self) {
    free(self->dl_tensor.data);
    free(self->manager_ctx);
    free(self);
}

static DLManagedTensorVersioned *owned_tensor(const void *values, size_t size, int32_t ndim,
                                              const int64_t *shape, DLDataType dtype) {
    Layout *layout = calloc(1, sizeof(Layout));
    DLManagedTensorVersioned *tensor = calloc(1, sizeof(DLManagedTensorVersioned));
    void *data = malloc(size);
    memcpy(data, values, size);
    int64_t stride = 1;
    for (int32_t i = ndim - 1; i >= 0; i--) {
        layout->shape[i] = shape[i];
        layout->strides[i] = stride;
        stride *= shape[i];
    }
    tensor->version = (DLPackVersion){DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    tensor->manager_ctx = layout;
    tensor->deleter = owned_deleter;
    tensor->dl_tensor = (DLTensor){
        .data = data,
        .device = {kDLCPU, 0},
        .ndim = ndim,
        .dtype = dtype,
        .shape = layout->shape,
        .strides = layout->strides,
    };
    return tensor;
}

// Copy float64 values of a block, or of one of its gradients, to `out`
static void block_values(mts_tensormap_t *tensor, const char *gradient, double *out, size_t count) {
    mts_block_t *block = NULL;
    check_mts(mts_tensormap_block_by_id(tensor, &block, 0), "mts_tensormap_block_by_id");
    if (gradient != NULL) {
        check_mts(mts_block_gradient(block, gradient, &block), "mts_block_gradient");
    }
    mts_array_t array = {0};
    check_mts(mts_block_data(block, &array), "mts_block_data");
    DLManagedTensorVersioned *view = NULL;
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    check_mts(array.as_dlpack(array.ptr, &view, (DLDevice){kDLCPU, 0}, NULL, version), "as_dlpack");
    if (array.destroy != NULL) {
        array.destroy(array.ptr);
    }
    memcpy(out, (char *)view->dl_tensor.data + view->dl_tensor.byte_offset, count * sizeof(double));
    view->deleter(view);
}

// Energy and forces for the current positions, through mta_execute_model
static double compute(mta_model_t model, const double *positions, const double *cell, bool nc,
                      double *forces) {
    int32_t types[N_ATOMS];
    for (int i = 0; i < N_ATOMS; i++) {
        types[i] = 14;
    }
    const bool pbc[3] = {true, true, true};
    const DLDataType F64 = {kDLFloat, 64, 1};

    mta_system_t *system = NULL;
    check(mta_system_create(
              "angstrom",
              owned_tensor(types, sizeof(types), 1, (int64_t[]){N_ATOMS}, (DLDataType){kDLInt, 32, 1}),
              owned_tensor(positions, sizeof(double) * 3 * N_ATOMS, 2, (int64_t[]){N_ATOMS, 3}, F64),
              owned_tensor(cell, sizeof(double) * 9, 2, (int64_t[]){3, 3}, F64),
              owned_tensor(pbc, sizeof(pbc), 1, (int64_t[]){3}, (DLDataType){kDLBool, 8, 1}),
              &system),
          "mta_system_create");

    const char *requests =
        nc ? "[{\"type\": \"metatomic_quantity\", \"name\": \"energy\", \"unit\": \"eV\", "
             "\"sample_kind\": \"system\", \"gradients\": []},"
             " {\"type\": \"metatomic_quantity\", \"name\": \"non_conservative_force\", "
             "\"unit\": \"eV/Angstrom\", \"sample_kind\": \"atom\", \"gradients\": []}]"
           : "[{\"type\": \"metatomic_quantity\", \"name\": \"energy\", \"unit\": \"eV\", "
             "\"sample_kind\": \"system\", \"gradients\": [\"positions\"]}]";
    uintptr_t n_outputs = nc ? 2 : 1;
    mts_tensormap_t *outputs[2] = {NULL, NULL};
    check(mta_execute_model(model, (const mta_system_t *const *)&system, 1, NULL, requests, false,
                            outputs, n_outputs),
          "mta_execute_model");

    double energy = 0.0;
    block_values(outputs[0], NULL, &energy, 1);
    if (nc) {
        block_values(outputs[1], NULL, forces, 3 * N_ATOMS);
        // like metatomic's ASE calculator, remove any spurious net force
        for (int k = 0; k < 3; k++) {
            double mean = 0.0;
            for (int i = 0; i < N_ATOMS; i++) mean += forces[3 * i + k] / N_ATOMS;
            for (int i = 0; i < N_ATOMS; i++) forces[3 * i + k] -= mean;
        }
    } else {
        block_values(outputs[0], "positions", forces, 3 * N_ATOMS);
        for (int i = 0; i < 3 * N_ATOMS; i++) forces[i] = -forces[i];
    }

    for (uintptr_t i = 0; i < n_outputs; i++) {
        mts_tensormap_free(outputs[i]);
    }
    mta_system_free(system);
    return energy;
}

static double gaussian(void) {
    double u = (rand() + 1.0) / (RAND_MAX + 2.0), v = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

static double kinetic(const double *velocities) {
    double sum = 0.0;
    for (int i = 0; i < 3 * N_ATOMS; i++) sum += velocities[i] * velocities[i];
    return 0.5 * MASS * sum / ACCELERATION;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <plugin.so> <model> [--nc] [--steps N] [--dt FS] [--temperature K]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    bool nc = false;
    int steps = 500;
    double dt = 1.0, temperature = 300.0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--nc") == 0) nc = true;
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dt") == 0 && i + 1 < argc) dt = atof(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = atof(argv[++i]);
        else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }

    check(mta_load_plugin(argv[1]), "mta_load_plugin");
    mta_model_t model = {0};
    check(mta_load_model(argv[2], NULL, NULL, &model), "mta_load_model");

    // 2x2x2 cubic diamond silicon, velocities at `temperature`
    static const double BASIS[8][3] = {
        {0.00, 0.00, 0.00}, {0.00, 0.50, 0.50}, {0.50, 0.00, 0.50}, {0.50, 0.50, 0.00},
        {0.25, 0.25, 0.25}, {0.25, 0.75, 0.75}, {0.75, 0.25, 0.75}, {0.75, 0.75, 0.25},
    };
    double positions[3 * N_ATOMS], velocities[3 * N_ATOMS], forces[3 * N_ATOMS];
    double cell[9] = {0};
    for (int k = 0; k < 3; k++) cell[4 * k] = N_CELLS * LATTICE;
    int atom = 0;
    for (int a = 0; a < N_CELLS; a++)
        for (int b = 0; b < N_CELLS; b++)
            for (int c = 0; c < N_CELLS; c++)
                for (int s = 0; s < 8; s++, atom++) {
                    positions[3 * atom + 0] = LATTICE * (a + BASIS[s][0]);
                    positions[3 * atom + 1] = LATTICE * (b + BASIS[s][1]);
                    positions[3 * atom + 2] = LATTICE * (c + BASIS[s][2]);
                }

    srand(1234);
    double sigma = sqrt(BOLTZMANN * temperature / MASS * ACCELERATION); // Angstrom/fs
    for (int i = 0; i < 3 * N_ATOMS; i++) velocities[i] = sigma * gaussian();
    for (int k = 0; k < 3; k++) {
        double mean = 0.0;
        for (int i = 0; i < N_ATOMS; i++) mean += velocities[3 * i + k] / N_ATOMS;
        for (int i = 0; i < N_ATOMS; i++) velocities[3 * i + k] -= mean;
    }

    printf("# %d Si atoms, dt = %.2f fs, %s forces\n", N_ATOMS, dt, nc ? "non-conservative" : "conservative");
    printf("# %6s %14s %12s %14s %9s\n", "step", "E_pot (eV)", "E_kin (eV)", "E_tot (eV)", "T (K)");

    double potential = compute(model, positions, cell, nc, forces);
    double start = potential + kinetic(velocities);
    for (int step = 0; step <= steps; step++) {
        if (step % (steps / 10 > 0 ? steps / 10 : 1) == 0) {
            double k = kinetic(velocities);
            printf("  %6d %14.6f %12.6f %14.6f %9.1f\n", step, potential, k, potential + k,
                   2.0 * k / (3.0 * (N_ATOMS - 1) * BOLTZMANN));
        }
        if (step == steps) break;
        for (int i = 0; i < 3 * N_ATOMS; i++) {
            velocities[i] += 0.5 * dt * forces[i] / MASS * ACCELERATION;
            positions[i] += dt * velocities[i];
        }
        potential = compute(model, positions, cell, nc, forces);
        for (int i = 0; i < 3 * N_ATOMS; i++) {
            velocities[i] += 0.5 * dt * forces[i] / MASS * ACCELERATION;
        }
    }

    double drift = potential + kinetic(velocities) - start;
    printf("# total energy drift: %.3e eV (%.3e eV/atom)\n", drift, drift / N_ATOMS);
    check(model.unload(model.data), "unload");
    return EXIT_SUCCESS;
}
