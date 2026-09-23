// Stream a (gzip-compressed) extended XYZ file into metatomic systems.
//
// Every frame is parsed straight from the compressed stream with zlib (plain
// .extxyz works too, gzopen reads uncompressed files transparently), turned
// into an `mta_system_t` through the metatomic C API, and optionally saved to
// the metatomic `.mta` format. Positions, cell and pbc come from the standard
// extxyz fields (`pos`, `Lattice`, `pbc`); everything else is ignored.
//
// usage: extxyz-gz-to-mta <input.extxyz[.gz]> [--first N] [--every K] [--save DIR]
//
//   --first N   stop after N frames (default: read everything)
//   --every K   save every K-th frame (default: 0 = save nothing)
//   --save DIR  directory for the saved frames, named <frame index>.mta
//
// It prints the number of frames and atoms read and the throughput, which is
// what matters when streaming datasets with millions of frames.

#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <zlib.h>

#include <metatomic.h>
#include <metatensor/dlpack/dlpack.h>

// DLPack tensors that own their data: the deleter frees data, shape and
// strides, so a system keeps its arrays alive after the reader moves on.
typedef struct {
    int64_t shape[2];
    int64_t strides[2];
    void* data;
} OwnedContext;

static void owned_deleter(DLManagedTensorVersioned* self) {
    if (self) {
        OwnedContext* ctx = self->manager_ctx;
        free(ctx->data);
        free(ctx);
        free(self);
    }
}

static DLManagedTensorVersioned* owned_tensor(void* data, int32_t ndim, const int64_t* shape, DLDataType dtype) {
    OwnedContext* ctx = calloc(1, sizeof(OwnedContext));
    DLManagedTensorVersioned* tensor = calloc(1, sizeof(DLManagedTensorVersioned));
    if (!ctx || !tensor) {
        free(ctx), free(tensor), free(data);
        return NULL;
    }
    int64_t stride = 1;
    for (int32_t i = ndim - 1; i >= 0; i--) {
        ctx->shape[i] = shape[i];
        ctx->strides[i] = stride;
        stride *= shape[i];
    }
    ctx->data = data;
    tensor->version.major = DLPACK_MAJOR_VERSION;
    tensor->version.minor = DLPACK_MINOR_VERSION;
    tensor->manager_ctx = ctx;
    tensor->deleter = owned_deleter;
    tensor->dl_tensor = (DLTensor){
        .data = data,
        .device = {kDLCPU, 0},
        .ndim = ndim,
        .dtype = dtype,
        .shape = ctx->shape,
        .strides = ctx->strides,
        .byte_offset = 0,
    };
    return tensor;
}

static const DLDataType F64 = {kDLFloat, 64, 1};
static const DLDataType I32 = {kDLInt, 32, 1};
static const DLDataType BOOL = {kDLBool, 8, 1};

static const char* ELEMENTS[] = {
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
    "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
    "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
};

static int32_t atomic_number(const char* symbol) {
    for (int32_t z = 1; z < (int32_t)(sizeof(ELEMENTS) / sizeof(*ELEMENTS)); z++) {
        if (strcmp(symbol, ELEMENTS[z]) == 0) {
            return z;
        }
    }
    return -1;
}

// Find `key=` in an extxyz comment line and return a pointer to its value
// (after the opening quote, if any). Only matches at the start of a token.
static const char* header_value(const char* line, const char* key) {
    size_t len = strlen(key);
    for (const char* p = strstr(line, key); p; p = strstr(p + 1, key)) {
        if ((p == line || p[-1] == ' ') && p[len] == '=') {
            p += len + 1;
            return *p == '"' ? p + 1 : p;
        }
    }
    return NULL;
}

// Column of the positions in the per-atom lines, from `Properties=...`.
static int positions_column(const char* line) {
    const char* p = header_value(line, "Properties");
    int column = 0;
    char name[64];
    int count;
    while (p && sscanf(p, "%63[^:]:%*[^:]:%d", name, &count) == 2) {
        if (strcmp(name, "pos") == 0) {
            return column;
        }
        column += count;
        for (int colons = 0; *p && colons < 3; p++) {
            colons += *p == ':';
        }
    }
    return -1;
}

static void check(mta_status_t status, const char* what) {
    if (status != MTA_SUCCESS) {
        const char* message = NULL;
        mta_last_error(&message, NULL, NULL);
        fprintf(stderr, "error: %s: %s\n", what, message);
        exit(EXIT_FAILURE);
    }
}

// Read one frame and create the corresponding system. Returns NULL at the end
// of the file.
static mta_system_t* read_frame(gzFile file, char* line, int line_size, size_t* n_atoms_out) {
    if (!gzgets(file, line, line_size)) {
        return NULL;
    }
    long n_atoms = strtol(line, NULL, 10);
    if (n_atoms <= 0 || !gzgets(file, line, line_size)) {
        fprintf(stderr, "error: malformed frame header\n");
        exit(EXIT_FAILURE);
    }

    double* cell = calloc(9, sizeof(double));
    bool* pbc = calloc(3, sizeof(bool));
    const char* lattice = header_value(line, "Lattice");
    if (lattice) {
        sscanf(lattice, "%lf %lf %lf %lf %lf %lf %lf %lf %lf", &cell[0], &cell[1], &cell[2], &cell[3], &cell[4],
               &cell[5], &cell[6], &cell[7], &cell[8]);
    }
    const char* pbc_value = header_value(line, "pbc");
    for (int i = 0; pbc_value && i < 3; i++) {
        pbc[i] = pbc_value[2 * i] == 'T';
    }
    for (int i = 0; i < 3; i++) {
        if (!pbc[i]) {  // metatomic requires a zero cell vector along non-periodic axes
            cell[3 * i] = cell[3 * i + 1] = cell[3 * i + 2] = 0.0;
        }
    }
    int column = positions_column(line);
    if (column < 1) {
        fprintf(stderr, "error: expected species before pos in Properties\n");
        exit(EXIT_FAILURE);
    }

    int32_t* types = malloc(n_atoms * sizeof(int32_t));
    double* positions = malloc(3 * n_atoms * sizeof(double));
    for (long i = 0; i < n_atoms; i++) {
        char symbol[8];
        if (!gzgets(file, line, line_size) || sscanf(line, "%7s", symbol) != 1) {
            fprintf(stderr, "error: truncated frame\n");
            exit(EXIT_FAILURE);
        }
        types[i] = atomic_number(symbol);
        char* p = line;
        for (int c = 0; c < column; c++) {  // skip to the pos columns
            while (*p == ' ') p++;
            while (*p && *p != ' ') p++;
        }
        for (int k = 0; k < 3; k++) {
            positions[3 * i + k] = strtod(p, &p);
        }
    }

    mta_system_t* system = NULL;
    check(mta_system_create(
              "angstrom",
              owned_tensor(types, 1, (int64_t[]){n_atoms}, I32),
              owned_tensor(positions, 2, (int64_t[]){n_atoms, 3}, F64),
              owned_tensor(cell, 2, (int64_t[]){3, 3}, F64),
              owned_tensor(pbc, 1, (int64_t[]){3}, BOOL),
              &system
          ),
          "mta_system_create");
    *n_atoms_out = (size_t)n_atoms;
    return system;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input.extxyz[.gz]> [--first N] [--every K] [--save DIR]\n", argv[0]);
        return EXIT_FAILURE;
    }
    long first = -1, every = 0;
    const char* save_dir = ".";
    for (int i = 2; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--first") == 0) first = atol(argv[i + 1]);
        else if (strcmp(argv[i], "--every") == 0) every = atol(argv[i + 1]);
        else if (strcmp(argv[i], "--save") == 0) save_dir = argv[i + 1];
    }

    gzFile file = gzopen(argv[1], "rb");
    if (!file) {
        fprintf(stderr, "error: could not open %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    gzbuffer(file, 1 << 20);

    static char line[1 << 16];
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    long frames = 0, saved = 0;
    size_t atoms = 0, n_atoms = 0;
    mta_system_t* system;
    while ((first < 0 || frames < first) && (system = read_frame(file, line, sizeof(line), &n_atoms))) {
        if (every > 0 && frames % every == 0) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/%ld.mta", save_dir, frames);
            check(mta_save(path, system), "mta_save");
            saved++;
        }
        mta_system_free(system);
        frames++, atoms += n_atoms;
    }
    gzclose(file);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double seconds = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    printf("%ld frames, %zu atoms in %.2f s (%.0f frames/s), saved %ld systems to %s\n", frames, atoms, seconds,
           frames / seconds, saved, save_dir);
    return EXIT_SUCCESS;
}
