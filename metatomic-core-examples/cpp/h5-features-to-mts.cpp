// Convert rows of an HDF5 feature file into metatensor / metatomic data.
//
// MAD-CORE ships its selection features as `madcore_features_X-Y.h5`, with a
// float32 `features` dataset of shape (rows, 512), a `mean` (512,) and a
// scalar `scale`; the selection metric uses `(features - mean) / scale`. This
// example reads a range of rows with the HDF5 C API, stores the scaled
// features as a `metatensor::TensorMap` (samples `system` = global row,
// properties `feature`) and saves it as `.mts`. With `--attach`, it also
// loads a system saved as `.mta` (e.g. by extxyz-gz-to-mta), attaches the
// features of its row as the custom data `madcore::features`, and saves it
// again, so the structure and its descriptor travel together.
//
// usage: h5-features-to-mts <features.h5> <output.mts> [--rows START:STOP] [--offset ROW0]
//                           [--attach SYSTEM.mta ROW OUTPUT.mta]
//
//   --rows START:STOP  rows of the file to convert (default 0:1024)
//   --offset ROW0      global row of the first row in the file, used for the
//                      `system` sample labels (default 0; e.g. 262144 for
//                      madcore_features_2pow18-2pow19.h5)

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>

#include <metatensor.hpp>
#include <metatomic.hpp>

struct Features {
    std::vector<double> values;  // (n_rows, n_features), scaled
    size_t n_rows = 0;
    size_t n_features = 0;
};

static void h5_check(herr_t status, const char* what) {
    if (status < 0) {
        throw std::runtime_error(std::string("HDF5 error in ") + what);
    }
}

static Features read_features(const std::string& path, hsize_t start, hsize_t stop) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("could not open " + path);
    }

    hid_t dataset = H5Dopen2(file, "features", H5P_DEFAULT);
    hid_t space = H5Dget_space(dataset);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    stop = std::min(stop, dims[0]);
    if (start >= stop) {
        throw std::runtime_error("empty row range");
    }

    // read only the requested rows: a hyperslab in the file, a dense buffer in memory
    hsize_t offset[2] = {start, 0}, count[2] = {stop - start, dims[1]};
    h5_check(H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, nullptr, count, nullptr), "select_hyperslab");
    hid_t memory = H5Screate_simple(2, count, nullptr);
    std::vector<float> raw(count[0] * count[1]);
    h5_check(H5Dread(dataset, H5T_NATIVE_FLOAT, memory, space, H5P_DEFAULT, raw.data()), "read features");
    H5Sclose(memory), H5Sclose(space), H5Dclose(dataset);

    std::vector<double> mean(dims[1]);
    hid_t mean_set = H5Dopen2(file, "mean", H5P_DEFAULT);
    h5_check(H5Dread(mean_set, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mean.data()), "read mean");
    H5Dclose(mean_set);

    double scale = 1.0;
    hid_t scale_set = H5Dopen2(file, "scale", H5P_DEFAULT);
    h5_check(H5Dread(scale_set, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &scale), "read scale");
    H5Dclose(scale_set), H5Fclose(file);

    Features features{std::vector<double>(raw.size()), count[0], count[1]};
    for (size_t i = 0; i < raw.size(); i++) {
        features.values[i] = (raw[i] - mean[i % count[1]]) / scale;
    }
    return features;
}

static metatensor::TensorMap to_tensormap(const Features& features, size_t first_row) {
    auto n_rows = features.n_rows;
    auto samples = std::vector<int32_t>(n_rows);
    for (size_t i = 0; i < n_rows; i++) {
        samples[i] = static_cast<int32_t>(first_row + i);
    }
    auto properties = std::vector<int32_t>(features.n_features);
    for (size_t j = 0; j < features.n_features; j++) {
        properties[j] = static_cast<int32_t>(j);
    }

    auto blocks = std::vector<metatensor::TensorBlock>();  // TensorBlock is move-only
    blocks.push_back(metatensor::TensorBlock(
        std::make_unique<metatensor::SimpleDataArray<double>>(
            std::vector<uintptr_t>{n_rows, features.n_features},
            std::vector<double>(features.values.begin(), features.values.begin() + n_rows * features.n_features)
        ),
        metatensor::Labels({"system"}, samples.data(), samples.size()),
        {},
        metatensor::Labels({"feature"}, properties.data(), properties.size())
    ));
    return metatensor::TensorMap(metatensor::Labels({"_"}, {{0}}), std::move(blocks));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <features.h5> <output.mts> [--rows START:STOP] [--offset ROW0] "
                             "[--attach SYSTEM.mta ROW OUTPUT.mta]\n", argv[0]);
        return 1;
    }
    hsize_t start = 0, stop = 1024;
    size_t offset = 0;
    const char *attach_in = nullptr, *attach_out = nullptr;
    size_t attach_row = 0;
    for (int i = 3; i < argc; i++) {
        auto arg = std::string(argv[i]);
        if (arg == "--rows" && i + 1 < argc) {
            std::sscanf(argv[++i], "%llu:%llu", reinterpret_cast<unsigned long long*>(&start),
                        reinterpret_cast<unsigned long long*>(&stop));
        } else if (arg == "--offset" && i + 1 < argc) {
            offset = std::stoull(argv[++i]);
        } else if (arg == "--attach" && i + 3 < argc) {
            attach_in = argv[++i], attach_row = std::stoull(argv[++i]), attach_out = argv[++i];
        }
    }

    try {
        auto features = read_features(argv[1], start, stop);
        auto tensor = to_tensormap(features, offset + start);
        metatensor::io::save(argv[2], tensor);
        std::printf("saved %zu x %zu scaled features (rows %llu:%llu) to %s\n", features.n_rows, features.n_features,
                    static_cast<unsigned long long>(start), static_cast<unsigned long long>(start + features.n_rows),
                    argv[2]);

        if (attach_in) {
            // one row of features, as per-system custom data with a single `system` sample
            auto row = read_features(argv[1], attach_row, attach_row + 1);
            auto system = metatomic::io::load(attach_in);
            auto data = to_tensormap(row, 0);
            system.add_custom_data("madcore::features", std::move(data));
            metatomic::io::save(attach_out, system);
            std::printf("attached features of row %zu to %s -> %s\n", attach_row, attach_in, attach_out);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
