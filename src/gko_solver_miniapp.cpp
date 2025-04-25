#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <mpi.h>

#include <petsc.h>

#include <ginkgo/ginkgo.hpp>

#include <ginkgo/extensions/config/json_config.hpp>

#include "config.hpp"

//#include "case_names.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>

#include "yaml-cpp/yaml.h"


using Path = std::filesystem::path;


PetscErrorCode load_vector_from_binary(const Path& filename, Vec& v)
{
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename.c_str(),
                                    FILE_MODE_READ, &viewer));

    PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
    PetscCall(VecLoad(v, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscFunctionReturn(0);
}

PetscErrorCode load_matrix_from_binary(const Path& filename, Mat& A)
{
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename.c_str(),
                                    FILE_MODE_READ, &viewer));

    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatLoad(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscFunctionReturn(0);
}

PetscErrorCode output_stdout(const std::string& system_name,
                             const std::string& solver_type,
                             const std::string& pc_type, PetscReal tol,
                             int iters, double timing, int nrep)
{
    std::string message = "Solver timings for case " + system_name + ":\n";
    message += "\tSolver type: " + std::string(solver_type) + "\n";
    message += "\tSolver relative tolerance: " + std::to_string(tol) + "\n";
    message += "\tNum iters for convergence: " + std::to_string(iters) + "\n";
    message += "\tPreconditioner type: " + std::string(pc_type) + "\n";
    message += "\n";
    message += "\tNumber of solve repetitions: " + std::to_string(nrep) + "\n";
    message +=
        "\tAverage time per repetition: " + std::to_string(timing / nrep) +
        "s\n";
    message += "\n";
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, message.c_str()));

    PetscFunctionReturn(0);
}

void output_yaml(const std::string& system_name, const std::string& solver_type,
                 PetscReal tol, const std::string& pc_type, double timing,
                 int nrep, int iters)
{
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "system_name";
    out << YAML::Value << system_name;

    out << YAML::Key << "solver";
    out << YAML::Value;
    out << YAML::BeginMap;
    out << YAML::Key << "type";
    out << YAML::Value << solver_type;
    out << YAML::Key << "rtol";
    out << YAML::Value << tol;
    out << YAML::EndMap;

    out << YAML::Key << "preconditioner";
    out << YAML::Value;
    out << YAML::BeginMap;
    out << YAML::Key << "type";
    out << YAML::Value << pc_type;
    out << YAML::EndMap;

    out << YAML::Key << "results";
    out << YAML::Value;
    out << YAML::BeginMap;
    out << YAML::Key << "avg_time_per_rep";
    out << YAML::Value << timing / nrep;
    out << YAML::Key << "nrep";
    out << YAML::Value << nrep;
    out << YAML::Key << "n_iters";
    out << YAML::Value << iters;
    out << YAML::EndMap;
    out << YAML::EndMap;

    std::ofstream result_file;
    std::string result_file_name = "ginkgo_solve_" + system_name + ".yaml";
    result_file.open(result_file_name, std::ios::out);
    result_file << out.c_str();
    result_file.close();
}

PetscErrorCode print_run_info(const std::string& system_name,
                              gko::config::pnode& config, int conv_iters,
                              double timing, int nrep)
{
    auto prec_type = config.get("preconditioner")
                         ? config.get("preconditioner").get("type")
                         : gko::config::pnode{};
    auto prec = prec_type ? prec_type.get_string() : "none";
    auto solver = config.get("type").get_string();
    auto crit = config.get("criteria").get_array();
    auto iters = crit.at(0).get("max_iters").get_integer();
    auto tol = crit.at(1).get("reduction_factor").get_real();
    PetscCall(output_stdout(system_name, solver, prec, tol, conv_iters, timing,
                            nrep));
    output_yaml(system_name, solver, tol, prec, timing, nrep, conv_iters);

    PetscFunctionReturn(0);
}

// Load system Ax = b and solve it
PetscErrorCode solve_system_named(const std::shared_ptr<gko::Executor>& exec,
                                  std::string system_name,
                                  gko::config::pnode& config,
                                  const Path& root_dir, int nrep, int sys)
{
    using vtype = PetscScalar;
    using itype = PetscInt;
    using gko_vec = gko::matrix::Dense<vtype>;
    using gko_mat = gko::matrix::Csr<vtype, itype>;
    // using gko_cg = gko::solver::Cg<vtype>;

    // Load RHS
    Vec b{};
    std::string filename = std::to_string(sys) + "_RHS_" + system_name;
    Path file_path = root_dir / filename;
    PetscCall(load_vector_from_binary(file_path, b));
    PetscCall(VecScale(b, -1.0));
    vtype* rhs_values;
    itype rhs_size;
    PetscCall(VecGetArray(b, &rhs_values));
    PetscCall(VecGetSize(b, &rhs_size));
    auto rhs_host = gko_vec::create_const(
        exec->get_master(), gko::dim<2>(rhs_size, 1),
        gko::array<vtype>::const_view(exec->get_master(), rhs_size, rhs_values),
        1);
    auto rhs = gko_vec::create(exec);
    rhs->copy_from(rhs_host);

    // Load LHS and create preconditioner matrix
    Mat A{};
    switch (sys) {
    case 1:
        filename = "p_doubleprime_" + system_name;
        break;
    case 2:
        filename = "Laplacian_" + system_name;
        break;
    case 3:
        filename = "U_" + system_name;
        break;
    }
    file_path = root_dir / filename;
    PetscCall(load_matrix_from_binary(file_path, A));
    itype* col_idxs;
    itype* row_ptrs;
    vtype* values;
    itype num_rows;
    PetscBool done;
    PetscBool symm = PETSC_FALSE;
    PetscBool inodecomp = PETSC_FALSE;

    PetscCall(MatScale(A, -1.0));

    PetscCall(MatSeqAIJGetArray(A, &values));
    PetscCall(MatGetRowIJ(A, 0, symm, inodecomp, &num_rows,
                          const_cast<const itype**>(&row_ptrs),
                          const_cast<const itype**>(&col_idxs), &done));

    if (rhs_size != num_rows) {
        PetscCall(PetscError(PETSC_COMM_WORLD, __LINE__, "solve_system_named",
                             __FILE__, PETSC_ERR_ARG_SIZ, PETSC_ERROR_INITIAL,
                             "expected rhs and size to be of same size"));
    }
    itype num_nnz = row_ptrs[num_rows];
    auto mat_host = gko::share(gko_mat::create_const(
        exec->get_master(), gko::dim<2>(num_rows, num_rows),
        gko::array<vtype>::const_view(exec->get_master(), num_nnz, values),
        gko::array<itype>::const_view(exec->get_master(), num_nnz, col_idxs),
        gko::array<itype>::const_view(exec->get_master(), num_rows + 1,
                                      row_ptrs)));

    auto mat = gko::share(gko_mat::create(exec));
    mat->copy_from(mat_host);

    // Create solution vector
    auto sol = gko::share(gko_vec::create_with_config_of(rhs));

    // load guess
    Vec guess{};
    filename = std::to_string(sys) + "_initialguess_" + system_name;
    file_path = root_dir / filename;
    PetscCall(load_vector_from_binary(file_path, guess));
    vtype* guess_values;
    itype guess_size;
    PetscCall(VecGetArray(guess, &guess_values));
    PetscCall(VecGetSize(guess, &guess_size));
    auto guess_host =
        gko_vec::create_const(exec->get_master(), gko::dim<2>(guess_size, 1),
                              gko::array<vtype>::const_view(
                                  exec->get_master(), guess_size, guess_values),
                              1);
    auto guess_dev = gko_vec::create(exec);
    guess_dev->copy_from(guess_host);

    auto reg = gko::config::registry();
    auto td = gko::config::make_type_descriptor<vtype, itype>();

    auto solver_gen = gko::config::parse(config, reg, td).on(exec);
    auto solver = solver_gen->generate(mat);

    std::shared_ptr<gko::log::Convergence<vtype>> logger =
        gko::log::Convergence<vtype>::create();
    solver->add_logger(logger);

    double timing_solve = 0.0;
    for (int rep = 0; rep < nrep; ++rep) {
        sol->copy_from(guess_dev);
        auto t1 = MPI_Wtime();
        solver->apply(rhs, sol);
        timing_solve += (MPI_Wtime() - t1);
    }

    const int conv_iters = logger->get_num_iterations();
    auto res = gko::as<gko_vec>(logger->get_residual_norm());

    const auto fres = exec->copy_val_to_host(res->get_const_values());

    PetscCall(
        print_run_info(system_name, config, conv_iters, timing_solve, nrep));

    PetscFunctionReturn(0);
}

int main(int argc, char* argv[])
{
    constexpr int NREP = 50;

    constexpr int number_of_cases = 64;
    std::array<std::string, number_of_cases> case_names = {};

    std::ostringstream oss;

    for (int i = 0; i < number_of_cases; i++) {
        oss.str("");
        oss << std::setw(3) << std::setfill('0') << i;
        case_names[i] = oss.str();
    }

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    Path root_dir(MINIAPP_DATA_PATH);

    std::string configfile = argc >= 2 ? argv[1] : "gko-config/cg.json";

    auto config = gko::ext::config::parse_json_file(configfile);
    auto executor_string = config.get("exec").get_string();

    std::cout << executor_string << std::endl;

    const auto exec = exec_map.at(executor_string)();

    int sys = argc >= 3 ? std::stoi(argv[2]) : 3;

    std::cout << "###########\nSolving with ginkgo system: " << sys
              << std::endl;

    for (auto case_name : case_names) {
        PetscCall(
            solve_system_named(exec, case_name, config, root_dir, NREP, sys));
    }

    PetscCall(PetscFinalize());
    return 0;
}
