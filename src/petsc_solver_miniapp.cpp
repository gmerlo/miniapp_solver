#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <mpi.h>

#include <petsc.h>

//#include "case_names.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>

#include "config.hpp"
#include "yaml-cpp/yaml.h"

constexpr bool FILE_DEBUG = true;

using Path = std::filesystem::path;

PetscErrorCode load_vector_from_binary(const Path& filename, Vec& v)
{
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename.c_str(),
                                    FILE_MODE_READ, &viewer));

    PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
    PetscCall(VecLoad(v, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(VecSetFromOptions(v));
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
                             const KSPType& solver_type, PetscReal rtol,
                             const PCType& pc_type, double timing, int nrep,
                             int iters)
{
    std::string message = "Solver timings for case " + system_name + ":\n";
    message += "\tSolver type: " + std::string(solver_type) + "\n";
    message += "\tSolver relative tolerance: " + std::to_string(rtol) + "\n";
    message += "\tPreconditioner type: " + std::string(pc_type) + "\n";
    message += "\n";
    message += "\tNumber of solve repetitions: " + std::to_string(nrep) + "\n";
    message += "\tNum iters for convergence: " + std::to_string(iters) + "\n";
    message +=
        "\tAverage time per repetition: " + std::to_string(timing / nrep) +
        "s\n";
    message += "\n";
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, message.c_str()));

    PetscFunctionReturn(0);
}

void output_yaml(const std::string& system_name, const KSPType& solver_type,
                 PetscReal rtol, const PCType& pc_type, double timing, int nrep,
                 int iters)
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
    out << YAML::Value << rtol;
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
    std::string result_file_name = "petsc_solve_" + system_name + ".yaml";
    result_file.open(result_file_name, std::ios::out);
    result_file << out.c_str();
    result_file.close();
}

PetscErrorCode print_run_info(const std::string& system_name, const KSP& solver,
                              double timing, int nrep, int iters)
{
    KSPType solver_type;
    PetscCall(KSPGetType(solver, &solver_type));

    PetscReal rtol;
    PetscCall(KSPGetTolerances(solver, &rtol, NULL, NULL, NULL));

    PC pc;
    PetscCall(KSPGetPC(solver, &pc));
    PCType pc_type;
    PetscCall(PCGetType(pc, &pc_type));

    PetscCall(output_stdout(system_name, solver_type, rtol, pc_type, timing,
                            nrep, iters));
    output_yaml(system_name, solver_type, rtol, pc_type, timing, nrep, iters);

    PetscFunctionReturn(0);
}

// Load system Ax = b and solve it
PetscErrorCode solve_system_named(std::string system_name, const Path& root_dir,
                                  int nrep, int sys)
{
    // Load RHS
    Vec b{};
    std::string filename = std::to_string(sys) + "_RHS_" + system_name;
    Path file_path = root_dir / filename;
    if (FILE_DEBUG) {
        std::cout << "Rhs path: " << file_path.string() << std::endl;
    }
    PetscCall(load_vector_from_binary(file_path, b));

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
    if (FILE_DEBUG) {
        std::cout << "Matrix path" << file_path.string() << std::endl;
    }
    PetscCall(load_matrix_from_binary(file_path, A));

    PetscCall(MatConvert(A, MATSEQAIJCUSPARSE, MAT_INPLACE_MATRIX, &A));

    //    Mat P{};
    // PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &P));

    // Load and store guess  to reuse
    Vec guess{};
    filename = std::to_string(sys) + "_initialguess_" + system_name;
    file_path = root_dir / filename;
    PetscCall(load_vector_from_binary(file_path, guess));

    // Create solution vector
    Vec x{};
    PetscCall(VecDuplicate(guess, &x));

    // Create solver & solve the system
    KSP solver;
    KSPType mytype;

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &solver));
    PetscCall(KSPSetOperators(solver, A, A));
    PetscCall(KSPSetFromOptions(solver));
    PetscCall(KSPGetType(solver, &mytype));
    if (mytype == "ksppreonly") {
        // no guess here
    } else {
        std::cout << "type is " << mytype << "-" << KSPPREONLY << std::endl;
        PetscCall(KSPSetInitialGuessNonzero(solver, PETSC_TRUE));
    }
    PetscCall(KSPSetUp(solver));

    double timing_solve = 0.0;
    PetscInt its, its_avg = 0.0;

    for (int rep = 0; rep < nrep; ++rep) {
        PetscCall(VecCopy(guess, x));
        auto t1 = MPI_Wtime();
        PetscCall(KSPSolve(solver, b, x));
        timing_solve += (MPI_Wtime() - t1);
        PetscCall(KSPGetIterationNumber(solver, &its));
        its_avg += its;
    }
    PetscCall(print_run_info(system_name, solver, timing_solve, nrep,
                             its_avg / nrep));

    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&guess));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
    //    PetscCall(MatDestroy(&P));
    PetscCall(KSPDestroy(&solver));

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
    static_assert(std::is_same_v<float, PetscScalar> == true);

    Path root_dir(MINIAPP_DATA_PATH);

    int sys = argc >= 2 ? std::stoi(argv[1]) : 3;

    std::cout << "###########\nSolving with petsc system " << sys << ", "
              << case_names[0] << std::endl;

    for (auto case_name : case_names) {
        PetscCall(solve_system_named(case_name, root_dir, NREP, sys));
    }

    PetscCall(PetscFinalize());
    return 0;
}
