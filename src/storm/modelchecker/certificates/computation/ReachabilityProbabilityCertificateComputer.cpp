
#include "storm/modelchecker/certificates/computation/ReachabilityProbabilityCertificateComputer.h"
#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/modelchecker/certificates/ReachabilityProbabilityCertificate.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"

namespace storm::modelchecker {

template<typename ValueType>
std::unique_ptr<ReachabilityProbabilityCertificate<ValueType>> computeReachabilityProbabilityCertificate(
    storm::Environment const& env, std::optional<storm::OptimizationDirection> dir, storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix,
    storm::storage::BitVector targetStates, std::string targetLabel) {
    // TODO: Implement the computation of the reachability probability certificate
    return std::make_unique<ReachabilityProbabilityCertificate<ValueType>>(dir, std::move(targetStates), std::move(targetLabel));
}

template std::unique_ptr<ReachabilityProbabilityCertificate<double>> computeReachabilityProbabilityCertificate<double>(
    storm::Environment const& env, std::optional<storm::OptimizationDirection> dir, storm::storage::SparseMatrix<double> const& transitionProbabilityMatrix,
    storm::storage::BitVector targetStates, std::string targetLabel);
template std::unique_ptr<ReachabilityProbabilityCertificate<storm::RationalNumber>> computeReachabilityProbabilityCertificate<storm::RationalNumber>(
    storm::Environment const& env, std::optional<storm::OptimizationDirection> dir,
    storm::storage::SparseMatrix<storm::RationalNumber> const& transitionProbabilityMatrix, storm::storage::BitVector targetStates, std::string targetLabel);

}  // namespace storm::modelchecker