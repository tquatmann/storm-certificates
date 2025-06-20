#pragma once

#include <memory>
#include <optional>
#include <string>

#include "storm/modelchecker/certificates/ReachabilityProbabilityCertificate.h"
#include "storm/modelchecker/certificates/ReachabilityRewardCertificate.h"
#include "storm/solver/OptimizationDirection.h"

namespace storm {
class Environment;

namespace storage {
class BitVector;
template<typename ValueType>
class SparseMatrix;
}  // namespace storage

namespace modelchecker {

template<typename ValueType>
std::unique_ptr<ReachabilityProbabilityCertificate<ValueType>> computeReachabilityProbabilityCertificate(
    storm::Environment const& env, std::optional<storm::OptimizationDirection> dir, storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix,
    storm::storage::BitVector targetStates, std::optional<storm::storage::BitVector> constraintStates = std::nullopt, std::string targetLabel = "goal",
    std::optional<std::string> constraintLabel = std::nullopt);

template<typename ValueType>
std::unique_ptr<ReachabilityRewardCertificate<ValueType>> computeReachabilityRewardCertificate(
    storm::Environment const& env, std::optional<storm::OptimizationDirection> dir, storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix,
    storm::storage::BitVector targetStates, std::vector<ValueType> stateActionRewardVector, std::string targetLabel = "goal", std::string rewardModelName = "");

}  // namespace modelchecker
}  // namespace storm