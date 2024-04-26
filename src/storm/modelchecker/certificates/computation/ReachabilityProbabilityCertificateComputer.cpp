#include "storm/modelchecker/certificates/computation/ReachabilityProbabilityCertificateComputer.h"

#include <vector>

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/environment/solver/MinMaxSolverEnvironment.h"
#include "storm/modelchecker/certificates/ReachabilityProbabilityCertificate.h"
#include "storm/modelchecker/reachability/ReachabilityProbabilityToRewardTransformer.h"
#include "storm/solver/helper/IntervalterationHelper.h"
#include "storm/solver/helper/ValueIterationOperator.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/utility/Extremum.h"
#include "storm/utility/OptionalRef.h"
#include "storm/utility/vector.h"

namespace storm::modelchecker {

template<typename ValueType, bool Nondeterministic, storm::OptimizationDirection Dir>
class LowerUpperValueCertificateComputer {
   public:
    enum class Algorithm { IntervalIteration };
    struct SubsystemData {
        storm::storage::SparseMatrix<ValueType> transitions;
        std::shared_ptr<storm::solver::helper::ValueIterationOperator<ValueType, !Nondeterministic>> viOp;
    };

    LowerUpperValueCertificateComputer(storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix,
                                       storm::storage::BitVector const& terminalStates, storm::storage::BitVector const& prob1States)
        : transitionProbabilityMatrix(transitionProbabilityMatrix), terminalStates(terminalStates), prob1States(prob1States) {}

    std::pair<std::vector<ValueType>, std::vector<ValueType>> compute(Algorithm alg, bool relative, ValueType const& precision) {
        initializeValueVectors(alg);
        // TODO: topological?
        // TODO: Mec decomposition?
        // TODO: Choices tracking?
        computeForSubsystem(alg, relative, precision, ~terminalStates);
        return {std::move(globalLowerValues), std::move(globalUpperValues)};
    }

   private:
    void initializeValueVectors(Algorithm alg) {
        STORM_LOG_ASSERT(alg == Algorithm::IntervalIteration, "Unsupported algorithm.");
        globalLowerValues.assign(terminalStates.size(), storm::utility::zero<ValueType>());
        globalUpperValues.assign(terminalStates.size(), storm::utility::one<ValueType>());
        storm::utility::vector::setVectorValues(globalLowerValues, prob1States, storm::utility::one<ValueType>());
        auto prob0States = terminalStates ^ prob1States;
        storm::utility::vector::setVectorValues(globalUpperValues, prob0States, storm::utility::zero<ValueType>());
    }

    SubsystemData initializeSubsystemData(storm::storage::BitVector const& subsystem) const {
        auto result = SubsystemData{transitionProbabilityMatrix.getSubmatrix(true, subsystem, subsystem),
                                    std::make_shared<storm::solver::helper::ValueIterationOperator<ValueType, !Nondeterministic>>()};
        result.viOp->setMatrixBackwards(result.transitions);
        return result;
    }

    std::vector<ValueType> getSubsystemExitValues(storm::storage::BitVector const& subsystem, uint64_t numSubystemRows,
                                                  std::vector<ValueType> const& globalValues) {
        std::vector<ValueType> result;
        result.reserve(numSubystemRows);
        for (auto const state : subsystem) {
            for (auto const rowIndex : transitionProbabilityMatrix.getRowGroupIndices(state)) {
                auto rowValue = storm::utility::zero<ValueType>();
                for (auto const& entry : transitionProbabilityMatrix.getRow(rowIndex)) {
                    if (!subsystem.get(entry.getColumn())) {
                        rowValue += entry.getValue() * globalValues[entry.getColumn()];
                    }
                }
                result.push_back(std::move(rowValue));
            }
        }
        result.shrink_to_fit();
        STORM_LOG_ASSERT(result.size() == numSubystemRows, "Unexpected number of rows in subsystem.");
        return result;
    }

    void computeForSubsystem(Algorithm alg, bool relative, ValueType const& precision, storm::storage::BitVector const& subsystemStates) {
        STORM_LOG_ASSERT(alg == Algorithm::IntervalIteration, "Unsupported algorithm.");
        auto subsystemData = initializeSubsystemData(subsystemStates);
        storm::solver::helper::IntervalIterationHelper<ValueType, !Nondeterministic> iiHelper(subsystemData.viOp);
        auto subsystemOffsets = getSubsystemExitValues(subsystemStates, subsystemData.transitions.getRowCount(), globalLowerValues);
        auto xy = std::make_pair(storm::utility::vector::filterVector(globalLowerValues, subsystemStates),
                                 storm::utility::vector::filterVector(globalUpperValues, subsystemStates));
        uint64_t numIterations = 0;
        iiHelper.template II<Dir>(xy, subsystemOffsets, numIterations, relative, precision);  // TODO: relevant values?
        storm::utility::vector::setVectorValues(globalLowerValues, subsystemStates, xy.first);
        storm::utility::vector::setVectorValues(globalUpperValues, subsystemStates, xy.second);
    }

    storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix;
    storm::storage::BitVector const& terminalStates;
    storm::storage::BitVector const& prob1States;
    std::vector<ValueType> globalLowerValues, globalUpperValues;
    storm::OptionalRef<storm::storage::BitVector> globalPossibleChoices;
};

template<typename ValueType, storm::OptimizationDirection Dir>
std::vector<typename ReachabilityProbabilityCertificate<ValueType>::RankingType> computeLowerBoundRanking(
    storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions,
    storm::storage::BitVector const& targetStates, storm::OptionalRef<storm::storage::BitVector const> constraintStates,
    storm::OptionalRef<storm::storage::BitVector const> choiceConstraint = storm::NullRef) {
    using RankingType = typename ReachabilityProbabilityCertificate<ValueType>::RankingType;
    RankingType const InfRank = ReachabilityProbabilityCertificate<ValueType>::InfRank;
    auto const& rowGroupIndices = transitionProbabilityMatrix.getRowGroupIndices();

    std::vector<RankingType> ranks(targetStates.size(), InfRank);
    storm::utility::vector::setVectorValues<RankingType>(ranks, targetStates, 0);
    std::vector<uint64_t> stateQueue(targetStates.begin(), targetStates.end());
    while (!stateQueue.empty()) {
        auto const state = stateQueue.back();
        stateQueue.pop_back();
        for (auto const& preEntry : backwardTransitions.getRow(state)) {
            auto const preState = preEntry.getColumn();
            if (ranks[preState] != InfRank) {
                continue;
            }
            if (constraintStates.has_value() && !constraintStates->get(preState)) {
                continue;
            }
            storm::utility::Extremum<storm::solver::invert(Dir), RankingType> newRankMinusOne;
            // Iterate over all (selected) choices of the pre-state
            uint64_t currentChoice = rowGroupIndices[preState];
            if (choiceConstraint.has_value()) {
                currentChoice = choiceConstraint->getNextSetIndex(currentChoice);
            }
            auto nextChoice = [&choiceConstraint](auto c) { return choiceConstraint.has_value() ? choiceConstraint->getNextSetIndex(c + 1) : c + 1; };
            for (auto const endChoice = rowGroupIndices[preState + 1]; currentChoice < endChoice; currentChoice = nextChoice(currentChoice)) {
                // Compute current rank for this choice
                storm::utility::Minimum<RankingType> choiceRank;
                for (auto const& entry : transitionProbabilityMatrix.getRow(currentChoice)) {
                    if (storm::utility::isZero(entry.getValue())) {
                        continue;
                    }
                    choiceRank &= ranks[entry.getColumn()];
                }
                STORM_LOG_ASSERT(!choiceRank.empty(),
                                 "Ranking operator failed to compute rank for state " << state << " since the row " << choiceConstraint << " is empty.");
                newRankMinusOne &= *choiceRank;
            }
            STORM_LOG_ASSERT(!newRankMinusOne.empty(), "Ranking operator failed to compute rank for state " << state << ".");
            if (*newRankMinusOne != InfRank) {
                ranks[preState] = *newRankMinusOne + 1;
                stateQueue.push_back(preState);
            }
        }
    }
    return ranks;
}

template<typename ValueType>
struct CertificateData {
    std::vector<ValueType> lowerValues, upperValues;
    std::vector<typename ReachabilityProbabilityCertificate<ValueType>::RankingType> ranks;
};

template<typename ValueType, bool Nondeterministic, storm::OptimizationDirection Dir = storm::OptimizationDirection::Minimize>
CertificateData<ValueType> computeReachabilityProbabilityCertificateData(storm::Environment const& env,
                                                                         storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix,
                                                                         storm::storage::BitVector const& targetStates,
                                                                         storm::OptionalRef<storm::storage::BitVector const> constraintStates) {
    std::optional<storm::OptimizationDirection> constexpr optionalDir = Nondeterministic ? std::optional<storm::OptimizationDirection>(Dir) : std::nullopt;
    utility::BackwardTransitionCache<ValueType> backwardTransitionCache(transitionProbabilityMatrix);
    auto toRewardData = ReachabilityProbabilityToRewardTransformer<ValueType>(transitionProbabilityMatrix, backwardTransitionCache)
                            .transform(optionalDir, targetStates, constraintStates);
    STORM_LOG_ASSERT(toRewardData.terminalStateValues.size() == 1 && toRewardData.terminalStateValues.front().first.isOne(),
                     "Expected exactly one terminal state value with value 1.");
    LowerUpperValueCertificateComputer<ValueType, Nondeterministic, Dir> computer(transitionProbabilityMatrix, toRewardData.terminalStates,
                                                                                  toRewardData.terminalStateValues.front().second);

    auto [lowerValues, upperValues] = computer.compute(LowerUpperValueCertificateComputer<ValueType, Nondeterministic, Dir>::Algorithm::IntervalIteration,
                                                       env.solver().minMax().getRelativeTerminationCriterion(),
                                                       storm::utility::convertNumber<ValueType>(env.solver().minMax().getPrecision()));
    auto ranks = computeLowerBoundRanking<ValueType, Dir>(transitionProbabilityMatrix, backwardTransitionCache.get(), targetStates,
                                                          constraintStates);  // TODO: choice constraints for max reach
    return {std::move(lowerValues), std::move(upperValues), std::move(ranks)};
}

template<typename ValueType>
CertificateData<ValueType> computeReachabilityProbabilityCertificateData(storm::Environment const& env, std::optional<storm::OptimizationDirection> dir,
                                                                         storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix,
                                                                         storm::storage::BitVector const& targetStates,
                                                                         storm::OptionalRef<storm::storage::BitVector const> constraintStates) {
    STORM_PRINT_AND_LOG("Transforming to floating point arithmetic.");
    auto matrixDouble = transitionProbabilityMatrix.template toValueType<double>();
    auto dataDouble = computeReachabilityProbabilityCertificateData(env, dir, matrixDouble, targetStates, constraintStates);
    return CertificateData<ValueType>{storm::utility::vector::convertNumericVector<ValueType>(dataDouble.lowerValues),
                                      storm::utility::vector::convertNumericVector<ValueType>(dataDouble.upperValues), std::move(dataDouble.ranks)};
}

template<>
CertificateData<double> computeReachabilityProbabilityCertificateData(storm::Environment const& env, std::optional<storm::OptimizationDirection> dir,
                                                                      storm::storage::SparseMatrix<double> const& transitionProbabilityMatrix,
                                                                      storm::storage::BitVector const& targetStates,
                                                                      storm::OptionalRef<storm::storage::BitVector const> constraintStates) {
    using ValueType = double;
    CertificateData<ValueType> data;
    if (dir.has_value()) {
        if (storm::solver::maximize(*dir)) {
            data = computeReachabilityProbabilityCertificateData<ValueType, true, storm::OptimizationDirection::Maximize>(env, transitionProbabilityMatrix,
                                                                                                                          targetStates, constraintStates);
        } else {
            data = computeReachabilityProbabilityCertificateData<ValueType, true, storm::OptimizationDirection::Minimize>(env, transitionProbabilityMatrix,
                                                                                                                          targetStates, constraintStates);
        }
    } else {
        data = computeReachabilityProbabilityCertificateData<ValueType, false>(env, transitionProbabilityMatrix, targetStates, constraintStates);
    }
    return data;
}

template<typename ValueType>
std::unique_ptr<ReachabilityProbabilityCertificate<ValueType>> computeReachabilityProbabilityCertificate(
    storm::Environment const& env, std::optional<storm::OptimizationDirection> dir, storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix,
    storm::storage::BitVector targetStates, std::optional<storm::storage::BitVector> constraintStates, std::string targetLabel,
    std::optional<std::string> constraintLabel) {
    CertificateData<ValueType> data;
    std::unique_ptr<ReachabilityProbabilityCertificate<ValueType>> result;
    if (constraintStates.has_value()) {
        data = computeReachabilityProbabilityCertificateData<ValueType>(env, dir, transitionProbabilityMatrix, targetStates, constraintStates.value());
        result = std::make_unique<ReachabilityProbabilityCertificate<ValueType>>(dir, std::move(targetStates), std::move(constraintStates.value()),
                                                                                 std::move(targetLabel), constraintLabel.value_or("constraint"));
    } else {
        data = computeReachabilityProbabilityCertificateData<ValueType>(env, dir, transitionProbabilityMatrix, targetStates, storm::NullRef);
        result = std::make_unique<ReachabilityProbabilityCertificate<ValueType>>(dir, std::move(targetStates), std::move(targetLabel));
    }
    result->setLowerBoundsCertificate(std::move(data.lowerValues), std::move(data.ranks));
    result->setUpperBoundsCertificate(std::move(data.upperValues));
    return result;
}

template std::unique_ptr<ReachabilityProbabilityCertificate<double>> computeReachabilityProbabilityCertificate<double>(
    storm::Environment const& env, std::optional<storm::OptimizationDirection> dir, storm::storage::SparseMatrix<double> const& transitionProbabilityMatrix,
    storm::storage::BitVector targetStates, std::optional<storm::storage::BitVector> constraintStates, std::string targetLabel,
    std::optional<std::string> constraintLabel);
template std::unique_ptr<ReachabilityProbabilityCertificate<storm::RationalNumber>> computeReachabilityProbabilityCertificate<storm::RationalNumber>(
    storm::Environment const& env, std::optional<storm::OptimizationDirection> dir,
    storm::storage::SparseMatrix<storm::RationalNumber> const& transitionProbabilityMatrix, storm::storage::BitVector targetStates,
    std::optional<storm::storage::BitVector> constraintStates, std::string targetLabel, std::optional<std::string> constraintLabel);

}  // namespace storm::modelchecker