#pragma once

#include <vector>
#include "storm/modelchecker/helper/utility/BackwardTransitionCache.h"
#include "storm/modelchecker/totalreward/TotalRewardSpecification.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/utility/OptionalRef.h"
#include "storm/utility/graph.h"

namespace storm::modelchecker {

template<typename ValueType>
class ReachabilityRewardTransformer {
   public:
    // TODO: allow symbolic representations, too
    using StateSet = storm::storage::BitVector;

    ReachabilityRewardTransformer(storm::storage::SparseMatrix<ValueType> const& transitionMatrix)
        : transitionMatrix(transitionMatrix), backwardTransitionCache(transitionMatrix) {
        // Intentionally left empty.
    }
    ReachabilityRewardTransformer(storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                                  utility::BackwardTransitionCache<ValueType> backwardTransitionCache)
        : transitionMatrix(transitionMatrix), backwardTransitionCache(backwardTransitionCache) {
        // Intentionally left empty.
    }

    TotalRewardSpecification<ValueType> transform(StateSet const& targetStates, std::vector<ValueType> const& actionRewards) {
        STORM_LOG_ASSERT(transitionMatrix.hasTrivialRowGrouping(), "Transition matrix has non-trivial row grouping but no optimization direction is given.");

        StateSet allStates(transitionMatrix.getRowGroupCount(), true);
        auto infStates = storm::utility::graph::performProb1(backwardTransitionCache.get(), allStates, targetStates);
        infStates.complement();
        return TotalRewardSpecification<ValueType>{.terminalStates = targetStates | infStates,
                                                   .terminalStateValues = {{TerminalStateValue<ValueType>::getPlusInfinity(), std::move(infStates)}},
                                                   .terminalStatesUniversallyAlmostSurelyReached = true,
                                                   .actionRewards = actionRewards};
    }

    TotalRewardSpecification<ValueType> transform(std::optional<storm::OptimizationDirection> dir, StateSet const& targetStates,
                                                  std::vector<ValueType> const& actionRewards) {
        if (!dir.has_value()) {
            return transform(targetStates, actionRewards);
        }
        StateSet allStates(transitionMatrix.getRowGroupCount(), true);
        auto infStates = storm::solver::minimize(*dir) ? storm::utility::graph::performProb1E(transitionMatrix, transitionMatrix.getRowGroupIndices(),
                                                                                              backwardTransitionCache.get(), allStates, targetStates)
                                                       : storm::utility::graph::performProb1A(transitionMatrix, transitionMatrix.getRowGroupIndices(),
                                                                                              backwardTransitionCache.get(), allStates, targetStates);
        infStates.complement();
        return TotalRewardSpecification<ValueType>{.terminalStates = targetStates | infStates,
                                                   .terminalStateValues = {{TerminalStateValue<ValueType>::getPlusInfinity(), std::move(infStates)}},
                                                   .terminalStatesUniversallyAlmostSurelyReached = storm::solver::maximize(*dir),
                                                   .actionRewards = actionRewards};
    }

    storm::storage::SparseMatrix<ValueType> transitionMatrix;
    utility::BackwardTransitionCache<ValueType> backwardTransitionCache;
};

}  // namespace storm::modelchecker