#pragma once

#include <boost/container/flat_map.hpp>
#include <vector>
#include "storm/modelchecker/helper/utility/BackwardTransitionCache.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/utility/OptionalRef.h"
#include "storm/utility/graph.h"

namespace storm::modelchecker {

template<typename ValueType>
class ReachabilityProbabilityToRewardTransformer {
   public:
    // TODO: allow symbolic representations, too
    using StateSet = storm::storage::BitVector;

    struct ReturnType {
        StateSet terminalStates;                            /// Those states that are target w.r.t. the total reachability reward query
        StateSet prob1States;                               /// Those states that can reach a target with probability 1 (subset of targetStates)
        bool terminalStatesUniversallyAlmostSurelyReached;  /// If true, the target states are reached almost surely from every state under every scheduler
    };

    ReachabilityProbabilityToRewardTransformer(storm::storage::SparseMatrix<ValueType> const& transitionMatrix)
        : transitionMatrix(transitionMatrix), backwardTransitionCache(transitionMatrix) {
        // Intentionally left empty.
    }
    ReachabilityProbabilityToRewardTransformer(storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                                               utility::BackwardTransitionCache<ValueType> backwardTransitionCache)
        : transitionMatrix(transitionMatrix), backwardTransitionCache(backwardTransitionCache) {
        // Intentionally left empty.
    }

    ReturnType transform(StateSet const& targetStates, storm::OptionalRef<StateSet const> constraintStates = storm::NullRef) {
        STORM_LOG_ASSERT(transitionMatrix.hasTrivialRowGrouping(), "Transition matrix has non-trivial row grouping but no optimization direction is given.");
        StateSet defaultConstraintStates;
        if (!constraintStates.has_value()) {
            defaultConstraintStates = StateSet(transitionMatrix.getRowGroupCount(), true);
            constraintStates.reset(defaultConstraintStates);
        }
        auto [prob01States, prob1States] = storm::utility::graph::performProb01(backwardTransitionCache.get(), constraintStates.value(), targetStates);
        prob01States |= prob1States;

        return ReturnType{std::move(prob01States), std::move(prob1States), true};
    }

    ReturnType transform(std::optional<storm::OptimizationDirection> dir, StateSet const& targetStates,
                         storm::OptionalRef<StateSet const> constraintStates = storm::NullRef) {
        if (!dir.has_value()) {
            return transform(targetStates, constraintStates);
        }
        StateSet defaultConstraintStates;
        if (!constraintStates.has_value()) {
            defaultConstraintStates = StateSet(transitionMatrix.getRowGroupCount(), true);
            constraintStates.reset(defaultConstraintStates);
        }

        auto [prob01States, prob1States] = storm::solver::minimize(*dir)
                                               ? storm::utility::graph::performProb01Min(transitionMatrix, transitionMatrix.getRowGroupIndices(),
                                                                                         backwardTransitionCache.get(), constraintStates.value(), targetStates)
                                               : storm::utility::graph::performProb01Max(transitionMatrix, transitionMatrix.getRowGroupIndices(),
                                                                                         backwardTransitionCache.get(), constraintStates.value(), targetStates);
        prob01States |= prob1States;

        return ReturnType{std::move(prob01States), std::move(prob1States), storm::solver::minimize(*dir)};
    }

    storm::storage::SparseMatrix<ValueType> transitionMatrix;
    utility::BackwardTransitionCache<ValueType> backwardTransitionCache;
};

}  // namespace storm::modelchecker