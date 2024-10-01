#pragma once

#include <optional>

#include "storm/modelchecker/prctl/helper/BaierUpperRewardBoundsComputer.h"
#include "storm/modelchecker/prctl/helper/DsMpiUpperRewardBoundsComputer.h"
#include "storm/solver/OptimizationDirection.h"
#include "storm/utility/Extremum.h"
#include "storm/utility/OptionalRef.h"
#include "storm/utility/macros.h"
#include "storm/utility/vector.h"

namespace storm::modelchecker {

template<typename ValueType, bool Lower>
std::vector<ValueType> computeLowerOrUpperBoundsTotalReward(storm::storage::SparseMatrix<ValueType> const& matrix, std::vector<ValueType> const& offsets,
                                                            storm::storage::BitVector const& exitRows, std::vector<ValueType> const& exitProbabilities,
                                                            std::optional<storm::OptimizationDirection> dir = {}) {
    STORM_LOG_ASSERT(matrix.hasTrivialRowGrouping() || dir.has_value(), "Optimization direction needs to be given if matrix is nondeterministic.");
    storm::utility::Extremum<Lower ? storm::OptimizationDirection::Minimize : storm::OptimizationDirection::Maximize, ValueType> globalBound;
    // Try to compute a lower and an upper bound the "easy" way
    auto const insideRows = ~exitRows;
    std::conditional_t<Lower, std::greater_equal<ValueType>, std::less_equal<ValueType>> comp;
    bool const canEasy =
        std::all_of(insideRows.begin(), insideRows.end(), [&offsets, &comp](uint64_t row) { return comp(offsets[row], storm::utility::zero<ValueType>()); });

    if (canEasy) {
        for (auto const row : exitRows) {
            STORM_LOG_ASSERT(!storm::utility::isZero(exitProbabilities.at(row)), "Exit probability must not be zero.");
            ValueType const rowValue = offsets[row] / exitProbabilities[row];
            globalBound &= rowValue;
        }
    }

    // We might have to invoke the respective reward bound computers.
    if (!canEasy) {
        // For lower bounds, we actually compute upper bounds for the negated rewards.
        // For upper bounds, we potentially cut away negative offsets.
        std::optional<std::vector<ValueType>> tmpOffsets;
        if (Lower) {
            tmpOffsets.emplace(offsets.size());
            storm::utility::vector::applyPointwise(offsets, *tmpOffsets,
                                                   [](ValueType const& v) { return std::max<ValueType>(storm::utility::zero<ValueType>(), -v); });
        } else if (std::any_of(offsets.begin(), offsets.end(), [](ValueType const& v) { return v < storm::utility::zero<ValueType>(); })) {
            tmpOffsets.emplace(offsets.size());
            storm::utility::vector::applyPointwise(offsets, *tmpOffsets, [](ValueType const& v) { return std::max(storm::utility::zero<ValueType>(), v); });
        }
        if (dir.has_value() && (Lower == minimize(*dir))) {
            auto bound = storm::modelchecker::helper::BaierUpperRewardBoundsComputer<ValueType>(matrix, tmpOffsets.has_value() ? tmpOffsets.value() : offsets,
                                                                                                exitProbabilities)
                             .computeUpperBound();
            if (Lower) {
                bound = -bound;
            }
            globalBound &= bound;
        } else {
            auto bounds = storm::modelchecker::helper::DsMpiMdpUpperRewardBoundsComputer<ValueType>(
                              matrix, tmpOffsets.has_value() ? tmpOffsets.value() : offsets, exitProbabilities)
                              .computeUpperBounds();
            if (Lower) {
                storm::utility::vector::applyPointwise(bounds, bounds, [](ValueType const& v) { return -v; });
            }
            return bounds;
        }
    }
    STORM_LOG_ASSERT(!globalBound.empty(), "Could not compute lower or upper bound for total reward.");
    return std::vector<ValueType>(matrix.getRowGroupCount(), *globalBound);
}
}  // namespace storm::modelchecker