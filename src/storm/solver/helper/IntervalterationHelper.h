#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "storm/solver/OptimizationDirection.h"
#include "storm/solver/SolverStatus.h"
#include "storm/solver/helper/ValueIterationOperatorForward.h"
#include "storm/storage/BitVector.h"

namespace storm::solver::helper {

template<typename ValueType>
struct IIData {
    std::vector<ValueType> const& x;
    std::vector<ValueType> const& y;
    SolverStatus const status;
};

/*!
 * Implements interval iteration
 * @see https://doi.org/10.1007/978-3-319-63387-9_8
 */
template<typename ValueType, bool TrivialRowGrouping>
class IntervalIterationHelper {
   public:
    IntervalIterationHelper(std::shared_ptr<ValueIterationOperator<ValueType, TrivialRowGrouping>> viOperator);

    template<OptimizationDirection Dir, typename OffsetType, bool Smooth = false, bool InvertedXDir = false>
    SolverStatus II(std::pair<std::vector<ValueType>, std::vector<ValueType>>& xy, OffsetType const& offsets, uint64_t& numIterations, bool relative,
                    ValueType const& precision, std::function<SolverStatus(IIData<ValueType> const&)> const& iterationCallback = {},
                    std::optional<storm::storage::BitVector> const& relevantValues = {}, ValueType const& gamma = 0.0, ValueType const& delta = 0.0) const;

    SolverStatus smoothII(std::pair<std::vector<ValueType>, std::vector<ValueType>>& xy,
                          std::pair<std::vector<ValueType>, std::vector<ValueType>> const& offsets, uint64_t& numIterations, bool relative,
                          ValueType const& precision, ValueType const& gamma, ValueType const& delta,
                          std::optional<storm::OptimizationDirection> const& dir = {},
                          std::function<SolverStatus(IIData<ValueType> const&)> const& iterationCallback = {},
                          std::optional<storm::storage::BitVector> const& relevantValues = {}) const;

    SolverStatus roundII(std::pair<std::vector<ValueType>, std::vector<ValueType>>& xy,
                         std::pair<std::vector<ValueType>, std::vector<ValueType>> const& offsets, uint64_t& numIterations, bool relative,
                         ValueType const& precision, ValueType const& gamma, ValueType const& delta,
                         std::optional<storm::OptimizationDirection> const& dir = {},
                         std::function<SolverStatus(IIData<ValueType> const&)> const& iterationCallback = {},
                         std::optional<storm::storage::BitVector> const& relevantValues = {}) const;

    SolverStatus II(std::pair<std::vector<ValueType>, std::vector<ValueType>>& xy, std::pair<std::vector<ValueType>, std::vector<ValueType>> const& offsets,
                    uint64_t& numIterations, bool relative, ValueType const& precision, std::optional<storm::OptimizationDirection> const& dir = {},
                    std::function<SolverStatus(IIData<ValueType> const&)> const& iterationCallback = {},
                    std::optional<storm::storage::BitVector> const& relevantValues = {}) const;

    SolverStatus II(std::vector<ValueType>& operand, std::vector<ValueType> const& offsets, uint64_t& numIterations, bool relative, ValueType const& precision,
                    std::function<void(std::vector<ValueType>&)> const& prepareLowerBounds,
                    std::function<void(std::vector<ValueType>&)> const& prepareUpperBounds, std::optional<storm::OptimizationDirection> const& dir = {},
                    std::function<SolverStatus(IIData<ValueType> const&)> const& iterationCallback = {},
                    std::optional<storm::storage::BitVector> const& relevantValues = {}) const;

    SolverStatus II(std::vector<ValueType>& operand, std::vector<ValueType> const& offsets, bool relative, ValueType const& precision,
                    std::function<void(std::vector<ValueType>&)> const& prepareLowerBounds,
                    std::function<void(std::vector<ValueType>&)> const& prepareUpperBounds, std::optional<storm::OptimizationDirection> const& dir = {},
                    std::function<SolverStatus(IIData<ValueType> const&)> const& iterationCallback = {},
                    std::optional<storm::storage::BitVector> const& relevantValues = {}) const;

   private:
    std::shared_ptr<ValueIterationOperator<ValueType, TrivialRowGrouping>> viOperator;
};

}  // namespace storm::solver::helper
