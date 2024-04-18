#pragma once

#include "storm/storage/SparseMatrix.h"

#include "storm/utility/macros.h"

namespace storm::modelchecker::utility {

/*!
 * This class caches the backward transitions of a given forward transition matrix.
 * The forward transition matrix reference needs to remain valid throughout the lifetime of this cache.
 * The cache can be copied and moved, but the actual backward transitions are shared.
 * This means that if the matrix is computed in one of the copies, it is available at the other copies as well.
 * @tparam ValueType
 */
template<typename ValueType>
class BackwardTransitionCache {
   public:
    explicit BackwardTransitionCache(storm::storage::SparseMatrix<ValueType> const& forwardTransitionMatrix)
        : forwardTransitionMatrixRef(forwardTransitionMatrix), backwardTransitions(std::make_shared<storm::storage::SparseMatrix<ValueType>>()) {
        // Intentionally left empty.
    }

    explicit BackwardTransitionCache(storm::storage::SparseMatrix<ValueType> const& forwardTransitionMatrix,
                                     storm::storage::SparseMatrix<ValueType>&& backwardTransitionMatrix)
        : forwardTransitionMatrixRef(forwardTransitionMatrix),
          backwardTransitions(std::make_shared<storm::storage::SparseMatrix<ValueType>>(std::move(backwardTransitionMatrix))) {
        // Intentionally left empty.
    }

    BackwardTransitionCache() = delete;
    BackwardTransitionCache(BackwardTransitionCache const&) = default;
    BackwardTransitionCache(BackwardTransitionCache&&) = default;
    BackwardTransitionCache& operator=(BackwardTransitionCache const&) = default;
    BackwardTransitionCache& operator=(BackwardTransitionCache&&) = default;

    storm::storage::SparseMatrix<ValueType> const& get() const {
        if (!isCached()) {
            *this->backwardTransitions = forwardTransitionMatrixRef.transpose(true, false);  // will drop zeroes
        }
        return *backwardTransitions;
    }

    void clearCache() const {
        STORM_LOG_ASSERT(this->backwardTransitions != nullptr, "Backward transitions should never be nullptr");
        *this->backwardTransitions = storm::storage::SparseMatrix<ValueType>();
    }

    bool isCached() const {
        STORM_LOG_ASSERT(this->backwardTransitions != nullptr, "Backward transitions should never be nullptr");
        return this->backwardTransitions->getRowCount() > 0;
    }

   private:
    storm::storage::SparseMatrix<ValueType> const& forwardTransitionMatrixRef;
    mutable std::shared_ptr<storm::storage::SparseMatrix<ValueType>> backwardTransitions;
};
}  // namespace storm::modelchecker::utility