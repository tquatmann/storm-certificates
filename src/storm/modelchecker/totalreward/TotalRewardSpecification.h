#pragma once

#include <vector>
#include "storm/storage/BitVector.h"
#include "storm/utility/constants.h"

namespace storm::modelchecker {

template<typename ValueType>
class TerminalStateValue {
   public:
    explicit TerminalStateValue(ValueType value) : value(value), infinite(false) {  // TODO: handle non-finite floats
        // Intentionally left empty.
    }
    static TerminalStateValue getPlusInfinity() {
        return TerminalStateValue{storm::utility::one<ValueType>(), true};
    }
    static TerminalStateValue getMinusInfinity() {
        return TerminalStateValue{-storm::utility::one<ValueType>(), true};
    }
    static TerminalStateValue getOne() {
        return TerminalStateValue(storm::utility::one<ValueType>());
    }

    bool isFinite() const {
        return !infinite;
    }

    bool isOne() const {
        return isFinite() && storm::utility::isOne<ValueType>(value);
    }
    
    bool isPositiveInfinity() const {
        return infinite && value > storm::utility::zero<ValueType>();
    }

    bool isNegativeInfinity() const {
        return infinite && value <= storm::utility::zero<ValueType>();
    }

    ValueType getFiniteValue() const {
        STORM_LOG_ASSERT(isFinite(), "Cannot get finite value from infinite terminal state value.");
        return value;
    }

   private:
    ValueType const value;
    bool const infinite;  /// if true, this represents +inf (if value is positive) and -inf (otherwise)
};

template<typename ValueType>
struct TotalRewardSpecification {
    /// Those states that are target w.r.t. the total reachability reward query
    storm::storage::BitVector terminalStates;
    /// Rewards collected upon entering terminal states (unmentioned states have reward 0)
    /// The occurring state sets must be subsets of terminalStates and pairwise disjoint.
    std::vector<std::pair<TerminalStateValue<ValueType>, storm::storage::BitVector>> terminalStateValues;
    bool terminalStatesUniversallyAlmostSurelyReached;  /// If true, the target states are reached almost surely from every state under every scheduler
};
}  // namespace storm::modelchecker