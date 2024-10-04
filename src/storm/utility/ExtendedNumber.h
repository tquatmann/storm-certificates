#include "storm/exceptions/InvalidOperationException.h"
#include "storm/solver/OptimizationDirection.h"
#include "storm/utility/constants.h"
#include "storm/utility/macros.h"

namespace storm {

template<typename T>
class Extended {
   public:
    enum class State { Finite, PosInfinity, NegInfinity };

    Extended() : value(0), state(State::Finite) {}

    Extended(T val) : value(val), state(State::Finite) {}

    static Extended posInfinity() {
        Extended inf;
        inf.state = State::PosInfinity;
        return inf;
    }

    static Extended negInfinity() {
        Extended inf;
        inf.state = State::NegInfinity;
        return inf;
    }

    bool isPositiveInfinity() const {
        return state == State::PosInfinity;
    }

    bool isNegativeInfinity() const {
        return state == State::NegInfinity;
    }

    bool isFinite() const {
        return state == State::Finite;
    }

    T const& getValue() const {
        STORM_LOG_THROW(isFinite(), storm::exceptions::InvalidOperationException, "Cannot get non-finite value");
        return value;
    }

    T& getValue() {
        STORM_LOG_THROW(isFinite(), storm::exceptions::InvalidOperationException, "Cannot get non-finite value");
        return value;
    }

    Extended operator+(Extended const& other) const {
        if (isFinite() && other.isFinite()) {
            return Extended(value + other.value);
        } else if (this->isFinite()) {
            return *other;
        } else if (other.isFinite()) {
            return *this;
        } else if (state == other.state) {
            return *this;
        }
        STORM_LOG_THROW(false, storm::exceptions::InvalidOperationException, "Operation '" << *this << " + " << other << "' is undefined.");
    }

    Extended operator-() const {
        if (isFinite()) {
            return Extended(-value);
        } else {
            return isPositiveInfinity() ? negInfinity() : posInfinity();
        }
    }

    Extended operator-(Extended const& other) const {
        if (isFinite() && other.isFinite()) {
            return Extended(value - other.value);
        } else if (this->isFinite()) {
            return -other;
        } else if (other.isFinite()) {
            return *this;
        } else if (isPositiveInfinity() && other.isNegativeInfinity()) {
            return posInfinity();
        } else if (isNegativeInfinity() && other.isPositiveInfinity()) {
            return negInfinity();
        }
        STORM_LOG_THROW(false, storm::exceptions::InvalidOperationException, "Operation '" << *this << " - " << other << "' is undefined.");
    }

    bool operator<(Extended const& other) const {
        if (isPositiveInfinity() || other.isNegativeInfinity()) {
            return false;
        }
        if (isFinite() && other.isFinite()) {
            return value < other.value;
        }
        return true;  // either '-inf < a' or 'a < inf'
    }

    bool operator<=(Extended const& other) const {
        return isNegativeInfinity() || other.isPositiveInfinity() || (isFinite() && other.isFinite() && value <= other.value);
    }

    bool operator==(Extended const& other) const {
        return state == other.state && (!isFinite() || value == other.value);
    }

    template<storm::OptimizationDirection Dir>
    Extended<T> const& optimum(Extended<T> const& other) const {
        if constexpr (storm::solver::maximize(Dir)) {
            return *this < other ? other : *this;
        } else {
            return other < *this ? other : *this;
        }
    }

    double asDouble() const {
        switch (state) {
            case State::Finite:
                return storm::utility::convertNumber<double>(value);
            case State::PosInfinity:
                return storm::utility::infinity<double>();
            case State::NegInfinity:
                return -storm::utility::infinity<double>();
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Extended& num) {
        switch (num.state) {
            case State::Finite:
                os << num.value;
                break;
            case State::PosInfinity:
                os << "inf";
                break;
            case State::NegInfinity:
                os << "-inf";
                break;
        }
        return os;
    }

   private:
    T value;
    State state;
};
}  // namespace storm
